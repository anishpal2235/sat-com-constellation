"""
Algorithm 2: State-Aware Satellite UPF Service Optimization Algorithm.
Integrates switch control (Algorithm 1), state migration, and traffic routing.
"""

import numpy as np

try:
    import torch
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config import Config
from src.rl.policy_network import PolicyNetworkTorch, create_policy_network
from src.rl.environment import SatelliteUPFEnvironment


class SatelliteUPFOptimizer:
    """
    State-Aware Satellite UPF Service Optimization (Algorithm 2).

    Offline training:
      1. Set switch control parameters (Algorithm 1)
      2. Train policy network using policy gradients
    Online testing:
      1. Use trained policy network to select target UPFs
      2. BFS shortest-delay path for state migration and traffic routing
    """

    def __init__(self, env=None, seed=None):
        self.seed = seed if seed is not None else Config.SEED
        np.random.seed(self.seed)

        if env is None:
            self.env = SatelliteUPFEnvironment(seed=self.seed)
        else:
            self.env = env

        if TORCH_AVAILABLE:
            # 4 features per UPF per Eq. 28: [c_u, b_sum_u, f_avg_u, g_swi_u]
            self.policy_net = PolicyNetworkTorch(
                num_upfs=Config.NUM_UPFS, feature_dim=4, hidden_dim=64
            )
            self.optimizer = optim.Adam(
                self.policy_net.parameters(), lr=Config.LEARNING_RATE
            )
        else:
            self.policy_net = create_policy_network(Config.NUM_UPFS)
            self.optimizer = None

        self.loss_history = []
        self.reward_history = []
        self.name = "Proposed"

    def train(self, num_epochs=None, batch_size=None, verbose=True):
        """Offline training phase of Algorithm 2."""
        if num_epochs is None:
            num_epochs = Config.NUM_EPOCHS
        if batch_size is None:
            batch_size = Config.BATCH_SIZE

        if not TORCH_AVAILABLE:
            return self._train_tf(num_epochs, batch_size, verbose)

        self.env.set_mode('train')

        if verbose:
            print(f"Starting training: {num_epochs} epochs, "
                  f"{len(self.env.requests)} requests")

        for epoch in range(num_epochs):
            state = self.env.reset()
            epoch_loss = 0.0
            epoch_reward = 0.0
            batch_states = []
            batch_actions = []
            batch_masks = []
            batch_rewards = []
            num_steps = 0

            for r_idx in range(len(self.env.requests)):
                state_matrix = state

                available_upfs = self.env.get_available_upfs()
                if not available_upfs:
                    next_state, reward, done, info = self.env.step(0)
                    state = next_state
                    if done:
                        break
                    continue

                available_mask = np.zeros(Config.NUM_UPFS)
                for upf_id in available_upfs:
                    available_mask[upf_id] = 1.0

                action, log_prob = self.policy_net.select_action(
                    state_matrix, available_mask, deterministic=False
                )

                next_state, reward, done, info = self.env.step(action)

                batch_states.append(state_matrix.copy())
                batch_actions.append(action)
                batch_masks.append(available_mask.copy())
                batch_rewards.append(reward)
                epoch_reward += reward
                num_steps += 1

                if len(batch_states) >= batch_size:
                    loss = self._update_policy(
                        batch_states, batch_actions, batch_masks, batch_rewards
                    )
                    epoch_loss += loss
                    batch_states = []
                    batch_actions = []
                    batch_masks = []
                    batch_rewards = []

                state = next_state
                if done:
                    break

            if batch_states:
                loss = self._update_policy(
                    batch_states, batch_actions, batch_masks, batch_rewards
                )
                epoch_loss += loss

            avg_loss = epoch_loss / max(num_steps // batch_size, 1)
            self.loss_history.append(avg_loss)
            self.reward_history.append(epoch_reward / max(num_steps, 1))

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}: "
                      f"Loss={avg_loss:.4f}, "
                      f"Avg Reward={epoch_reward/num_steps:.4f}, "
                      f"Acceptance={self.env.accepted_requests}/"
                      f"{self.env.total_requests_processed}")

        if verbose:
            print("Training complete.")

    def _update_policy(self, states, actions, masks, rewards):
        """
        Update policy network per Algorithm 2 (lines 12-13) using the
        cross-entropy loss in Eq. 29:
            L(h, o) = - sum_u h_u * log o_u
        where h_u is a one-hot label at the chosen target UPF and o_u is the
        softmax output. Per the paper's Section V-B, the chosen action is
        treated as the optimal label and parameters are updated via
        backpropagation.

        We multiply the cross-entropy by the (baseline-subtracted, normalized)
        reward from Eq. 27 so that high-reward actions have their probability
        increased and low-reward actions decreased - this is the standard
        REINFORCE-with-baseline interpretation of the paper's procedure.
        """
        if not TORCH_AVAILABLE:
            return 0.0

        device = self.policy_net.device

        # Stack batch into tensors
        state_tensor = torch.FloatTensor(np.array(states)).to(device)
        mask_tensor = torch.FloatTensor(np.array(masks)).to(device)
        action_tensor = torch.LongTensor(actions).to(device)
        reward_tensor = torch.FloatTensor(rewards).to(device)

        # REINFORCE advantage: subtract batch-mean baseline, then normalize.
        # Higher reward => more positive advantage => probability of that
        # action is increased.
        advantage = reward_tensor - reward_tensor.mean()
        if len(advantage) > 1:
            adv_std = advantage.std()
            if adv_std > 1e-8:
                advantage = advantage / adv_std

        # Forward pass: probabilities o_u for all UPFs.
        probs = self.policy_net(state_tensor, mask_tensor)  # (batch, U)

        # Eq. 29: cross-entropy loss with one-hot label h at the chosen action.
        log_probs = torch.log(probs + 1e-8)
        ce_loss = -log_probs[range(len(actions)), action_tensor]

        # Final REINFORCE objective: minimize -advantage * log pi(a|s)
        # (equivalent to advantage-weighted cross-entropy).
        weighted_loss = -(advantage * (-ce_loss)).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Return raw cross-entropy magnitude for plotting (Fig. 5 shows the
        # cross-entropy of Eq. 29, not the policy-gradient objective).
        return ce_loss.detach().mean().item()

    def test(self, verbose=True):
        """Online testing phase of Algorithm 2."""
        self.env.set_mode('test')
        state = self.env.reset()

        if verbose:
            print(f"Starting testing: {len(self.env.requests)} requests")

        for r_idx in range(len(self.env.requests)):
            state_matrix = state

            available_upfs = self.env.get_available_upfs()
            if not available_upfs:
                next_state, reward, done, info = self.env.step(0)
                state = next_state
                if done:
                    break
                continue

            available_mask = np.zeros(Config.NUM_UPFS)
            for upf_id in available_upfs:
                available_mask[upf_id] = 1.0

            action, _ = self.policy_net.select_action(
                state_matrix, available_mask, deterministic=True
            )

            next_state, reward, done, info = self.env.step(action)
            state = next_state
            if done:
                break

        results = {
            'total_delay': self.env.total_delay,
            'total_resource': self.env.cumulative_resource,
            'total_energy': self.env.total_energy_consumption,
            'acceptance_rate': (self.env.accepted_requests /
                               max(self.env.total_requests_processed, 1)),
            'accepted': self.env.accepted_requests,
            'total_processed': self.env.total_requests_processed,
            'delay_history': list(self.env.delay_history),
            'resource_history': list(self.env.resource_history),
            'acceptance_history': list(self.env.acceptance_history),
            'time_history': list(self.env.time_history),
        }

        if verbose:
            print(f"Testing complete: "
                  f"Delay={results['total_delay']:.2f}ms, "
                  f"Acceptance={results['acceptance_rate']:.4f}")

        return results

    def _train_tf(self, num_epochs, batch_size, verbose):
        """Simplified training for TensorFlow backend."""
        self.env.set_mode('train')

        for epoch in range(num_epochs):
            state = self.env.reset()
            epoch_reward = 0.0

            for r_idx in range(len(self.env.requests)):
                available_upfs = self.env.get_available_upfs()
                available_mask = np.zeros(Config.NUM_UPFS)
                for upf_id in available_upfs:
                    available_mask[upf_id] = 1.0

                action, log_prob = self.policy_net.select_action(
                    state, available_mask, deterministic=False
                )
                next_state, reward, done, info = self.env.step(action)
                epoch_reward += reward
                state = next_state
                if done:
                    break

            self.loss_history.append(-epoch_reward / len(self.env.requests))
            self.reward_history.append(epoch_reward / len(self.env.requests))

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}: "
                      f"Avg Reward={epoch_reward/len(self.env.requests):.4f}")


def run_baseline_evaluation(env, baseline_algo, verbose=True):
    """Run evaluation for a baseline algorithm."""
    env.set_mode('test')
    state = env.reset()

    if verbose:
        print(f"Evaluating {baseline_algo.name}: {len(env.requests)} requests")

    for r_idx in range(len(env.requests)):
        request = env.get_current_request()
        if request is None:
            break

        available_upfs = env.get_available_upfs()
        if not available_upfs:
            next_state, reward, done, info = env.step(0)
            state = next_state
            if done:
                break
            continue

        action = baseline_algo.select_target_upf(request, available_upfs)

        next_state, reward, done, info = env.step(action)
        state = next_state
        if done:
            break

    results = {
        'total_delay': env.total_delay,
        'total_resource': env.cumulative_resource,
        'total_energy': env.total_energy_consumption,
        'acceptance_rate': env.accepted_requests / max(env.total_requests_processed, 1),
        'accepted': env.accepted_requests,
        'total_processed': env.total_requests_processed,
        'delay_history': list(env.delay_history),
        'resource_history': list(env.resource_history),
        'acceptance_history': list(env.acceptance_history),
        'time_history': list(env.time_history),
    }

    if verbose:
        print(f"  {baseline_algo.name}: Delay={results['total_delay']:.2f}ms, "
              f"Acceptance={results['acceptance_rate']:.4f}")

    return results
