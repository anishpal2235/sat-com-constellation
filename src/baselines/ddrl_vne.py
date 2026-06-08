"""
DDRL-VNE: Delay-sensitive Virtual Network Embedding based on Deep RL.
Reference: [45] in the paper.

Key characteristics:
- Selects target node closest to source (ignoring load)
- Uses hop-based shortest path for routing
- Considers link delays but not node load
"""

import numpy as np
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config import Config


class DDRLVNENetwork(nn.Module):
    """Neural network for DDRL-VNE algorithm."""

    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DDRLVNENetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


class DDRLVNEAlgorithm:
    """DDRL-VNE (labeled RLQVNE in paper figures)."""

    def __init__(self, network, hidden_dim=128):
        self.network = network
        self.name = "RLQVNE"

        if TORCH_AVAILABLE:
            self.device = torch.device(Config.get_device())
            self.input_dim = 4 + Config.NUM_UPFS
            self.model = DDRLVNENetwork(
                self.input_dim, Config.NUM_UPFS, hidden_dim
            ).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
            self.trained = False

    def _get_state(self, request):
        """Get state representation for the request."""
        access_sat = request.access_sat_id
        sat = self.network.satellites[access_sat]

        cpu_norm = sat.available_cpu / sat.cpu_capacity
        total_bw = self.network.get_total_bandwidth_for_sat(access_sat)
        max_bw = 4 * Config.BANDWIDTH_MAX
        bw_norm = total_bw / max_bw if max_bw > 0 else 0

        features = [cpu_norm, bw_norm, request.v_state / Config.STATE_PACKET_SIZE_MAX,
                    request.v_data / Config.DATA_PACKET_SIZE_MAX]

        max_dist = Config.INTRA_PLANE_DISTANCE * Config.SATS_PER_PLANE
        for upf_id in range(Config.NUM_UPFS):
            upf_sat = self.network.get_upf_satellite(upf_id)
            dist = self.network.get_distance(access_sat, upf_sat)
            features.append(min(dist / max_dist, 1.0) if dist < float('inf') else 1.0)

        return np.array(features, dtype=np.float32)

    def select_target_upf(self, request, available_upfs=None):
        """Select target UPF prioritizing proximity to source."""
        candidates = list(available_upfs) if available_upfs else list(range(Config.NUM_UPFS))

        if TORCH_AVAILABLE and self.trained:
            state = self._get_state(request)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                probs = self.model(state_tensor).squeeze()

            mask = torch.zeros(Config.NUM_UPFS, device=self.device)
            for upf_id in candidates:
                mask[upf_id] = 1.0
            probs = probs * mask
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = mask / mask.sum()

            action = torch.argmax(probs).item()
            return action

        # Heuristic: select closest UPF (ignoring load)
        access_sat = request.access_sat_id
        best_upf = None
        best_dist = float('inf')

        for upf_id in candidates:
            target_sat = self.network.get_upf_satellite(upf_id)
            dist = self.network.get_distance(access_sat, target_sat)
            if dist < best_dist:
                best_dist = dist
                best_upf = upf_id

        return best_upf if best_upf is not None else np.random.choice(candidates)

    def train(self, requests, network, episodes=50):
        """Train the DDRL-VNE model to minimize delay (ignoring load)."""
        if not TORCH_AVAILABLE:
            return

        baseline = 0.0
        for episode in range(episodes):
            for req in requests[:200]:
                state = self._get_state(req)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                probs = self.model(state_tensor)

                dist_cat = torch.distributions.Categorical(probs.squeeze())
                action = dist_cat.sample()

                target_sat = network.get_upf_satellite(action.item())
                path, delay = network.bfs_shortest_delay_path(
                    req.access_sat_id, target_sat
                )
                # Reward: high for low delay, bounded
                if delay < float('inf'):
                    reward = 1.0 / (1.0 + delay * 100)
                else:
                    reward = -1.0

                advantage = reward - baseline
                baseline = 0.99 * baseline + 0.01 * reward

                log_prob = dist_cat.log_prob(action)
                loss = -log_prob * advantage

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

        self.trained = True
