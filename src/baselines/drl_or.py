"""
DRL-OR: Deep Reinforcement Learning-based Online Routing.
Deploys an agent on each routing node for next-hop decisions.
Reference: [44] in the paper.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config import Config


class DRLORNetwork(nn.Module):
    """Per-node agent network for next-hop routing decisions."""

    def __init__(self, input_dim, max_neighbors=4, hidden_dim=64):
        super(DRLORNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, max_neighbors)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


class DRLORAlgorithm:
    """
    DRL-OR: Deep RL-based Online Routing for multiple agents.
    Each routing node has an agent making next-hop decisions.
    """

    def __init__(self, network, hidden_dim=64):
        self.network = network
        self.name = "DRL-OR"
        self.max_neighbors = 4

        if TORCH_AVAILABLE:
            self.device = torch.device(Config.get_device())
            self.input_dim = 4 + self.max_neighbors
            self.agents = {}
            for sat_id in range(Config.NUM_SATELLITES):
                self.agents[sat_id] = DRLORNetwork(
                    self.input_dim, self.max_neighbors, hidden_dim
                ).to(self.device)
            self.trained = False

    def _get_local_state(self, sat_id):
        """Get local state for a satellite node."""
        sat = self.network.satellites[sat_id]
        neighbors = self.network.adjacency[sat_id]

        cpu_norm = sat.available_cpu / sat.cpu_capacity
        total_bw = sum(bw for _, bw in neighbors)
        max_bw = self.max_neighbors * Config.BANDWIDTH_MAX
        bw_norm = total_bw / max_bw if max_bw > 0 else 0

        neighbor_bws = [bw / Config.BANDWIDTH_MAX for _, bw in neighbors]
        while len(neighbor_bws) < self.max_neighbors:
            neighbor_bws.append(0.0)
        neighbor_bws = neighbor_bws[:self.max_neighbors]

        state = [cpu_norm, bw_norm, 0.5, 0.5] + neighbor_bws
        return np.array(state, dtype=np.float32)

    def select_target_upf(self, request, available_upfs=None):
        """
        Select target UPF for DRL-OR.
        Paper's DRL-OR [44] deploys per-node agents for next-hop ROUTING
        decisions, NOT global UPF selection. UPF selection uses simple
        hop-based proximity from source UPF (similar to Greedy).
        DRL-OR's advantage in the paper is routing quality, not UPF selection.
        """
        candidates = list(available_upfs) if available_upfs else list(range(Config.NUM_UPFS))
        source_sat = self.network.get_upf_satellite(request.source_upf_id)

        best_upf = None
        best_hops = float('inf')

        for upf_id in candidates:
            target_sat_id = self.network.get_upf_satellite(upf_id)
            path, _ = self.network.bfs_shortest_delay_path(source_sat, target_sat_id)
            hops = len(path) - 1 if path else float('inf')
            if hops < best_hops:
                best_hops = hops
                best_upf = upf_id

        return best_upf if best_upf is not None else np.random.choice(candidates)

    def _route_to_target(self, source, destination, max_hops=15):
        """Route from source to destination using per-hop agent decisions."""
        current = source
        path = [current]
        total_delay = 0.0
        visited = set([current])

        for _ in range(max_hops):
            if current == destination:
                break

            neighbors = self.network.adjacency[current]
            if not neighbors:
                break

            local_state = self._get_local_state(current)
            state_tensor = torch.FloatTensor(local_state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                probs = self.agents[current](state_tensor).squeeze()

            valid_neighbors = [(n, bw) for n, bw in neighbors if n not in visited]
            if not valid_neighbors:
                break

            best_idx = 0
            best_score = -float('inf')
            for i, (n, bw) in enumerate(valid_neighbors):
                dist = self.network.get_distance(n, destination)
                nn_score = probs[i % len(probs)].item() if i < len(probs) else 0
                score = nn_score - dist / 50000.0
                if score > best_score:
                    best_score = score
                    best_idx = i

            next_sat, bw = valid_neighbors[best_idx]
            sat_curr = self.network.satellites[current]
            sat_next = self.network.satellites[next_sat]
            if sat_curr.orbit_plane == sat_next.orbit_plane:
                link_dist = Config.INTRA_PLANE_DISTANCE
            else:
                link_dist = Config.INTER_PLANE_DISTANCE

            total_delay += link_dist / Config.SPEED_OF_LIGHT
            visited.add(next_sat)
            path.append(next_sat)
            current = next_sat

        if current != destination:
            total_delay = float('inf')

        return path, total_delay

    def _heuristic_select(self, request, candidates):
        """Heuristic fallback for DRL-OR."""
        access_sat = request.access_sat_id
        best_upf = None
        best_dist = float('inf')

        for upf_id in candidates:
            target_sat = self.network.get_upf_satellite(upf_id)
            dist = self.network.get_distance(access_sat, target_sat)
            dist += np.random.uniform(0, 2000)
            if dist < best_dist:
                best_dist = dist
                best_upf = upf_id

        return best_upf if best_upf is not None else np.random.choice(candidates)

    def train_agents(self, requests, network, episodes=50):
        """Train the per-node DRL-OR agents using per-hop REINFORCE."""
        if not TORCH_AVAILABLE:
            return

        optimizers = {
            sat_id: optim.Adam(agent.parameters(), lr=1e-4)
            for sat_id, agent in self.agents.items()
        }

        for episode in range(episodes):
            for req in requests[:200]:
                # Sample a random target UPF and route to it
                upf_id = np.random.randint(0, Config.NUM_UPFS)
                target_sat = network.get_upf_satellite(upf_id)
                path, delay = self._route_to_target(
                    req.access_sat_id, target_sat
                )
                if delay < float('inf') and len(path) > 1:
                    # Per-hop credit: shorter paths get higher reward
                    reward = 1.0 / (1.0 + delay)
                    for i in range(len(path) - 1):
                        sat = path[i]
                        local_state = self._get_local_state(sat)
                        state_tensor = torch.FloatTensor(local_state).unsqueeze(0).to(self.device)
                        probs = self.agents[sat](state_tensor)
                        # Use the actual neighbor index chosen
                        neighbors = [n for n, _ in network.adjacency[sat]]
                        next_sat = path[i + 1]
                        if next_sat in neighbors:
                            idx = neighbors.index(next_sat) % self.max_neighbors
                            log_prob = torch.log(probs[0, idx] + 1e-8)
                            loss = -log_prob * reward
                            optimizers[sat].zero_grad()
                            loss.backward()
                            optimizers[sat].step()

        self.trained = True
