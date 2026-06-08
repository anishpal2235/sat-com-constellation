"""
Greedy Algorithm: Always migrate to the satellite UPF with
the fewest inter-satellite link hops from the source UPF.
"""

import numpy as np
from config import Config


class GreedyAlgorithm:
    def __init__(self, network):
        self.network = network
        self.name = "Greedy"

    def select_target_upf(self, request, available_upfs=None):
        """Select the nearest UPF by hop count from source UPF."""
        source_sat = self.network.get_upf_satellite(request.source_upf_id) #where the source UPF is deployed
        min_hops = float('inf')
        best_upf = None

        candidates = available_upfs if available_upfs else range(Config.NUM_UPFS)

        for upf_id in candidates:
            target_sat = self.network.get_upf_satellite(upf_id)
            if target_sat == source_sat:
                # Same satellite = 0 hops, always wins trivially
                # Greedy picks this if available (no migration needed)
                return upf_id
            path, _ = self.network.bfs_shortest_delay_path(source_sat, target_sat) #breadth first search
            hops = len(path) - 1 if path else float('inf') #edges = no. of nodes - 1

            if hops < min_hops:
                min_hops = hops
                best_upf = upf_id

        if best_upf is None:
            best_upf = np.random.choice(
                list(candidates) if available_upfs else list(range(Config.NUM_UPFS))
            )

        return best_upf
