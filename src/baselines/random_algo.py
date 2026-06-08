"""
Random Algorithm: Randomly select a satellite UPF from available UPFs
and use shortest delay path for migration/routing.
"""

import numpy as np
from config import Config


class RandomAlgorithm:
    def __init__(self, network):
        self.network = network
        self.name = "Random"

    def select_target_upf(self, request, available_upfs=None):
        """Randomly select a target UPF."""
        candidates = list(available_upfs) if available_upfs else list(range(Config.NUM_UPFS))
        if request.source_upf_id in candidates and len(candidates) > 1:
            candidates.remove(request.source_upf_id)
        return np.random.choice(candidates)
