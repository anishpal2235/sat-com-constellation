"""
Configuration parameters for Satellite UPF Service Optimization.
Based on Table I from the paper.
"""

import numpy as np


class Config:
    # Constellation parameters
    NUM_SATELLITES = 66          # S: Total number of satellites
    NUM_ORBIT_PLANES = 6         # Number of orbital planes
    SATS_PER_PLANE = 11          # Satellites per orbital plane(66/6)
    NUM_UPFS = 36                # U: Number of UPFs (6 per orbital plane)
    UPFS_PER_PLANE = 6           # UPFs deployed per orbital plane(36/6)

    # Satellite resources
    CPU_SATELLITES_MIN = 100     # TFLOPS
    CPU_SATELLITES_MAX = 200     # TFLOPS

    # Inter-satellite link bandwidth
    BANDWIDTH_MIN = 1            # Gbps
    BANDWIDTH_MAX = 10           # Gbps

    # User requests
    NUM_USER_REQUESTS = 6000     # R: Total user requests
    TRAIN_REQUESTS = 3000        # Training set size
    TEST_REQUESTS = 3000         # Test set size
    AVG_ARRIVAL_RATE = 40        # lambda: average arrival rate per second

    # CPU requirements for state/data packets
    CPU_STATE_MIN = 1            # TFLOPS
    CPU_STATE_MAX = 10           # TFLOPS
    CPU_DATA_MIN = 1             # TFLOPS
    CPU_DATA_MAX = 50            # TFLOPS

    # Packet sizes
    STATE_PACKET_SIZE_MIN = 0.05  # Mb
    STATE_PACKET_SIZE_MAX = 0.5   # Mb
    DATA_PACKET_SIZE_MIN = 0.5    # Mb
    DATA_PACKET_SIZE_MAX = 5.0    # Mb

    # UPF switch control parameters
    CPU_SERVICE = 30             # c_u^ser: TFLOPS during service stage
    CPU_STANDBY = 20             # c_u^sta: TFLOPS during standby stage
    CPU_SETUP = 20               # c_u^set: TFLOPS during setup stage
    CPU_CLOSE = 5                # c_u^clo: TFLOPS during close stage
    SERVICE_DURATION_MEAN = 5e-3 # A(t_u^ser): mean service time in seconds (5 ms)
    SESSION_DURATION_MEAN = 4.5  # Mean session/resource holding duration (seconds)

    # Orbital parameters (Iridium-like)
    ORBIT_ALTITUDE = 781         # km (Iridium altitude)
    EARTH_RADIUS = 6371          # km
    ORBIT_PERIOD = 6024          # seconds (~100 min for Iridium)
    SATELLITE_SPEED = 7.5        # km/s
    SPEED_OF_LIGHT = 299792      # km/s (for propagation delay)

    # Inter-satellite link distances (approximate for Iridium)
    INTRA_PLANE_DISTANCE = 4080  # km (same orbit, adjacent)
    INTER_PLANE_DISTANCE = 4400  # km (adjacent orbit)

    # Topology stability window
    TOPOLOGY_STABLE_WINDOW = 860  # seconds (from Fig. 4)

    # RL Training parameters
    NUM_EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    GAMMA = 0.99                 # Discount factor

    # Reward weights
    ETA1 = 0.4                   # Weight for delay
    ETA2 = 0.3                   # Weight for energy consumption
    ETA3 = 0.3                   # Weight for switch resource penalty

    # Switch control parameters (initial, will be optimized by Algorithm 1)
    N_THRESHOLD = 3              # N-limited scheme threshold
    STANDBY_DURATION = 50e-3     # t_sta_u in seconds (50 ms)
    SETUP_DURATION_MEAN = 5e-3   # A(t_set_u) in seconds
    SETUP_DURATION_VAR_COEFF = 0.1   # S(t_set_u)^2 (Eq. 10)
    SERVICE_DURATION_VAR_COEFF = 0.1 # S(t_ser_u)^2 (separate from setup)

    # Resource/energy thresholds (constraints C1, C2 in Eq. 17)
    PHI1 = 1e7                   # Switch resource consumption threshold C^swi <= phi1
    PHI2 = 1e7                   # Energy consumption threshold W <= phi2

    # Transmit power
    TRANSMIT_POWER = 10          # Watts per satellite

    # Simulation time parameters
    NUM_TIME_SLOTS = 4           # Number of topology snapshots
    TIME_SLOT_DURATION = 860     # seconds per slot

    # Device: "auto" (GPU if available), "cuda", or "cpu"
    DEVICE = "auto"

    # Random seed
    SEED = 42

    @staticmethod
    def get_device():
        """Resolve the compute device (CPU or GPU)."""
        try:
            import torch
        except ImportError:
            return "cpu"
        if Config.DEVICE == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return Config.DEVICE

    @staticmethod
    def get_satellite_cpu():
        """Generate random CPU capacity for each satellite."""
        np.random.seed(Config.SEED)
        return np.random.uniform(
            Config.CPU_SATELLITES_MIN,
            Config.CPU_SATELLITES_MAX,
            Config.NUM_SATELLITES
        )

    @staticmethod
    def get_bandwidth():
        """Generate random bandwidth for inter-satellite links."""
        return np.random.uniform(
            Config.BANDWIDTH_MIN,
            Config.BANDWIDTH_MAX
        )
