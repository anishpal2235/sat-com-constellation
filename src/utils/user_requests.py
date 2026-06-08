"""
User Request generation and management.
Each request has state data packets, traffic data packets, and resource demands.
"""

import numpy as np
from config import Config


class UserRequest:
    """Represents a single user request with state and data traffic."""

    def __init__(self, request_id, arrival_time, access_sat_id,
                 v_state, v_data, c_state, c_data):
        self.request_id = request_id
        self.arrival_time = arrival_time
        self.access_sat_id = access_sat_id

        self.v_state = v_state
        self.v_data = v_data
        self.c_state = c_state
        self.c_data = c_data

        self.source_upf_id = None
        self.target_upf_id = None
        self.migration_path = []
        self.routing_path = []
        self.internet_sat_id = None

        self.total_delay = 0.0
        self.energy_consumption = 0.0
        self.accepted = False


def generate_user_requests(num_requests, num_satellites, seed=None):
    """Generate a set of user requests with random parameters."""
    if seed is not None:
        np.random.seed(seed)

    requests = []
    inter_arrival_times = np.random.exponential(
        1.0 / Config.AVG_ARRIVAL_RATE, num_requests
    )
    arrival_times = np.cumsum(inter_arrival_times)

    for i in range(num_requests):
        access_sat = np.random.randint(0, num_satellites)

        v_state = np.random.uniform(
            Config.STATE_PACKET_SIZE_MIN, Config.STATE_PACKET_SIZE_MAX
        )
        v_data = np.random.uniform(
            Config.DATA_PACKET_SIZE_MIN, Config.DATA_PACKET_SIZE_MAX
        )

        c_state = np.random.uniform(Config.CPU_STATE_MIN, Config.CPU_STATE_MAX)
        c_data = np.random.uniform(Config.CPU_DATA_MIN, Config.CPU_DATA_MAX)

        internet_sat = np.random.randint(0, num_satellites)
        while internet_sat == access_sat:
            internet_sat = np.random.randint(0, num_satellites)

        req = UserRequest(
            request_id=i,
            arrival_time=arrival_times[i],
            access_sat_id=access_sat,
            v_state=v_state,
            v_data=v_data,
            c_state=c_state,
            c_data=c_data
        )
        req.internet_sat_id = internet_sat
        req.source_upf_id = np.random.randint(0, Config.NUM_UPFS)

        requests.append(req)

    return requests
