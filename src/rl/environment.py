"""
Reinforcement Learning Environment for Satellite UPF Service Optimization.
Implements the state, action, reward as described in Section IV-B.
"""

import numpy as np
import heapq
from config import Config
from src.network.satellite_network import SatelliteNetwork
from src.upf.switch_control import UPFSwitchManager, UPFSwitchState
from src.utils.user_requests import UserRequest, generate_user_requests


class SatelliteUPFEnvironment:
    """
    RL Environment for satellite UPF service optimization.
    State: State matrix M(t) with UPF features [c_u, b_sum_u, f_avg_u, g_swi_u]
    Action: Select target UPF for state migration and traffic routing
    Reward: Based on delay, energy consumption, and switch state (Eq. 27)
    """

    def __init__(self, seed=None):
        self.seed = seed if seed is not None else Config.SEED
        np.random.seed(self.seed)

        self.network = SatelliteNetwork(seed=self.seed)
        self.switch_manager = UPFSwitchManager(Config.NUM_UPFS, optimize=True)

        self.all_requests = generate_user_requests(
            Config.NUM_USER_REQUESTS, Config.NUM_SATELLITES, seed=self.seed
        )
        self.train_requests = self.all_requests[:Config.TRAIN_REQUESTS]
        self.test_requests = self.all_requests[Config.TRAIN_REQUESTS:]

        self.current_request_idx = 0
        self.current_time = 0.0
        self.requests = self.train_requests

        self.total_delay = 0.0
        self.total_resource_consumption = 0.0
        self.total_energy_consumption = 0.0
        self.accepted_requests = 0
        self.total_requests_processed = 0

        self.delay_history = []
        self.resource_history = []
        self.acceptance_history = []
        self.time_history = []
        self.cumulative_resource = 0.0

        # Track active sessions for resource release
        # Min-heap of (completion_time, sat_id, cpu_amount)
        self.active_services = []

    def set_mode(self, mode='train'):
        """Set training or testing mode."""
        if mode == 'train':
            self.requests = self.train_requests
        else:
            self.requests = self.test_requests
        self.reset()

    def reset(self):
        """Reset the environment for a new episode."""
        self.current_request_idx = 0
        self.current_time = 0.0
        self.total_delay = 0.0
        self.total_resource_consumption = 0.0
        self.total_energy_consumption = 0.0
        self.accepted_requests = 0
        self.total_requests_processed = 0

        self.delay_history = []
        self.resource_history = []
        self.acceptance_history = []
        self.time_history = []          # arrival times in ms
        self.cumulative_resource = 0.0  # cumulative resource consumption
        self.cumulative_switch_resource = 0.0  # for C1 constraint check
        self.active_services = []

        self.network.reset_resources()
        self.switch_manager.reset_all()

        return self._get_state()

    def _release_completed_services(self, current_time):
        """Release resources from sessions that have ended."""
        while (self.active_services and
               self.active_services[0][0] <= current_time):
            _, sat_id, cpu_amount = heapq.heappop(self.active_services)
            self.network.release_resources(sat_id, cpu_amount)

    def _get_state(self):
        """
        Get current state matrix M(t) of shape (U, 4) per Eq. 28.

        Features per UPF u (each element is normalized):
            [c_u(t), b_sum_u(t), f_avg_u(t), g_swi_u(t)]
          - c_u(t):     available CPU resources of the satellite hosting UPF u
          - b_sum_u(t): total ISL bandwidth at the satellite hosting UPF u
          - f_avg_u(t): average shortest distance from UPF u to all other UPFs
                        (Eq. 26)
          - g_swi_u(t): switch state of UPF u (service / standby / setup / close)
        """
        state_matrix = self.network.get_upf_state_matrix()
        state_matrix = self.switch_manager.update_state_matrix(state_matrix)
        return state_matrix

    def get_available_upfs(self):
        """Get list of UPF IDs that have enough resources to serve a request."""
        if self.current_request_idx >= len(self.requests):
            return []

        request = self.requests[self.current_request_idx]

        # Release ended sessions before checking availability
        self._release_completed_services(request.arrival_time)

        available = []
        for upf_id in range(Config.NUM_UPFS):
            sat_id = self.network.get_upf_satellite(upf_id)
            sat = self.network.satellites[sat_id]

            required_cpu = request.c_state + request.c_data
            if sat.available_cpu >= required_cpu:
                available.append(upf_id)

        return available

    def step(self, action):
        """
        Execute action (select target UPF) for current user request.
        Returns: next_state, reward, done, info
        """
        if self.current_request_idx >= len(self.requests):
            return self._get_state(), 0.0, True, {}

        request = self.requests[self.current_request_idx]
        target_upf_id = action

        # Advance simulation time and release ended sessions
        self.current_time = request.arrival_time
        self._release_completed_services(self.current_time)

        source_upf_sat = self.network.get_upf_satellite(request.source_upf_id)
        target_upf_sat = self.network.get_upf_satellite(target_upf_id)
        access_sat = request.access_sat_id
        internet_sat = request.internet_sat_id

        ctrl = self.switch_manager.get_controller(target_upf_id)
        target_sat = self.network.satellites[target_upf_sat]
        required_cpu = request.c_state + request.c_data
        accepted = target_sat.available_cpu >= required_cpu

        # gamma_u(t-1, t) per Eq. 9: 1 if the UPF transitions from "fewer than N
        # accumulated requests" to "at least N" via this arrival. We track an
        # accumulated request counter while the UPF is closed/setup; if this
        # arrival pushes it across N, the UPF is switched on (gamma = 1).
        gamma = 0

        if accepted:
            self.network.consume_resources(target_upf_sat, required_cpu)

            # Schedule resource release after session ends
            session_duration = np.random.exponential(Config.SESSION_DURATION_MEAN)
            completion_time = self.current_time + session_duration
            heapq.heappush(self.active_services,
                (completion_time, target_upf_sat, required_cpu)
            )

            migration_path, migration_delay = (
                self.network.bfs_shortest_delay_path_with_transmission(
                    source_upf_sat, target_upf_sat, request.v_state
                )
            )

            routing_path_1, routing_delay_1 = (
                self.network.bfs_shortest_delay_path_with_transmission(
                    access_sat, target_upf_sat, request.v_data
                )
            )
            routing_path_2, routing_delay_2 = (
                self.network.bfs_shortest_delay_path_with_transmission(
                    target_upf_sat, internet_sat, request.v_data
                )
            )

            routing_delay = routing_delay_1 + routing_delay_2
            switch_delay = ctrl.get_additional_delay()

            # Eq. 9: gamma_u(t-1, t) = 1 iff prev queued < N AND queued+1 >= N
            prev_queued = ctrl.queued_requests
            if not ctrl.is_active():
                if prev_queued < ctrl.N and (prev_queued + 1) >= ctrl.N:
                    gamma = 1
                    ctrl.state = UPFSwitchState.SERVICE

            total_delay = (migration_delay + routing_delay + switch_delay) * 1000

            energy = self._compute_energy(migration_path, request.v_state,
                                          routing_path_1 + routing_path_2[1:],
                                          request.v_data)

            request.target_upf_id = target_upf_id
            request.migration_path = migration_path
            request.routing_path = routing_path_1 + routing_path_2[1:]
            request.total_delay = total_delay
            request.energy_consumption = energy
            request.accepted = True

            self.accepted_requests += 1
        else:
            total_delay = 0
            energy = 0
            request.accepted = False

        self.total_requests_processed += 1

        self.total_delay += total_delay
        self.total_energy_consumption += energy

        # Track cumulative resource: CPU consumed + switch resource per request
        if accepted:
            switch_resource = ctrl.get_resource_consumption()
            self.cumulative_resource += required_cpu + switch_resource

        self.delay_history.append(self.total_delay)
        self.resource_history.append(self.cumulative_resource)
        acceptance_rate = self.accepted_requests / self.total_requests_processed
        self.acceptance_history.append(acceptance_rate)
        self.time_history.append(self.current_time * 1000)  # store in ms

        # Constraints C1 (switch resource <= phi1) and C2 (energy <= phi2).
        # Track cumulative violation as a soft penalty applied to the reward.
        switch_resource_now = ctrl.get_resource_consumption() if accepted else 0.0
        self.cumulative_switch_resource += switch_resource_now if gamma == 1 else 0.0
        constraint_penalty = 0.0
        if self.cumulative_switch_resource > Config.PHI1:
            constraint_penalty += 0.1
        if self.total_energy_consumption > Config.PHI2:
            constraint_penalty += 0.1

        reward = self._compute_reward(
            total_delay, energy, gamma, accepted,
            switch_resource=switch_resource_now,
        ) - constraint_penalty

        dt = 1.0 / Config.AVG_ARRIVAL_RATE
        for upf_id in range(Config.NUM_UPFS):
            arrivals = 1 if upf_id == target_upf_id and accepted else 0
            self.switch_manager.get_controller(upf_id).update_state(arrivals, dt)

        self.current_request_idx += 1

        time_slot = int(self.current_request_idx /
                       (len(self.requests) / Config.NUM_TIME_SLOTS))
        if (self.current_request_idx % (len(self.requests) // Config.NUM_TIME_SLOTS) == 0
                and time_slot < Config.NUM_TIME_SLOTS):
            self.network.update_topology(time_slot)

        done = self.current_request_idx >= len(self.requests)
        next_state = self._get_state()

        info = {
            'delay': total_delay,
            'energy': energy,
            'accepted': accepted,
            'total_delay': self.total_delay,
            'acceptance_rate': acceptance_rate,
            'gamma': gamma
        }

        return next_state, reward, done, info

    def _compute_reward(self, delay, energy, gamma, accepted=True,
                        switch_resource=0.0):
        """
        Per-request reward following Eq. 27.

        Paper formulation (HIGHER = BETTER):
            R(t) = eta1 / (sum_t sum_r D_r(t)) + eta2 / (sum_t sum_r W_r(t))
                   [- eta3 * sum_t sum_u C_swi_u(t)   if any gamma_u(t-1, t)=1]

        We approximate this online using the per-request quantities D_r and W_r.
        Inverse delay/energy mean a small delay yields a large reward, matching
        the paper's "large or positive rewards encourage effective actions".
        Rejected requests receive a strongly negative reward.
        """
        if not accepted:
            return -1.0

        # Inverse-delay / inverse-energy reward (paper Eq. 27 form), with
        # small constants to keep magnitudes bounded for normalized inputs.
        delay_term = Config.ETA1 / (delay + 1.0)        # ms
        energy_term = Config.ETA2 / (energy + 1e-3)     # joules-ish

        reward = delay_term + energy_term

        if gamma == 1:
            # UPF was just switched on; subtract switch resource penalty.
            reward -= Config.ETA3 * (switch_resource / Config.CPU_SERVICE)

        return reward

    def _compute_energy(self, state_path, v_state, data_path, v_data):
        """Compute energy consumption (Eq. 8)."""
        energy = 0.0

        for i in range(len(state_path) - 1):
            s = state_path[i]
            s_next = state_path[i + 1]
            bw = self.network.get_link_bandwidth(s, s_next)
            if bw > 0:
                trans_time = (v_state / 1000.0) / bw
                energy += Config.TRANSMIT_POWER * trans_time

        for i in range(len(data_path) - 1):
            s = data_path[i]
            s_next = data_path[i + 1]
            bw = self.network.get_link_bandwidth(s, s_next)
            if bw > 0:
                trans_time = (v_data / 1000.0) / bw
                energy += Config.TRANSMIT_POWER * trans_time

        return energy

    def _compute_total_resource(self):
        """Compute total resource consumption across all satellites."""
        total = 0.0
        for sat in self.network.satellites.values():
            used = sat.cpu_capacity - sat.available_cpu
            total += used

        for ctrl in self.switch_manager.controllers.values():
            total += ctrl.get_resource_consumption()

        return total

    def get_current_request(self):
        """Get the current user request being processed."""
        if self.current_request_idx < len(self.requests):
            return self.requests[self.current_request_idx]
        return None
