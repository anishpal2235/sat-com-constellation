"""
UPF Switch Control with M/G/1 Queue Model and N-limited scheme.
Implements the delay-resource consumption trade-off analysis (Section IV-A)
and Algorithm 1: Dynamic Adjustment of Switch Control Parameters.
"""

import numpy as np
from config import Config


class UPFSwitchState:
    """Possible states of a satellite UPF."""
    SERVICE = "service"
    STANDBY = "standby"
    CLOSE = "close"
    SETUP = "setup"


class UPFSwitchController:
    """
    Controls the switch state of a satellite UPF using the N-limited scheme.
    Models the switch control process as an M/G/1 queue.
    """

    def __init__(self, upf_id, N=None, t_sta=None):
        self.upf_id = upf_id
        self.N = N if N is not None else Config.N_THRESHOLD
        self.t_sta = t_sta if t_sta is not None else Config.STANDBY_DURATION

        # Queue model parameters (M/G/1 - Section IV-A)
        self.lambda_rate = Config.AVG_ARRIVAL_RATE
        self.A_ser = Config.SERVICE_DURATION_MEAN
        self.A_set = Config.SETUP_DURATION_MEAN
        # Eq. 10: S(t)^2 squared coefficient of variation. Service and setup
        # have separate distributions per the paper's general (G) assumption.
        self.S_set_sq = Config.SETUP_DURATION_VAR_COEFF
        self.S_ser_sq = Config.SERVICE_DURATION_VAR_COEFF

        # Resource consumption at each stage
        self.c_ser = Config.CPU_SERVICE
        self.c_sta = Config.CPU_STANDBY
        self.c_set = Config.CPU_SETUP
        self.c_clo = Config.CPU_CLOSE

        # Current state
        self.state = UPFSwitchState.SERVICE
        self.queued_requests = 0
        self.standby_timer = 0.0
        self.setup_timer = 0.0

        # Derived parameters
        self.rho = self.lambda_rate * self.A_ser
        self.rho_set = self.lambda_rate * self.A_set
        self.p_sta = np.exp(-self.lambda_rate * self.t_sta)

    def compute_switch_delay(self):
        """
        Compute additional delay caused by UPF switch control (Eq. 12/21).
        """
        N = self.N
        p_sta = self.p_sta
        rho_set = self.rho_set
        lam = self.lambda_rate
        S_set_sq = self.S_set_sq

        numerator = p_sta * (N * (N - 1) + 2 * N * rho_set +
                             (1 + S_set_sq) * rho_set ** 2)
        denominator = 2 * lam ** 2 * (p_sta * (N + rho_set) + 1 - p_sta)

        if denominator == 0:
            return float('inf')

        return numerator / denominator

    def compute_switch_resource(self):
        """
        Compute resource consumption of UPF switch control (Eq. 13/24).
        """
        lam = self.lambda_rate
        rho = self.rho
        p_sta = self.p_sta
        N = self.N

        E_L = self._compute_avg_cycle_length()

        if E_L == 0 or lam == 0:
            return float('inf')

        inner = (self.c_clo * (1 - p_sta) +
                 N * p_sta * self.c_clo / lam +
                 p_sta * self.A_set * self.c_set)

        C_swi = (1.0 / lam) * (rho * self.c_ser + (1.0 / E_L) * inner)
        return C_swi

    def compute_resource_modified(self):
        """
        Modified resource consumption expression (Eq. 24).
        """
        rho = self.rho
        p_sta = self.p_sta
        N = self.N
        rho_set = self.rho_set

        term1 = (1 - rho) * self.c_clo + rho * self.c_ser
        numerator = (1 - rho) * ((1 - p_sta) * (self.c_sta - self.c_clo) +
                                  p_sta * rho_set * (self.c_set - self.c_clo))
        denominator = 1 + p_sta * (N + rho_set - 1)

        if denominator == 0:
            return float('inf')

        return term1 + numerator / denominator

    def _compute_avg_cycle_length(self):
        """
        Compute average queue cycle length E[L] (Eq. 18).
        """
        p_sta = self.p_sta
        N = self.N
        rho_set = self.rho_set
        lam = self.lambda_rate
        rho = self.rho

        if (1 - rho) == 0 or lam == 0:
            return float('inf')

        return (1 - p_sta + p_sta * (N + rho_set)) / (lam * (1 - rho))

    def compute_full_sojourn_delay(self):
        """
        Compute the full average sojourn time (Eq. 22).
        """
        rho = self.rho
        N = self.N
        rho_set = self.rho_set
        lam = self.lambda_rate
        p_sta = self.p_sta
        S_ser_sq = self.S_ser_sq
        S_set_sq = self.S_set_sq

        if (1 - rho) == 0:
            return float('inf')

        term1 = (2 * self.A_ser * (1 - rho) +
                 (1 + S_ser_sq) * rho * self.A_ser) / (2 * (1 - rho))

        numerator = (N * (N - 1) + 2 * N * rho_set +
                     rho_set ** 2 * (1 + S_set_sq))
        denominator = 2 * lam * (N + rho_set + 1.0 / p_sta - 1) if p_sta > 0 else float('inf')

        if denominator == 0:
            return float('inf')

        term2 = numerator / denominator

        return term1 + term2

    def update_state(self, num_arrivals, dt):
        """Update the UPF switch state based on arrivals and elapsed time."""
        if self.state == UPFSwitchState.SERVICE:
            if num_arrivals == 0:
                self.state = UPFSwitchState.STANDBY
                self.standby_timer = 0.0

        elif self.state == UPFSwitchState.STANDBY:
            if num_arrivals > 0:
                self.state = UPFSwitchState.SERVICE
                self.queued_requests = 0
            else:
                self.standby_timer += dt
                if self.standby_timer >= self.t_sta:
                    self.state = UPFSwitchState.CLOSE
                    self.queued_requests = 0

        elif self.state == UPFSwitchState.CLOSE:
            self.queued_requests += num_arrivals
            if self.queued_requests >= self.N:
                self.state = UPFSwitchState.SETUP
                self.setup_timer = 0.0

        elif self.state == UPFSwitchState.SETUP:
            self.queued_requests += num_arrivals
            self.setup_timer += dt
            if self.setup_timer >= self.A_set:
                self.state = UPFSwitchState.SERVICE
                self.queued_requests = 0

        return self.state

    def is_active(self):
        """Check if UPF is in an active state (service or standby)."""
        return self.state in [UPFSwitchState.SERVICE, UPFSwitchState.STANDBY]

    def get_switch_state_value(self):
        """Get normalized switch state for RL state vector."""
        state_map = {
            UPFSwitchState.SERVICE: 1.0,
            UPFSwitchState.STANDBY: 0.75,
            UPFSwitchState.SETUP: 0.25,
            UPFSwitchState.CLOSE: 0.0,
        }
        return state_map.get(self.state, 0.0)

    def get_additional_delay(self):
        """Get additional delay if UPF is in close or setup stage."""
        if self.state == UPFSwitchState.CLOSE:
            return self.A_set + (self.N - self.queued_requests) / max(self.lambda_rate, 1e-6)
        elif self.state == UPFSwitchState.SETUP:
            remaining = max(0, self.A_set - self.setup_timer)
            return remaining
        return 0.0

    def get_resource_consumption(self):
        """Get current resource consumption based on state."""
        state_resources = {
            UPFSwitchState.SERVICE: self.c_ser,
            UPFSwitchState.STANDBY: self.c_sta,
            UPFSwitchState.SETUP: self.c_set,
            UPFSwitchState.CLOSE: self.c_clo,
        }
        return state_resources.get(self.state, 0)

    def reset(self):
        """Reset the controller to initial state."""
        self.state = UPFSwitchState.SERVICE
        self.queued_requests = 0
        self.standby_timer = 0.0
        self.setup_timer = 0.0


def dynamic_parameter_adjustment(num_upfs, iterations=100):
    """
    Algorithm 1: Dynamic Adjustment of Satellite UPF Switch Control Parameters.

    Per the paper, for each UPF u we randomly initialize (t_sta_u, N), and then
    iteratively perturb them; if either the delay D_swi_u (Eq. 12) OR the
    resource consumption C_swi_u (Eq. 13) improves, we keep the new params
    (provided the other metric does not regress beyond a small tolerance,
    enforcing the delay-resource trade-off).

    Returns: dict mapping upf_id -> (optimal_t_sta, optimal_N)
    """
    optimal_params = {}

    for u in range(num_upfs):
        controller = UPFSwitchController(u)

        # Random init (line 2 of Algorithm 1)
        best_t_sta = float(np.random.uniform(10e-3, 100e-3))
        best_N = int(np.random.randint(1, 9))

        controller.t_sta = best_t_sta
        controller.N = best_N
        controller.p_sta = np.exp(-controller.lambda_rate * controller.t_sta)
        controller.rho_set = controller.lambda_rate * controller.A_set
        best_delay = controller.compute_switch_delay()
        best_resource = controller.compute_switch_resource()

        for _ in range(iterations):
            new_t_sta = float(np.random.uniform(10e-3, 100e-3))
            new_N = int(np.random.randint(1, 9))

            controller.t_sta = new_t_sta
            controller.N = new_N
            controller.p_sta = np.exp(-controller.lambda_rate * new_t_sta)
            controller.rho_set = controller.lambda_rate * controller.A_set

            new_delay = controller.compute_switch_delay()
            new_resource = controller.compute_switch_resource()

            # Algorithm 1 line 13: accept if delay OR resource improves.
            improves = (new_delay < best_delay) or (new_resource < best_resource)
            tolerable = (new_delay <= best_delay * 1.1 and
                         new_resource <= best_resource * 1.1)
            if improves and tolerable:
                best_t_sta = new_t_sta
                best_N = new_N
                best_delay = new_delay
                best_resource = new_resource

        optimal_params[u] = (best_t_sta, best_N)

    return optimal_params


class UPFSwitchManager:
    """Manages switch controllers for all UPFs in the network."""

    def __init__(self, num_upfs, optimize=True):
        self.num_upfs = num_upfs
        self.controllers = {}

        if optimize:
            optimal_params = dynamic_parameter_adjustment(num_upfs)
            for u in range(num_upfs):
                t_sta, N = optimal_params[u]
                self.controllers[u] = UPFSwitchController(u, N=N, t_sta=t_sta)
        else:
            for u in range(num_upfs):
                self.controllers[u] = UPFSwitchController(u)

    def get_controller(self, upf_id):
        return self.controllers[upf_id]

    def get_all_switch_states(self):
        return {u: ctrl.get_switch_state_value()
                for u, ctrl in self.controllers.items()}

    def update_state_matrix(self, state_matrix):
        for upf_id, ctrl in self.controllers.items():
            if upf_id < len(state_matrix):
                state_matrix[upf_id, 3] = ctrl.get_switch_state_value()
        return state_matrix

    def reset_all(self):
        for ctrl in self.controllers.values():
            ctrl.reset()

    def get_total_switch_delay(self):
        return sum(ctrl.compute_switch_delay() for ctrl in self.controllers.values())

    def get_total_switch_resource(self):
        return sum(ctrl.compute_switch_resource() for ctrl in self.controllers.values())
