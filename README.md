# Satellite UPF Service Optimization using Deep Reinforcement Learning

Implementation of the paper: **"Delay- and Resource-Aware Satellite UPF Service Optimization"** (IEEE Transactions on Mobile Computing, 2025).

This project simulates an Iridium-like LEO satellite constellation with deployed User Plane Functions (UPFs) and uses a policy-network-based reinforcement learning algorithm to optimize UPF selection for user requests, balancing end-to-end delay, energy consumption, and UPF switch-control overhead.

---

## Table of Contents

1. [Problem Overview](#problem-overview)
2. [System Architecture](#system-architecture)
3. [Repository Structure](#repository-structure)
4. [Detailed Module Descriptions](#detailed-module-descriptions)
5. [Algorithm Description](#algorithm-description)
6. [RL Formulation](#rl-formulation)
7. [Baseline Algorithms](#baseline-algorithms)
8. [Experiments and Figures](#experiments-and-figures)
9. [Configuration Parameters](#configuration-parameters)
10. [Setup and Installation](#setup-and-installation)
11. [Running the Project](#running-the-project)
12. [Running on Google Colab](#running-on-google-colab)
13. [Results](#results)
14. [Notes and Limitations](#notes-and-limitations)
15. [Paper Reference](#paper-reference)

---

## Problem Overview

In a Non-Terrestrial Network (NTN), User Plane Functions (UPFs) are deployed on LEO satellites to process user traffic closer to the edge. As satellites orbit, a user's serving UPF may change, requiring **state migration** (transferring session context) and **traffic re-routing** (redirecting data packets). Poor UPF selection leads to:

- **High delay**: long migration and routing paths across inter-satellite links (ISLs)
- **High energy consumption**: more transmission power over more hops
- **Wasted resources**: unnecessary UPF activations (switch-on/setup costs)

The goal is to select a target UPF for each incoming user request that **jointly minimizes** delay, energy, and switch-control overhead, subject to satellite CPU capacity constraints.

---

## System Architecture

```
User Request ──> Access Satellite ──> [ISL Path] ──> Target UPF Satellite ──> [ISL Path] ──> Internet Satellite
                                            |
                                     Source UPF Satellite
                                     (state migration via ISL)
```

**Key components:**

| Component | Description |
|-----------|-------------|
| **Constellation** | 66 LEO satellites in 6 orbital planes (Iridium-like, +Grid topology) |
| **UPFs** | 36 UPF instances deployed on selected satellites (6 per orbital plane) |
| **ISL Network** | Intra-plane and inter-plane links with random bandwidth (1-10 Gbps) |
| **User Requests** | 6000 total (3000 train / 3000 test), Poisson arrivals (lambda=40/s) |
| **Switch Control** | M/G/1 queue model with N-limited scheme for UPF activation management |

---

## Repository Structure

```
DRL/
|-- config.py                    # Central configuration (Table I parameters)
|-- main.py                      # Entry point: runs all 5 experiments
|-- requirements.txt             # Python dependencies
|-- run_on_colab.ipynb           # Google Colab notebook
|-- README.md                    # This file
|-- docs/
|   `-- Delay-_and_Resource-Aware_Satellite_UPF_Service_Optimization.pdf
|-- results/                     # Generated plots and JSON summary
|   |-- fig5_training_loss.png
|   |-- fig6_performance_comparison.png
|   |-- fig7_switch_control_comparison.png
|   |-- fig8_topology_comparison.png
|   |-- fig9_switch_parameter_impact.png
|   |-- fig10_delay_resource_tradeoff.png
|   `-- results_summary.json
`-- src/
    |-- __init__.py
    |-- network/
    |   |-- __init__.py
    |   `-- satellite_network.py     # Constellation model, ISL topology, routing
    |-- upf/
    |   |-- __init__.py
    |   `-- switch_control.py        # UPF switch states, M/G/1 queue, Algorithm 1
    |-- rl/
    |   |-- __init__.py
    |   |-- environment.py           # RL environment (state, action, reward)
    |   |-- policy_network.py        # Policy network (PyTorch / TF fallback)
    |   `-- algorithm.py             # Algorithm 2: training and testing loops
    |-- baselines/
    |   |-- __init__.py
    |   |-- greedy.py                # Greedy (minimum hops)
    |   |-- random_algo.py           # Random UPF selection
    |   |-- drl_or.py                # DRL-OR (per-node RL agents)
    |   `-- ddrl_vne.py              # DDRL-VNE / RLQVNE (delay-sensitive VNE)
    `-- utils/
        |-- __init__.py
        |-- user_requests.py         # Request generation (Poisson arrivals)
        `-- evaluation.py            # Plotting (Figs 5-10) and result tables
```

---

## Detailed Module Descriptions

### `config.py` -- Configuration

Central configuration class matching **Table I** from the paper. Key parameter groups:

- **Constellation**: 66 satellites, 6 orbital planes, 11 sats/plane, 36 UPFs
- **Satellite resources**: CPU capacity 100-200 TFLOPS per satellite
- **ISL bandwidth**: 1-10 Gbps per link
- **User requests**: 6000 total, Poisson arrival rate lambda=40/s
- **Packet parameters**: state packets (0.05-0.5 Mb, 1-10 TFLOPS), data packets (0.5-5.0 Mb, 1-50 TFLOPS)
- **Switch control**: CPU costs per stage (service=30, standby=20, setup=20, close=5 TFLOPS), N-threshold=3, standby duration=50ms
- **Orbital mechanics**: Iridium altitude 781 km, orbit period ~100 min, satellite speed 7.5 km/s
- **RL hyperparameters**: 100 epochs, batch size 32, learning rate 1e-3, gamma=0.99
- **Reward weights**: eta1=0.4 (delay), eta2=0.3 (energy), eta3=0.3 (switch cost)

Includes helper methods `get_device()` (auto-detect GPU), `get_satellite_cpu()`, and `get_bandwidth()`.

### `src/network/satellite_network.py` -- Satellite Network Model

- **`Satellite`**: Data class storing satellite ID, orbital plane, position index, CPU capacity/availability, and UPF assignment
- **`SatelliteNetwork`**: Builds the full constellation:
  - Creates 66 satellites across 6 orbital planes
  - Deploys 36 UPFs on evenly-spaced satellites (6 per plane)
  - Establishes intra-plane ISLs (adjacent satellites in same orbit) and inter-plane ISLs (satellites at same position in adjacent orbits) -- "+Grid" topology
  - Computes link distances using orbital geometry (intra-plane ~4080 km, inter-plane ~4400 km)
  - Runs Floyd-Warshall for all-pairs shortest distances
  - Provides BFS-based shortest-delay routing with transmission delay calculation
  - Supports topology updates across time slots (simulating orbital motion)
  - Tracks per-satellite resource consumption and release

### `src/upf/switch_control.py` -- UPF Switch Control (Algorithm 1)

Implements the M/G/1 queue-based UPF switch control from Section III of the paper.

- **`UPFSwitchState`**: Enum for four operational states: `SERVICE`, `STANDBY`, `CLOSE`, `SETUP`
- **`UPFSwitchController`**: Per-UPF controller that:
  - Manages state transitions based on arrival patterns and timers
  - Computes additional switch delay (setup time when waking from close/standby)
  - Computes switch resource consumption per state
  - Implements the N-limited scheme: UPF enters standby after N consecutive idle periods, then closes after standby timeout
  - Tracks queue statistics (arrival rate, utilization, idle probability)
- **`UPFSwitchManager`**: Manages all 36 controllers
  - `dynamic_parameter_adjustment()`: searches for optimal N and t_sta via grid search (Algorithm 1)
  - `update_state_matrix()`: injects switch state features into the RL observation

### `src/rl/environment.py` -- RL Environment

Implements the MDP formulation from **Section IV-B**:

- **State** `M(t)`: Matrix of shape `(U, 4)` where each UPF has features:
  - `c_u`: normalized remaining CPU capacity
  - `b_sum_u`: normalized total bandwidth of neighboring ISLs
  - `f_avg_u`: normalized average distance to other UPFs
  - `g_swi_u`: normalized switch state (0=service, 0.33=standby, 0.67=setup, 1=close)

- **Action**: Integer in `[0, U-1]` selecting the target UPF

- **Reward/Cost** (Eq. 27): `R(t) = eta1 * avg_delay_norm + eta2 * avg_energy_norm [+ eta3 * switch_cost_norm if gamma=1]`
  - This is a **cost** (lower is better) used to weight the cross-entropy loss
  - `gamma=1` when a UPF must be newly activated (cold start penalty)

- **Resource management**: Session-based holding with exponential duration (mean 4.5s). Resources are allocated on acceptance and released when the session completes, creating realistic contention.

- **Topology updates**: Network topology changes every `len(requests)/NUM_TIME_SLOTS` steps to simulate orbital motion.

### `src/rl/policy_network.py` -- Policy Network

- **`PolicyNetworkTorch`**: PyTorch implementation
  - Architecture: `Input(U,4) -> FC(64) -> ReLU -> FC(64) -> ReLU -> FC(1) -> Squeeze -> MaskedSoftmax`
  - Each UPF's 4-feature vector is scored independently, then softmax produces a probability distribution over all UPFs
  - Available mask sets infeasible UPF probabilities to zero before normalization
  - `select_action()`: samples from the distribution (training) or takes argmax (testing)
- **`PolicyNetworkSimple`**: TensorFlow/Keras fallback for environments without PyTorch

### `src/rl/algorithm.py` -- Algorithm 2: Training and Testing

- **`SatelliteUPFOptimizer`**: Main algorithm class
  - **Training** (offline phase):
    1. For each epoch, iterate over all training requests
    2. Get state matrix, compute available UPF mask
    3. Sample action from policy network
    4. Execute action in environment, collect reward (cost)
    5. Accumulate batches; update policy every `batch_size` steps
  - **Policy update** (Eq. 29): Cross-entropy loss weighted by reward
    ```
    L = mean( -log(pi(a|s)) * R(t) )
    ```
    with gradient clipping (max_norm=1.0)
  - **Testing** (online phase): Deterministic action selection (argmax) on test requests
- **`run_baseline_evaluation()`**: Runs any baseline algorithm through the same environment for fair comparison

### `src/baselines/` -- Baseline Algorithms

| Algorithm | Class | Strategy | Paper Reference |
|-----------|-------|----------|-----------------|
| **Greedy** | `GreedyAlgorithm` | Select UPF with minimum hop count from source | Basic greedy heuristic |
| **Random** | `RandomAlgorithm` | Uniformly random feasible UPF | Random baseline |
| **DRL-OR** | `DRLORAlgorithm` | Per-node RL agents optimizing routing delay + light load factor (0.1 weight) | Ref [43] |
| **DDRL-VNE / RLQVNE** | `DDRLVNEAlgorithm` | Delay-sensitive VNE selecting geographically closest UPF, with optional learned policy | Ref [45] |

Both DRL-OR and DDRL-VNE have trainable neural network components that are pre-trained before evaluation.

### `src/utils/user_requests.py` -- Request Generation

- **`UserRequest`**: Data class with fields:
  - `access_sat_id`, `internet_sat_id`: entry/exit satellites
  - `source_upf_id`: current serving UPF
  - `v_state`, `v_data`: packet sizes (Mb)
  - `c_state`, `c_data`: CPU requirements (TFLOPS)
  - `arrival_time`: generated via exponential inter-arrival times (Poisson process)
- **`generate_user_requests()`**: Creates `N` requests with random parameters within configured ranges

### `src/utils/evaluation.py` -- Plotting and Evaluation

Generates all paper figures using matplotlib (headless `Agg` backend):

| Function | Output | Paper Figure |
|----------|--------|-------------|
| `plot_training_loss()` | Training loss curve | Fig. 5 |
| `plot_performance_comparison()` | Delay, resource, acceptance bar charts | Fig. 6 |
| `plot_switch_control_comparison()` | With/without switch control | Fig. 7 |
| `plot_topology_comparison()` | Performance across topology snapshots | Fig. 8 |
| `plot_switch_parameter_impact()` | N and t_sta sweep results | Fig. 9 |
| `plot_delay_resource_tradeoff()` | Delay vs resource Pareto front | Fig. 10 |
| `print_comparison_table()` | Console summary table | -- |

---

## Algorithm Description

### Algorithm 1: UPF Switch Control Optimization

The M/G/1 queue model determines when UPFs should transition between states:

```
SERVICE  ──(idle for N periods)──>  STANDBY  ──(timeout t_sta)──>  CLOSE
    ^                                   |                             |
    |                                   |                             |
    <───────(new arrival)───────────────<─────(new arrival + setup)───<
```

- **Service**: UPF is actively processing (CPU cost = 30 TFLOPS)
- **Standby**: UPF is idle but warm, ready for quick reactivation (CPU cost = 20 TFLOPS)
- **Close**: UPF is powered down (CPU cost = 5 TFLOPS)
- **Setup**: UPF is being reactivated from close state (CPU cost = 20 TFLOPS, adds setup delay)

Parameters N (idle threshold) and t_sta (standby duration) are optimized via Algorithm 1 to balance resource savings against reactivation delay.

### Algorithm 2: State-Aware UPF Service Optimization

Two-phase approach:

1. **Offline Training**: Policy network learns to map UPF state matrices to optimal UPF selection probabilities using reward-weighted cross-entropy loss
2. **Online Inference**: Trained policy selects target UPFs deterministically; BFS shortest-delay paths handle state migration and traffic routing

---

## RL Formulation

| Component | Definition |
|-----------|-----------|
| **State** | `M(t) in R^{U x 4}`: per-UPF features [CPU, bandwidth, distance, switch state] |
| **Action** | `a(t) in {0, 1, ..., U-1}`: target UPF index |
| **Cost/Reward** | `R(t) = eta1 * D_norm + eta2 * W_norm + eta3 * C_swi_norm` (Eq. 27) |
| **Loss** | `L(h,o) = -sum( h_u * log(o_u) )` weighted by R(t) (Eq. 29) |
| **Policy** | `pi(a|M(t))`: softmax over masked UPF scores from neural network |

The reward is a **cost** (lower = better). The policy gradient pushes probability mass away from high-cost actions.

---

## Baseline Algorithms

1. **Greedy**: Selects the UPF reachable in the fewest hops from the source UPF. Simple but ignores load and energy.
2. **Random**: Uniformly random selection among feasible UPFs. Provides a lower-bound reference.
3. **DRL-OR** (Ref [43]): Per-node reinforcement learning agents that optimize routing decisions. Considers delay with a small load-balancing factor.
4. **DDRL-VNE / RLQVNE** (Ref [45]): Delay-sensitive virtual network embedding using deep RL. Prioritizes geographic proximity (shortest distance) to minimize propagation delay.

---

## Experiments and Figures

| Experiment | Description | Output |
|------------|-------------|--------|
| **1. Training** | Train proposed algorithm for 100 epochs on 3000 requests | `fig5_training_loss.png` |
| **2. Baseline Comparison** | Compare Proposed vs Greedy vs Random vs DRL-OR vs RLQVNE on test set | `fig6_performance_comparison.png` |
| **3. Switch Control** | Compare performance with and without UPF switch control | `fig7_switch_control_comparison.png` |
| **4. Topology Variation** | Evaluate all algorithms under 4 different topology snapshots | `fig8_topology_comparison.png` |
| **5a. Parameter Sweep** | Vary N (threshold) and t_sta (standby time) independently | `fig9_switch_parameter_impact.png` |
| **5b. Trade-off Analysis** | Map delay vs resource consumption across parameter space | `fig10_delay_resource_tradeoff.png` |

---

## Configuration Parameters

Key parameters from `config.py` (matching Table I of the paper):

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| NUM_SATELLITES | S | 66 | Total LEO satellites |
| NUM_ORBIT_PLANES | -- | 6 | Orbital planes |
| NUM_UPFS | U | 36 | Deployed UPF instances |
| NUM_USER_REQUESTS | R | 6000 | Total user requests (3000 train + 3000 test) |
| AVG_ARRIVAL_RATE | lambda | 40/s | Average Poisson arrival rate |
| CPU_SATELLITES_MIN/MAX | -- | 100-200 | Satellite CPU capacity (TFLOPS) |
| BANDWIDTH_MIN/MAX | -- | 1-10 | ISL bandwidth (Gbps) |
| CPU_STATE_MIN/MAX | -- | 1-10 | State packet CPU demand (TFLOPS) |
| CPU_DATA_MIN/MAX | -- | 1-50 | Data packet CPU demand (TFLOPS) |
| STATE_PACKET_SIZE | -- | 0.05-0.5 | State packet size (Mb) |
| DATA_PACKET_SIZE | -- | 0.5-5.0 | Data packet size (Mb) |
| CPU_SERVICE | c_u^ser | 30 | CPU during service (TFLOPS) |
| CPU_STANDBY | c_u^sta | 20 | CPU during standby (TFLOPS) |
| CPU_SETUP | c_u^set | 20 | CPU during setup (TFLOPS) |
| CPU_CLOSE | c_u^clo | 5 | CPU during close (TFLOPS) |
| N_THRESHOLD | N | 3 | N-limited scheme threshold |
| STANDBY_DURATION | t_sta | 50 ms | Standby timeout |
| SESSION_DURATION_MEAN | -- | 4.5 s | Mean resource holding time |
| SERVICE_DURATION_MEAN | A(t_u^ser) | 5 ms | Mean service time (M/G/1) |
| ORBIT_ALTITUDE | -- | 781 km | Iridium constellation altitude |
| NUM_EPOCHS | -- | 100 | Training epochs |
| BATCH_SIZE | -- | 32 | Mini-batch size |
| LEARNING_RATE | -- | 1e-3 | Adam optimizer learning rate |
| GAMMA | -- | 0.99 | Discount factor |
| ETA1/ETA2/ETA3 | eta | 0.4/0.3/0.3 | Reward weights (delay/energy/switch) |

---

## Setup and Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+ (GPU optional but recommended)
- NumPy 1.21+
- Matplotlib 3.5+

### Install

```bash
# Clone or download the repository
cd DRL

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Project

### Full experiment suite

```bash
python main.py
```

This runs all 5 experiments sequentially and produces:
- 6 PNG plots in `results/`
- Numerical summary in `results/results_summary.json`
- Console output with progress and comparison tables

Typical runtime: 10-30 minutes depending on hardware (faster with GPU).

### Individual experiments

```python
from config import Config
from src.rl.environment import SatelliteUPFEnvironment
from src.rl.algorithm import SatelliteUPFOptimizer

# Create environment and optimizer
env = SatelliteUPFEnvironment(seed=Config.SEED)
optimizer = SatelliteUPFOptimizer(env=env, seed=Config.SEED)

# Train
optimizer.train(num_epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE)

# Test
results = optimizer.test(verbose=True)
print(f"Delay: {results['total_delay']:.2f} ms")
print(f"Acceptance: {results['acceptance_rate']:.4f}")
```

---

## Running on Google Colab

1. Upload the `DRL/` folder to your Google Drive (e.g., `MyDrive/Sat_Optimisation/DRL/`)
2. Open `run_on_colab.ipynb` in Colab
3. (Optional) Select **Runtime > Change runtime type > T4 GPU**
4. Update `PROJECT_PATH` in the notebook to match your Drive path
5. Run all cells

The notebook handles Drive mounting, dependency installation, GPU detection, and experiment execution via subprocess.

---

## Results

### Sample Output (results_summary.json)

| Algorithm | Total Delay (ms) | Total Resource (TFLOPS) | Acceptance Rate |
|-----------|----------------:|------------------------:|----------------:|
| **Proposed** | 512,367 | 4,953 | 90.5% |
| Greedy | 478,140 | 5,031 | 90.1% |
| Random | 564,727 | 5,114 | 92.6% |
| DRL-OR | 467,889 | 4,796 | 89.7% |
| RLQVNE | 519,549 | 5,077 | 90.5% |

### Generated Figures

- **Fig. 5** (Training Loss): Loss decreases from ~1.04 to ~0.5 over 100 epochs, showing convergence
- **Fig. 6** (Performance Comparison): Bar charts comparing delay, resource consumption, and acceptance rate across all algorithms
- **Fig. 7** (Switch Control): Shows the resource savings from switch control vs the additional delay introduced
- **Fig. 8** (Topology Variation): Performance stability across 4 topology snapshots
- **Fig. 9** (Parameter Impact): Effect of varying N (1-10) and t_sta (10-200ms) on delay and resource consumption
- **Fig. 10** (Trade-off): Delay-resource Pareto frontier showing the achievable trade-off space

---

## Notes and Limitations

- This is a **simulation-based research implementation**, not a production system.
- The policy network uses fully-connected layers (per-UPF scoring) rather than the convolutional architecture mentioned in some paper descriptions.
- The reward function operates as a **weighted cost** (higher = worse), which weights the cross-entropy loss to push the policy toward lower-cost actions.
- Resource contention is modeled via session-based holding (mean 4.5s), which may differ from real-world traffic patterns.
- All algorithms share the same environment seed for fair comparison (except Random, which uses a different seed for diversity).
- No automated test suite is currently included.
- PyTorch is preferred; a simplified TensorFlow fallback exists but does not perform full gradient-based policy updates.

---

## Paper Reference

> **"Delay- and Resource-Aware Satellite UPF Service Optimization"**
> IEEE Transactions on Mobile Computing (TMC), 2025

The full paper is included at `docs/Delay-_and_Resource-Aware_Satellite_UPF_Service_Optimization.pdf`.
