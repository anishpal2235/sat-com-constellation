"""
Main Entry Point for Satellite UPF Service Optimization.
Replicates all experiments from the paper:
  1. Train the proposed algorithm (Algorithm 2)
  2. Evaluate against baselines (Greedy, Random, DRL-OR, DDRL-VNE)
  3. Compare with/without switch control
  4. Test under different network topologies
  5. Analyze switch parameter impact (Figs 9, 10)
  6. Generate all plots (Figs 5-10)
"""

import numpy as np
import time
import os
import json

from config import Config
from src.rl.environment import SatelliteUPFEnvironment
from src.rl.algorithm import SatelliteUPFOptimizer, run_baseline_evaluation
from src.baselines.greedy import GreedyAlgorithm
from src.baselines.random_algo import RandomAlgorithm
from src.baselines.drl_or import DRLORAlgorithm
from src.baselines.ddrl_vne import DDRLVNEAlgorithm
from src.utils.evaluation import (
    plot_training_loss,
    plot_performance_comparison,
    plot_switch_control_comparison,
    plot_topology_comparison,
    plot_switch_parameter_impact,
    plot_delay_resource_tradeoff,
    print_comparison_table,
)


def run_experiment_1_training(output_dir='results'):
    """
    Experiment 1: Train the proposed policy network-based algorithm.
    Generates Fig. 5 (training loss curve).
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Training the Proposed Algorithm")
    print("=" * 60)

    env = SatelliteUPFEnvironment(seed=Config.SEED) #Simulation is deterministic doesn't matter if we use CPU or GPU results don't change
    optimizer = SatelliteUPFOptimizer(env=env, seed=Config.SEED)

    start_time = time.time()
    optimizer.train(num_epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE)
    train_time = time.time() - start_time

    print(f"\nTraining completed in {train_time:.1f} seconds")

    plot_training_loss(optimizer.loss_history, output_dir)

    return optimizer


def run_experiment_2_baseline_comparison(optimizer, output_dir='results'):
    """
    Experiment 2: Compare proposed algorithm with baselines.
    Generates Fig. 6 (performance comparison).
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Baseline Comparison")
    print("=" * 60)

    all_results = {}

    # Test proposed algorithm
    print("\n--- Proposed Algorithm ---")
    proposed_results = optimizer.test(verbose=True)
    all_results['Proposed'] = proposed_results

    # Greedy
    print("\n--- Greedy Algorithm ---")
    env_greedy = SatelliteUPFEnvironment(seed=Config.SEED)
    greedy = GreedyAlgorithm(env_greedy.network)
    greedy_results = run_baseline_evaluation(env_greedy, greedy)
    all_results['Greedy'] = greedy_results

    # Random
    print("\n--- Random Algorithm ---")
    env_random = SatelliteUPFEnvironment(seed=Config.SEED)
    random_algo = RandomAlgorithm(env_random.network)
    random_results = run_baseline_evaluation(env_random, random_algo)
    all_results['Random'] = random_results

    # DRL-OR
    print("\n--- DRL-OR Algorithm ---")
    env_drl = SatelliteUPFEnvironment(seed=Config.SEED)
    drl_or = DRLORAlgorithm(env_drl.network)
    print("  Training DRL-OR agents...")
    drl_or.train_agents(env_drl.train_requests, env_drl.network, episodes=20)
    drl_or_results = run_baseline_evaluation(env_drl, drl_or)
    all_results['DRL-OR'] = drl_or_results

    # DDRL-VNE (RLQVNE in paper figures)
    print("\n--- DDRL-VNE (RLQVNE) Algorithm ---")
    env_vne = SatelliteUPFEnvironment(seed=Config.SEED)
    ddrl_vne = DDRLVNEAlgorithm(env_vne.network)
    print("  Training DDRL-VNE model...")
    ddrl_vne.train(env_vne.train_requests, env_vne.network, episodes=20)
    vne_results = run_baseline_evaluation(env_vne, ddrl_vne)
    all_results['RLQVNE'] = vne_results

    plot_performance_comparison(all_results, output_dir)
    print_comparison_table(all_results)

    return all_results


def run_experiment_3_switch_control(optimizer, output_dir='results'):
    """
    Experiment 3: Compare with/without switch control.
    Generates Fig. 7.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Switch Control Impact")
    print("=" * 60)

    # With switch control (proposed)
    print("\n--- With Switch Control ---")
    results_with = optimizer.test(verbose=True)

    # Without switch control
    print("\n--- Without Switch Control ---")
    env_no_switch = SatelliteUPFEnvironment(seed=Config.SEED)
    for ctrl in env_no_switch.switch_manager.controllers.values():
        ctrl.c_clo = ctrl.c_ser
        ctrl.c_sta = ctrl.c_ser
        ctrl.t_sta = 1e6
        ctrl.N = 1
        ctrl.p_sta = 0.0

    optimizer_no_switch = SatelliteUPFOptimizer(env=env_no_switch, seed=Config.SEED)
    optimizer_no_switch.train(num_epochs=50, batch_size=Config.BATCH_SIZE, verbose=False)
    results_no_switch = optimizer_no_switch.test(verbose=True)

    plot_switch_control_comparison(results_with, results_no_switch, output_dir)

    print(f"\nResource with switch: {results_with['total_resource']:.2f}")
    print(f"Resource without switch: {results_no_switch['total_resource']:.2f}")
    print(f"Delay with switch: {results_with['total_delay']:.2f} ms")
    print(f"Delay without switch: {results_no_switch['total_delay']:.2f} ms")

    return results_with, results_no_switch


def run_experiment_4_topology(optimizer, output_dir='results'):
    """
    Experiment 4: Performance under different satellite network topologies.
    Generates Fig. 8.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Different Network Topologies")
    print("=" * 60)

    topology_results = {}
    algo_configs = {
        'Proposed': None,
        'Greedy': GreedyAlgorithm,
        'Random': RandomAlgorithm,
        'DRL-OR': DRLORAlgorithm,
        'RLQVNE': DDRLVNEAlgorithm,
    }

    for algo_name in algo_configs:
        resource_per_slot = []
        delay_per_slot = []
        acceptance_per_slot = []

        for slot in range(Config.NUM_TIME_SLOTS):
            env = SatelliteUPFEnvironment(seed=Config.SEED + slot * 10)
            env.network.update_topology(slot)

            if algo_name == 'Proposed':
                opt = SatelliteUPFOptimizer(env=env, seed=Config.SEED)
                opt.train(num_epochs=30, batch_size=Config.BATCH_SIZE, verbose=False)
                results = opt.test(verbose=False)
            else:
                algo_class = algo_configs[algo_name]
                algo = algo_class(env.network)
                if hasattr(algo, 'train_agents'):
                    algo.train_agents(env.train_requests, env.network, episodes=10)
                elif hasattr(algo, 'train'):
                    algo.train(env.train_requests, env.network, episodes=10)
                results = run_baseline_evaluation(env, algo, verbose=False)

            n = max(results['total_processed'], 1)
            resource_per_slot.append(results['total_resource'] / n)
            delay_per_slot.append(results['total_delay'] / n)
            acceptance_per_slot.append(results['acceptance_rate'])

        topology_results[algo_name] = {
            'resource_per_slot': resource_per_slot,
            'delay_per_slot': delay_per_slot,
            'acceptance_per_slot': acceptance_per_slot,
        }

        print(f"  {algo_name}: Avg delay={np.mean(delay_per_slot):.2f}ms, "
              f"Avg acceptance={np.mean(acceptance_per_slot):.4f}")

    plot_topology_comparison(topology_results, output_dir)

    return topology_results


def run_experiment_5_switch_parameters(output_dir='results'):
    """
    Experiment 5: Impact of switch control parameters.
    Generates Fig. 9 and Fig. 10.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Switch Parameter Analysis")
    print("=" * 60)

    print("Generating Fig. 9: Switch parameter impact...")
    plot_switch_parameter_impact(output_dir)

    print("Generating Fig. 10: Delay-resource trade-off...")
    plot_delay_resource_tradeoff(output_dir)


def save_results(all_results, output_dir='results'):
    """Save all numerical results to a JSON file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    serializable = {}
    for algo_name, results in all_results.items():
        serializable[algo_name] = {
            'total_delay': float(results['total_delay']),
            'total_resource': float(results['total_resource']),
            'acceptance_rate': float(results['acceptance_rate']),
            'accepted': int(results['accepted']),
            'total_processed': int(results['total_processed']),
        }

    filepath = os.path.join(output_dir, 'results_summary.json')
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {filepath}")


def main():
    """Run all experiments end-to-end."""
    print("=" * 60)
    print("SATELLITE UPF SERVICE OPTIMIZATION")
    print("Delay- and Resource-Aware Optimization")
    print("Policy Network-Based Reinforcement Learning")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Satellites: {Config.NUM_SATELLITES}")
    print(f"  Orbital planes: {Config.NUM_ORBIT_PLANES}")
    print(f"  UPFs: {Config.NUM_UPFS}")
    print(f"  User requests: {Config.NUM_USER_REQUESTS}")
    print(f"  Training epochs: {Config.NUM_EPOCHS}")
    print(f"  Batch size: {Config.BATCH_SIZE}")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

    total_start = time.time()

    # Experiment 1: Training
    optimizer = run_experiment_1_training(output_dir)

    # Experiment 2: Baseline comparison (Fig 6)
    all_results = run_experiment_2_baseline_comparison(optimizer, output_dir)

    # Experiment 3: Switch control comparison (Fig 7)
    run_experiment_3_switch_control(optimizer, output_dir)

    # Experiment 4: Different topologies (Fig 8)
    run_experiment_4_topology(optimizer, output_dir)

    # Experiment 5: Switch parameter analysis (Figs 9, 10)
    run_experiment_5_switch_parameters(output_dir)

    # Save results
    save_results(all_results, output_dir)

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"ALL EXPERIMENTS COMPLETED in {total_time:.1f} seconds")
    print(f"Results saved to: {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
