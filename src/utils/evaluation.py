"""
Evaluation and Plotting Module.
Replicates Figures 5-10 from the paper:
  Fig 5: Training loss curve
  Fig 6: Performance comparison (resource, delay, acceptance rate)
  Fig 7: With/without switch control comparison
  Fig 8: Performance under different topologies
  Fig 9: Impact of N and t_sta on delay and resource consumption
  Fig 10: Delay-resource consumption trade-off
"""

import numpy as np
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False

from src.upf.switch_control import UPFSwitchController


def ensure_output_dir(output_dir='results'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def plot_training_loss(loss_history, output_dir='results'):
    """Plot Figure 5: Training results of loss."""
    if not PLT_AVAILABLE:
        print("matplotlib not available, skipping plot")
        return

    ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(8, 6))
    epochs = range(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, 'b-', linewidth=2)
    ax.set_xlabel('epoch', fontsize=14)
    ax.set_ylabel('loss', fontsize=14)
    ax.set_title('Fig. 5: Training Results of Loss', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_training_loss.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/fig5_training_loss.png")


def plot_performance_comparison(all_results, output_dir='results'):
    """Plot Figure 6: Performance comparison with baselines."""
    if not PLT_AVAILABLE:
        print("matplotlib not available, skipping plot")
        return

    ensure_output_dir(output_dir)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    colors = {
        'Proposed': 'blue', 'Greedy': 'orange', 'Random': 'green',
        'DRL-OR': 'red', 'RLQVNE': 'purple'
    }
    markers = {
        'Proposed': 'o', 'Greedy': 's', 'Random': '^',
        'DRL-OR': 'D', 'RLQVNE': 'v'
    }

    # Helper: get x-axis as actual time in units of x10^3 ms
    def get_sampled(results, key, num_points=20):
        history = results[key]
        times = results.get('time_history', [i * 0.5 for i in range(len(history))])
        step = max(1, len(history) // num_points)
        indices = list(range(0, len(history), step))
        x = [times[i] / 1000.0 for i in indices]  # convert ms to x10^3 ms
        y = [history[i] for i in indices]
        return x, y

    # (a) Total Resource Consumption (cumulative)
    ax = axes[0]
    for name, results in all_results.items():
        x, y = get_sampled(results, 'resource_history')
        ax.plot(x, y, color=colors.get(name, 'black'),
                marker=markers.get(name, 'o'), label=name,
                linewidth=2, markersize=6, markevery=1)
    ax.set_xlabel(r'time ($\times 10^3$ ms)', fontsize=12)
    ax.set_ylabel('total resource consumption', fontsize=12)
    ax.set_title('(a) Comparison of total resource consumption', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # (b) Total Service Delay
    ax = axes[1]
    for name, results in all_results.items():
        x, y = get_sampled(results, 'delay_history')
        ax.plot(x, y, color=colors.get(name, 'black'),
                marker=markers.get(name, 'o'), label=name,
                linewidth=2, markersize=6, markevery=1)
    ax.set_xlabel(r'time ($\times 10^3$ ms)', fontsize=12)
    ax.set_ylabel('total delay (ms)', fontsize=12)
    ax.set_title('(b) Comparison of total service delay', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # (c) User Request Acceptance Rate
    ax = axes[2]
    for name, results in all_results.items():
        x, y = get_sampled(results, 'acceptance_history')
        ax.plot(x, y, color=colors.get(name, 'black'),
                marker=markers.get(name, 'o'), label=name,
                linewidth=2, markersize=6, markevery=1)
    ax.set_xlabel(r'time ($\times 10^3$ ms)', fontsize=12)
    ax.set_ylabel('user request acceptance rate', fontsize=12)
    ax.set_title('(c) Comparison of user request acceptance rate', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Fig. 6: Performance Comparison with Baselines', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig6_performance_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/fig6_performance_comparison.png")


def plot_switch_control_comparison(results_with_switch, results_no_switch,
                                   output_dir='results'):
    """Plot Figure 7: With/without satellite UPF switch control."""
    if not PLT_AVAILABLE:
        return

    ensure_output_dir(output_dir)
    fig, axes = plt.subplots(2, 1, figsize=(8, 12))

    def get_sampled_fig7(results, key, num_points=20):
        history = results[key]
        times = results.get('time_history', [i * 0.5 for i in range(len(history))])
        step = max(1, len(history) // num_points)
        indices = list(range(0, len(history), step))
        x = [times[i] / 1000.0 for i in indices]  # x10^3 ms
        y = [history[i] for i in indices]
        return x, y

    # (a) Resource consumption (cumulative)
    ax = axes[0]
    x_w, y_w = get_sampled_fig7(results_with_switch, 'resource_history')
    x_no, y_no = get_sampled_fig7(results_no_switch, 'resource_history')
    ax.plot(x_w, y_w, 'bo-', label='Proposed', linewidth=2, markersize=6)
    ax.plot(x_no, y_no, 'rs-', label='No switch', linewidth=2, markersize=6)
    ax.set_xlabel(r'time ($\times 10^3$ ms)', fontsize=12)
    ax.set_ylabel('total resource consumption', fontsize=12)
    ax.set_title('(a) Comparison of total resource consumption\nwith/without satellite UPF switch control', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # (b) Service delay (cumulative)
    ax = axes[1]
    x_w, y_w = get_sampled_fig7(results_with_switch, 'delay_history')
    x_no, y_no = get_sampled_fig7(results_no_switch, 'delay_history')
    ax.plot(x_w, y_w, 'bo-', label='Proposed', linewidth=2, markersize=6)
    ax.plot(x_no, y_no, 'rs-', label='No Switch', linewidth=2, markersize=6)
    ax.set_xlabel(r'time ($\times 10^3$ ms)', fontsize=12)
    ax.set_ylabel('total delay (ms)', fontsize=12)
    ax.set_title('(b) Comparison of service delay\nwith/without satellite UPF switch control', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Fig. 7: With/Without Satellite UPF Switch Control', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig7_switch_control_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/fig7_switch_control_comparison.png")


def plot_topology_comparison(topology_results, output_dir='results'):
    """Plot Figure 8: Performance under different satellite network topologies."""
    if not PLT_AVAILABLE:
        return

    ensure_output_dir(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    time_slots = ['T1', 'T2', 'T3', 'T4']
    algo_names = list(topology_results.keys())
    num_algos = len(algo_names)
    x = np.arange(len(time_slots))
    width = 0.15

    colors = {
        'Proposed': 'blue', 'Greedy': 'orange', 'Random': 'green',
        'DRL-OR': 'red', 'RLQVNE': 'purple'
    }
    hatches = {
        'Proposed': '///', 'Greedy': '\\\\\\', 'Random': 'xxx',
        'DRL-OR': '...', 'RLQVNE': '+++'
    }

    # (a) Average resource consumption
    ax = axes[0]
    for i, name in enumerate(algo_names):
        values = topology_results[name]['resource_per_slot']
        ax.bar(x + i * width, values, width, label=name,
               color=colors.get(name, 'gray'), hatch=hatches.get(name, ''), alpha=0.8)
    ax.set_xlabel('satellite topology under T', fontsize=12)
    ax.set_ylabel('average resource consumption', fontsize=12)
    ax.set_title('(a) Average resource consumption', fontsize=11)
    ax.set_xticks(x + width * (num_algos - 1) / 2)
    ax.set_xticklabels(time_slots)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # (b) Average delay
    ax = axes[1]
    for i, name in enumerate(algo_names):
        values = topology_results[name]['delay_per_slot']
        ax.bar(x + i * width, values, width, label=name,
               color=colors.get(name, 'gray'), hatch=hatches.get(name, ''), alpha=0.8)
    ax.set_xlabel('satellite topology under T', fontsize=12)
    ax.set_ylabel('average delay (ms)', fontsize=12)
    ax.set_title('(b) Average delay', fontsize=11)
    ax.set_xticks(x + width * (num_algos - 1) / 2)
    ax.set_xticklabels(time_slots)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # (c) Average acceptance rate
    ax = axes[2]
    for i, name in enumerate(algo_names):
        values = topology_results[name]['acceptance_per_slot']
        ax.bar(x + i * width, values, width, label=name,
               color=colors.get(name, 'gray'), hatch=hatches.get(name, ''), alpha=0.8)
    ax.set_xlabel('satellite topology under T', fontsize=12)
    ax.set_ylabel('average acceptance rate', fontsize=12)
    ax.set_title('(c) Average acceptance rate', fontsize=11)
    ax.set_xticks(x + width * (num_algos - 1) / 2)
    ax.set_xticklabels(time_slots)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Fig. 8: Performance Under Different Topologies', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig8_topology_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/fig8_topology_comparison.png")


def plot_switch_parameter_impact(output_dir='results'):
    """Plot Figure 9: Impact of N and t_sta on delay and resource consumption."""
    if not PLT_AVAILABLE:
        return

    ensure_output_dir(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    rho_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    N_values = list(range(1, 9))
    t_sta_values = np.linspace(10, 80, 8)

    colors_rho = ['blue', 'orange', 'green', 'red', 'purple']
    markers_rho = ['o', 's', '^', 'D', 'v']

    # (a) Impact of N on average delay
    ax = axes[0, 0]
    for idx, rho in enumerate(rho_values):
        delays = []
        for N in N_values:
            ctrl = UPFSwitchController(0, N=N)
            ctrl.rho = rho
            ctrl.lambda_rate = rho / ctrl.A_ser
            ctrl.p_sta = np.exp(-ctrl.lambda_rate * ctrl.t_sta)
            ctrl.rho_set = ctrl.lambda_rate * ctrl.A_set
            delay = ctrl.compute_switch_delay() * 1000
            delays.append(delay)
        ax.plot(N_values, delays, color=colors_rho[idx],
                marker=markers_rho[idx], label=f'rho={rho}', linewidth=2, markersize=6)
    ax.set_xlabel('N', fontsize=12)
    ax.set_ylabel('delay (ms)', fontsize=12)
    ax.set_title('(a) The impact of N on average delay', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Impact of N on resource consumption
    ax = axes[0, 1]
    for idx, rho in enumerate(rho_values):
        resources = []
        for N in N_values:
            ctrl = UPFSwitchController(0, N=N)
            ctrl.rho = rho
            ctrl.lambda_rate = rho / ctrl.A_ser
            ctrl.p_sta = np.exp(-ctrl.lambda_rate * ctrl.t_sta)
            ctrl.rho_set = ctrl.lambda_rate * ctrl.A_set
            resource = ctrl.compute_resource_modified()
            resources.append(resource)
        ax.plot(N_values, resources, color=colors_rho[idx],
                marker=markers_rho[idx], label=f'rho={rho}', linewidth=2, markersize=6)
    ax.set_xlabel('N', fontsize=12)
    ax.set_ylabel('resource consumption', fontsize=12)
    ax.set_title('(b) The impact of N on resource consumption', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (c) Impact of standby time on average delay
    ax = axes[1, 0]
    for idx, rho in enumerate(rho_values):
        delays = []
        for t_sta in t_sta_values:
            ctrl = UPFSwitchController(0, t_sta=t_sta / 1000.0)
            ctrl.rho = rho
            ctrl.lambda_rate = rho / ctrl.A_ser
            ctrl.p_sta = np.exp(-ctrl.lambda_rate * ctrl.t_sta)
            ctrl.rho_set = ctrl.lambda_rate * ctrl.A_set
            delay = ctrl.compute_full_sojourn_delay() * 1000
            delays.append(delay)
        ax.plot(t_sta_values, delays, color=colors_rho[idx],
                marker=markers_rho[idx], label=f'rho={rho}', linewidth=2, markersize=6)
    ax.set_xlabel('standby time (ms)', fontsize=12)
    ax.set_ylabel('delay (ms)', fontsize=12)
    ax.set_title('(c) The impact of standby time on average delay', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (d) Impact of standby time on resource consumption
    ax = axes[1, 1]
    for idx, rho in enumerate(rho_values):
        resources = []
        for t_sta in t_sta_values:
            ctrl = UPFSwitchController(0, t_sta=t_sta / 1000.0)
            ctrl.rho = rho
            ctrl.lambda_rate = rho / ctrl.A_ser
            ctrl.p_sta = np.exp(-ctrl.lambda_rate * ctrl.t_sta)
            ctrl.rho_set = ctrl.lambda_rate * ctrl.A_set
            resource = ctrl.compute_resource_modified()
            resources.append(resource)
        ax.plot(t_sta_values, resources, color=colors_rho[idx],
                marker=markers_rho[idx], label=f'rho={rho}', linewidth=2, markersize=6)
    ax.set_xlabel('standby time (ms)', fontsize=12)
    ax.set_ylabel('resource consumption', fontsize=12)
    ax.set_title('(d) The impact of standby time on resource consumption', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Fig. 9: Impact of Switch Parameters N and t_sta', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig9_switch_parameter_impact.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/fig9_switch_parameter_impact.png")


def plot_delay_resource_tradeoff(output_dir='results'):
    """Plot Figure 10: Delay-resource consumption trade-off."""
    if not PLT_AVAILABLE:
        return

    ensure_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(8, 6))

    N_values_to_plot = [5, 6]
    colors_N = ['blue', 'orange']
    markers_N = ['o', 's']

    for n_idx, N in enumerate(N_values_to_plot):
        delays = []
        resources = []
        t_sta_sweep = np.linspace(10, 100, 20)

        for t_sta in t_sta_sweep:
            ctrl = UPFSwitchController(0, N=N, t_sta=t_sta / 1000.0)
            ctrl.rho = 0.3
            ctrl.lambda_rate = ctrl.rho / ctrl.A_ser
            ctrl.p_sta = np.exp(-ctrl.lambda_rate * ctrl.t_sta)
            ctrl.rho_set = ctrl.lambda_rate * ctrl.A_set

            delay = ctrl.compute_full_sojourn_delay() * 1000
            resource = ctrl.compute_resource_modified()

            delays.append(delay)
            resources.append(resource)

        ax.plot(resources, delays, color=colors_N[n_idx],
                marker=markers_N[n_idx], label=f'N={N}', linewidth=2, markersize=8)

    ax.set_xlabel('resource consumption', fontsize=14)
    ax.set_ylabel('delay (ms)', fontsize=14)
    ax.set_title('Fig. 10: Delay-Resource Consumption Trade-off', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig10_delay_resource_tradeoff.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/fig10_delay_resource_tradeoff.png")


def print_comparison_table(all_results):
    """Print a summary table comparing all algorithms."""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Algorithm':<15} {'Total Delay(ms)':<18} {'Total Resource':<18} "
          f"{'Acceptance Rate':<18}")
    print("-" * 80)

    proposed_delay = None
    proposed_resource = None
    proposed_accept = None

    for name, results in all_results.items():
        delay = results['total_delay']
        resource = results['total_resource']
        accept = results['acceptance_rate']

        if name == 'Proposed':
            proposed_delay = delay
            proposed_resource = resource
            proposed_accept = accept

        print(f"{name:<15} {delay:<18.2f} {resource:<18.2f} {accept:<18.4f}")

    if proposed_delay is not None:
        print("\n" + "-" * 80)
        print("IMPROVEMENTS OF PROPOSED ALGORITHM:")
        print("-" * 80)

        others_delay = []
        others_resource = []
        others_accept = []

        for name, results in all_results.items():
            if name != 'Proposed':
                delay_imp = (results['total_delay'] - proposed_delay) / results['total_delay'] * 100
                resource_imp = (results['total_resource'] - proposed_resource) / results['total_resource'] * 100
                accept_imp = (proposed_accept - results['acceptance_rate']) / max(results['acceptance_rate'], 1e-6) * 100

                others_delay.append(delay_imp)
                others_resource.append(resource_imp)
                others_accept.append(accept_imp)

                print(f"  vs {name:<12}: "
                      f"Delay {delay_imp:+.1f}%, "
                      f"Resource {resource_imp:+.1f}%, "
                      f"Acceptance {accept_imp:+.1f}%")

        if others_delay:
            print(f"\n  Average improvement: "
                  f"Delay {np.mean(others_delay):+.1f}%, "
                  f"Resource {np.mean(others_resource):+.1f}%, "
                  f"Acceptance {np.mean(others_accept):+.1f}%")

    print("=" * 80)
