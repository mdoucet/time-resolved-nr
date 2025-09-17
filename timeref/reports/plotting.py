"""
Plotting functions for time-resolved neutron reflectometry data and models.

This module provides plotting utilities for visualizing SLD environment states,
training results, and model comparisons.
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from ..rl_model import SLDEnv


def plot_sld_env_state(
    env: SLDEnv,
    scale: float = 1,
    newfig: bool = True,
    errors: bool = False,
    label: str = None,
):
    """
    Plot the current state of an SLD environment.

    Args:
        env: SLDEnv instance
        scale: Scaling factor for the reflectivity data
        newfig: Whether to create a new figure
        errors: Whether to plot error bars if available
        label: Custom label for the plot

    Returns:
        None
    """
    if newfig:
        plt.figure(dpi=100)
    plt.plot(env.q, env.refl * scale, color="gray")

    # Check if time_stamp is valid for the data
    if env.time_stamp >= len(env.data):
        # If time_stamp is out of range, use the last available data point
        time_idx = len(env.data) - 1
    else:
        time_idx = env.time_stamp

    # Check if data has error information (3rd column)
    time_data = env.data[time_idx]
    has_errors = len(time_data) >= 3 and time_data[2] is not None

    if has_errors:
        idx = time_data[1] > time_data[2]
    else:
        idx = slice(None)  # Use all data points

    if label is not None:
        _label = label
    else:
        _label = str(time_idx) + " s"

    if errors and has_errors:
        plt.errorbar(
            time_data[0][idx],
            time_data[1][idx] * scale,
            yerr=time_data[2][idx] * scale,
            label=_label,
            linestyle="",
            marker=".",
        )
    else:
        plt.plot(
            time_data[0][idx],
            time_data[1][idx] * scale,
            label=_label,
        )

    plt.gca().legend(frameon=False)
    plt.xlabel("q [$1/\\AA$]")
    plt.ylabel("R(q)")
    plt.xscale("log")
    plt.yscale("log")


def plot_training_results(
    env: SLDEnv,
    results: dict,
    output_path: Path,
):
    """
    Generate comprehensive training results visualization.

    Args:
        env: SLDEnv instance
        results: Dictionary containing training results
        output_path: Path to save the plot

    Returns:
        Path to saved plot file
    """
    plt.figure(figsize=(15, 10))

    # Plot 1: Final state
    plt.subplot(2, 2, 1)
    plot_sld_env_state(env, errors=True, label="Final state", newfig=False)
    plt.title("Final Model State")

    # Plot 2: Reward evolution
    episode_rewards = results["episode_rewards"]
    episode_actions = results["episode_actions"]
    time_points = results["time_points"]
    plt.subplot(2, 2, 2)
    if episode_rewards:
        plt.plot(range(len(episode_rewards)), episode_rewards, "b-", linewidth=2)
        plt.xlabel("Time Step")
        plt.ylabel("Reward")
        plt.title("Reward Evolution")
        plt.grid(True)

    # Plot 3: Action evolution
    trainable_params = env.unwrapped.par_labels
    plt.subplot(2, 2, 3)
    if episode_actions:
        episode_actions_array = np.array(episode_actions)
        for i in range(min(episode_actions_array.shape[1], len(trainable_params))):
            plt.plot(
                range(len(episode_actions)),
                episode_actions_array[:, i],
                label=f"Action {i + 1}: {trainable_params[i]}",
                linewidth=2,
            )
        plt.xlabel("Episode Step")
        plt.ylabel("Action Value")
        plt.title("Action Evolution")
        plt.legend()
        plt.grid(True)

    # Plot 4: Training summary
    plt.subplot(2, 2, 4)
    if episode_rewards:
        plt.text(0.1, 0.8, f"Final Reward: {episode_rewards[-1]:.2f}", fontsize=12)
        plt.text(0.1, 0.7, f"Episode Length: {len(episode_rewards)}", fontsize=12)
    plt.text(0.1, 0.6, f"Trainable Parameters: {len(trainable_params)}", fontsize=12)
    plt.text(0.1, 0.5, f"Data Points: {len(time_points)}", fontsize=12)
    plt.text(
        0.1, 0.4, f"Direction: {'Reverse' if env.reverse else 'Forward'}", fontsize=12
    )
    plt.axis("off")
    plt.title("Training Summary")

    plt.tight_layout()

    if output_path is not None:
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        plot_path = output_path / "training_results.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        logging.info(f"📈 Evaluation results saved: {plot_path}")
    plt.close()
    return plot_path


def plot_initial_state(env: SLDEnv, output_path: Path, show: bool = False) -> Path:
    """
    Plot and save the initial state of the environment.

    Args:
        env: SLDEnv instance
        output_path: Path to save the plot
        show: Whether to display the plot

    Returns:
        Path to saved plot file
    """
    plot_sld_env_state(env, errors=True, label="Initial state")
    plt.title(f"Initial State (Time point {env.time_stamp})")

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_path = output_path / "initial_state.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    logging.info(f"📈 Initial state plot saved: {plot_path}")
    if show:
        plt.show()
    return plot_path


def plot_parameter_evolution(
    env,
    results,
    initial_parameters=None,
    final_parameters=None,
    output_path=None,
    figsize=(6, 8),
    show=False,
):
    """
    Plot parameter evolution over time from RL model results.

    This creates a publication-quality figure showing how parameters evolve over time,
    based on the episode actions from a trained RL model.

    Args:
        env: SLDEnv instance used for parameter information
        results: Dictionary from run_model() containing episode data
        initial_parameters: Optional array of initial parameter values
        final_parameters: Optional array of final parameter values
        output_path: Optional path to save the plot
        figsize: Figure size tuple
        show: Whether to display the plot

    Returns:
        Path to saved plot file if output_path provided, else None
    """
    # Extract data from results
    episode_actions = np.array(results["episode_actions"])
    actions_uncertainties = np.array(results.get("actions_uncertainties", np.zeros_like(episode_actions)))
    time_points = np.array(results["time_points"])
    parameter_labels = results["parameter_labels"]

    # Calculate parameter statistics if not provided
    initial_parameters = env.parameters if not env.reverse else env.end_parameters
    final_parameters = env.end_parameters if not env.reverse else env.parameters
    
    # Transpose for easier indexing (n_params x n_times)
    #deltas = env.high_array - env.low_array
    #values = env.low_array + (1 + episode_actions[:, :len(env.parameters)]) * deltas / 2.0
    print("plotting", episode_actions.shape)
    parameters = env.convert_action_to_parameters(episode_actions).T
    errs = env.convert_action_uncertainties_to_parameters(actions_uncertainties).T

    # Use actual parameter labels if available
    axes_labels = [f"{label}" for label in parameter_labels]

    fig, axs = plt.subplots(
        len(parameters), 1, dpi=100, figsize=figsize, sharex=False
    )
    plt.subplots_adjust(left=0.15, right=0.95, top=0.98, bottom=0.1)

    # Calculate time range for initial/final markers
    t_delay = (time_points[-1] - time_points[0]) * 0.1  # 10% of total time range
    t_initial = time_points[0] - t_delay
    t_final = time_points[-1] + t_delay

    for i in range(len(parameters)):
        if len(parameters) > 1:
            plt.subplot(len(parameters), 1, i + 1)

        # Plot RL results
        plt.errorbar(
            time_points,
            parameters[i],
            yerr=errs[i],
            label="RL",
            marker=".",
            markersize=6,
            linestyle="-",
            linewidth=1.5,
        )

        # Plot initial and final parameter values as stars
        plt.plot(
            [t_initial, t_final],
            [initial_parameters[i], final_parameters[i]],
            linestyle="",
            marker="*",
            markersize=10,
            color="red",
            label="Initial/Final" if i == 0 else "",
        )

        # Set y-label
        if i < len(axes_labels):
            plt.ylabel(axes_labels[i])
        else:
            plt.ylabel(f"Parameter {i + 1}")

        plt.grid(True, alpha=0.3)

    plt.xlabel("Time Steps")

    if output_path:
        plot_path = Path(output_path) / "parameter_evolution.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        logging.info(f"📈 Parameter evolution plot saved: {plot_path}")
        if not show:
            plt.close()
        return plot_path
    elif show:
        plt.show()

    return None


def plot_reflectivity_evolution(
    env,
    model,
    results,
    output_path=None,
    figsize=(6, 15),
    show=False,
):
    """
    Plot reflectivity curves at different time points showing evolution.

    Args:
        env: SLDEnv instance
        model: Trained RL model
        results: Dictionary from run_model() containing episode data
        output_path: Optional path to save the plot
        figsize: Figure size tuple
        show: Whether to display the plot

    Returns:
        Path to saved plot file if output_path provided, else None
    """
    # Extract data from results
    episode_actions = results["episode_actions"]
    time_points = results["time_points"]
    n_times = len(episode_actions)

    # Reset environment
    obs, info = env.reset()

    fig, ax = plt.subplots(dpi=120, figsize=figsize)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.98, bottom=0.05)

    # Plot reflectivity at each time point with scaling
    for i in range(1, n_times, 1):
        # Use the actions from the results instead of predicting again
        action = episode_actions[i]
        obs, reward, terminated, truncated, info = env.step(action)

        # Plot with exponential scaling for visibility
        scale = 10.0**i
        plot_sld_env_state(
            env, 
            scale=scale, 
            newfig=False, 
            errors=True, 
            #label=f"{time_points[i]:.1f}s"
            label=f"Time step {i}"
        )

    # Reverse legend order to match time progression
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(
        handles[::-1], labels[::-1], frameon=False, prop={"size": 9}, loc="upper right"
    )

    if output_path:
        plot_path = Path(output_path) / "reflectivity_evolution.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        logging.info(f"📈 Reflectivity evolution plot saved: {plot_path}")
        if not show:
            plt.close()
        return plot_path
    elif show:
        plt.show()

    return None
