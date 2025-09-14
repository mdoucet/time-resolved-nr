"""
Plotting functions for time-resolved neutron reflectometry data and models.

This module provides plotting utilities for visualizing SLD environment states,
training results, and model comparisons.
"""

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
    episode_rewards: list,
    episode_actions: list,
    time_points: list,
    reverse: bool,
    steps: int,
    trainable_params: list,
    output_path: Path,
):
    """
    Generate comprehensive training results visualization.

    Args:
        env: SLDEnv instance
        episode_rewards: List of rewards from training episode
        episode_actions: List of actions from training episode
        time_points: List of time points
        reverse: Whether training was in reverse direction
        steps: Number of training steps
        trainable_params: List of trainable parameter names
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
    plt.subplot(2, 2, 2)
    if episode_rewards:
        plt.plot(range(len(episode_rewards)), episode_rewards, "b-", linewidth=2)
        plt.xlabel("Episode Step")
        plt.ylabel("Reward")
        plt.title("Training Progress")
        plt.grid(True)

    # Plot 3: Action evolution
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
    plt.text(0.1, 0.9, f"Training Steps: {steps}", fontsize=12)
    if episode_rewards:
        plt.text(0.1, 0.8, f"Final Reward: {episode_rewards[-1]:.2f}", fontsize=12)
        plt.text(0.1, 0.7, f"Episode Length: {len(episode_rewards)}", fontsize=12)
    plt.text(0.1, 0.6, f"Trainable Parameters: {len(trainable_params)}", fontsize=12)
    plt.text(0.1, 0.5, f"Data Points: {len(time_points)}", fontsize=12)
    plt.text(0.1, 0.4, f"Direction: {'Reverse' if reverse else 'Forward'}", fontsize=12)
    plt.axis("off")
    plt.title("Training Summary")

    plt.tight_layout()

    if output_path is not None:
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        plot_path = output_path / "training_results.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


def plot_initial_state(env: SLDEnv, output_path: Path):
    """
    Plot and save the initial state of the environment.

    Args:
        env: SLDEnv instance
        output_path: Path to save the plot

    Returns:
        Path to saved plot file
    """
    plot_sld_env_state(env, errors=True, label="Initial state")
    plt.title(f"Initial State (Time point {env.time_stamp})")

    plot_path = output_path / "initial_state.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    return plot_path
