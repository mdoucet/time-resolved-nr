"""
Command Line Interface for time-resolved neutron reflectometry RL training.

This module provides a CLI for training reinforcement learning models on
time-resolved neutron reflectometry data using the SLDEnv environment.
"""

import os
import json
import click
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.utils.env_checker import check_env

from . import rl_model
from . import __version__


class ProgressCallback(BaseCallback):
    """Custom callback for training progress reporting."""

    def __init__(self, total_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_steps = total_steps
        self.last_reported = 0

    def _on_step(self) -> bool:
        """Called at each step of training."""
        if self.num_timesteps % max(self.total_steps // 10, 1) == 0:
            if self.num_timesteps != self.last_reported:  # Avoid duplicate messages
                progress = self.num_timesteps / self.total_steps * 100
                click.echo(
                    f"🔄 Training progress: {progress:.1f}% ({self.num_timesteps}/{self.total_steps} steps)"
                )
                self.last_reported = self.num_timesteps
            return True


def generate_results_plot(
    output_path,
    env,
    episode_rewards,
    episode_actions,
    time_points,
    reverse,
    steps,
    trainable_params,
):
    """Generate comprehensive results visualization."""
    plt.figure(figsize=(15, 10))

    # Plot 1: Final state
    plt.subplot(2, 2, 1)
    env.unwrapped.plot(errors=True, label="Final state", newfig=False)
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
    plot_path = output_path / "training_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


def load_data_from_json(data_path):
    """Load time-resolved data from JSON file."""
    with open(data_path, "r") as fd:
        data_dict = json.load(fd)
        if "data" in data_dict:
            return data_dict["data"]
        else:
            raise ValueError(f"JSON file {data_path} does not contain 'data' key")


def load_data_from_directory(data_dir):
    """Load time-resolved data from directory of JSON files."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise ValueError(f"Directory {data_dir} does not exist")

    # Look for time-resolved data files
    json_files = list(data_dir.glob("*time-resolved*.json"))
    if not json_files:
        # Fallback to any JSON files
        json_files = list(data_dir.glob("*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in directory {data_dir}")

    # Use the first time-resolved file found
    data_file = json_files[0]
    click.echo(f"Loading data from: {data_file}")
    return load_data_from_json(data_file)


def find_state_files(data_input, initial_state, final_state):
    """Find initial and final state files, with smart defaults."""
    # If explicit files provided, use them
    if initial_state and final_state:
        return initial_state, final_state

    # If data_input is a directory, look for state files there
    if os.path.isdir(data_input):
        data_dir = Path(data_input)

        # Look for experiment files
        expt_files = list(data_dir.glob("*-expt.json")) + list(
            data_dir.glob("*expt*.json")
        )
        v2_expt_files = list(data_dir.glob("*-v2-expt.json"))

        if len(v2_expt_files) >= 2:
            # Use v2 experiment files if available
            files = sorted(v2_expt_files)
            initial_state = initial_state or str(files[0])
            final_state = final_state or str(files[-1])
        elif len(expt_files) >= 2:
            # Use regular experiment files
            files = sorted(expt_files)
            initial_state = initial_state or str(files[0])
            final_state = final_state or str(files[-1])

    return initial_state, final_state


def setup_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


@click.command()
@click.option(
    "--data",
    "-d",
    required=True,
    help="Input data: either a JSON file or directory containing data files",
)
@click.option(
    "--initial-state",
    "-i",
    help="Initial state JSON file (optional)",
)
@click.option(
    "--final-state",
    "-f",
    help="Final state JSON file (optional)",
)
@click.option(
    "--reverse/--forward",
    default=False,
    help="Direction: --reverse for backward time evolution, --forward for forward (default)",
)
@click.option(
    "--steps", "-s", default=1000, help="Number of training steps (default: 1000)"
)
@click.option(
    "--output-dir",
    "-o",
    default="./output",
    help="Output directory for trained model and results (default: ./output)",
)
@click.option(
    "--preview/--no-preview",
    default=False,
    help="Preview mode: build environment and plot initial state without training",
)
@click.option(
    "--allow-mixing/--no-mixing",
    default=False,
    help="Allow mixing between states during training",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.version_option(version=__version__, message="timeref version %(version)s")
def main(
    data,
    initial_state,
    final_state,
    reverse,
    steps,
    output_dir,
    preview,
    allow_mixing,
    verbose,
):
    """
    Time-resolved neutron reflectometry RL training CLI.

    Train reinforcement learning models on time-resolved neutron reflectometry data
    to learn parameter evolution over time.

    Examples:
        # Train on data directory

        timeref --data ./data --steps 5000 --output-dir ./results

        # Specify explicit state files

        timeref --data ./data/time-resolved.json --initial-state init.json --final-state final.json

        # Preview mode (no training)

        timeref --data ./data --preview

        # Forward time evolution

        timeref --data ./data --forward --steps 2000
    """
    try:
        click.echo(f"🚀 Time-resolved NR modeling (timeref v{__version__})")

        if verbose:
            click.echo(f"📁 Data input: {data}")
            click.echo(f"🔄 Direction: {'reverse' if reverse else 'forward'}")
            click.echo(f"📊 Training steps: {steps}")
            click.echo(f"📤 Output directory: {output_dir}")
            click.echo(f"👀 Preview mode: {preview}")

        # 1. Setup output directory
        output_path = setup_output_directory(output_dir)
        click.echo(f"📁 Output directory: {output_path}")

        # 2. Load data
        if verbose:
            click.echo("📊 Loading data...")
        if os.path.isfile(data):
            time_resolved_data = load_data_from_json(data)
        elif os.path.isdir(data):
            time_resolved_data = load_data_from_directory(data)
        else:
            raise click.BadParameter(
                f"Data input '{data}' is not a valid file or directory"
            )

        click.echo(f"✅ Loaded data with {len(time_resolved_data)} time points")

        # 3. Find state files
        initial_state, final_state = find_state_files(data, initial_state, final_state)

        if not initial_state or not final_state:
            click.echo("⚠️  Warning: Could not auto-detect state files")
            if not initial_state:
                click.echo("   Please specify --initial-state")
            if not final_state:
                click.echo("   Please specify --final-state")
            return

        if verbose:
            click.echo(f"🏁 Initial state: {initial_state}")
            click.echo(f"🎯 Final state: {final_state}")

        # 4. Register and create environment
        if verbose:
            click.echo("🏗️  Setting up RL environment...")
        gym.register(
            id="timeref/SLDEnv-v1",
            entry_point=rl_model.SLDEnv,
        )

        env = gym.make(
            "timeref/SLDEnv-v1",
            initial_state_file=initial_state,
            final_state_file=final_state,
            data=time_resolved_data,
            reverse=reverse,
            allow_mixing=allow_mixing,
        )

        # 5. Validate environment
        if verbose:
            click.echo("🔍 Validating environment...")
        check_env(env.unwrapped)
        click.echo("✅ Environment validated")

        # 6. Initialize environment and plot initial state
        state, info = env.reset()

        # Plot initial state
        plt.figure(figsize=(10, 6))
        env.unwrapped.plot(errors=True, label="Initial state", newfig=False)
        plt.title(f"Initial State (Time point {env.unwrapped.time_stamp})")
        plot_path = output_path / "initial_state.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        click.echo(f"📈 Initial state plot saved: {plot_path}")

        if not preview:
            plt.close()  # Close to save memory during training

        # 7. Training or preview

        click.echo("📊 Environment info:")
        if verbose:
            click.echo(f"   - Action space: {env.action_space}")
            click.echo(f"   - Observation space: {env.observation_space}")
        click.echo(f"   - Number of time points: {len(time_resolved_data)}")
        click.echo(f"   - Direction: {'reverse' if reverse else 'forward'}")

        # Show parameter info
        if hasattr(env.unwrapped, "par_labels"):
            click.echo(f"   - Trainable parameters: {len(env.unwrapped.par_labels)}")
            for i, label in enumerate(env.unwrapped.par_labels):
                click.echo(f"     {i + 1}. {label}")

        if preview:
            plt.show()
            return

        # 8. Training
        click.echo(f"🤖 Starting training for {steps} steps...")

        # Create SAC model
        model = SAC("MlpPolicy", env, verbose=1 if verbose else 0)

        # Create progress callback
        progress_callback = ProgressCallback(steps) if verbose else None

        # Train the model
        model.learn(total_timesteps=steps, callback=progress_callback)

        # 9. Save trained model
        model_path = output_path / "trained_model.zip"
        model.save(str(model_path))
        click.echo(f"💾 Trained model saved: {model_path}")

        # 10. Test trained model and generate results
        click.echo("🧪 Testing trained model...")
        try:
            obs, info = env.reset()

            # Run a full episode with the trained model
            episode_rewards = []
            episode_actions = []
            time_points = []

            done = False
            step_count = 0

            while not done and step_count < len(time_resolved_data):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                episode_rewards.append(reward)
                episode_actions.append(action.copy())
                time_points.append(env.unwrapped.time_stamp)

                done = terminated or truncated
                step_count += 1

            # Generate results visualization
            trainable_params = env.unwrapped.par_labels
            plot_path = generate_results_plot(
                output_path,
                env,
                episode_rewards,
                episode_actions,
                time_points,
                reverse,
                steps,
                trainable_params,
            )
            click.echo(f"📈 Training results saved: {plot_path}")

        except Exception as e:
            click.echo(f"❌ Error during model testing: {e}")
            if verbose:
                import traceback

                traceback.print_exc()
            raise click.ClickException(f"Model testing failed: {e}")

        # 11. Save training metadata
        final_reward = episode_rewards[-1] if episode_rewards else None
        metadata = {
            "version": __version__,
            "data_input": str(data),
            "initial_state": str(initial_state),
            "final_state": str(final_state),
            "reverse": reverse,
            "allow_mixing": allow_mixing,
            "training_steps": steps,
            "num_time_points": len(time_resolved_data),
            "final_reward": float(final_reward) if final_reward is not None else None,
            "episode_length": len(episode_rewards),
            "trainable_parameters": len(trainable_params),
            "parameter_labels": trainable_params,
        }

        metadata_path = output_path / "training_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        click.echo(f"📄 Training metadata saved: {metadata_path}")

        click.echo("🎉 Training completed successfully!")
        click.echo(f"📂 All results saved in: {output_path}")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if verbose:
            import traceback

            click.echo(traceback.format_exc(), err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
