"""
Command Line Interface for time-resolved neutron reflectometry RL training.

This module provides a CLI for training reinforcement learning models on
time-resolved neutron reflectometry data using the SLDEnv environment.
"""

import json
import click
from pathlib import Path

from . import workflow
from . import __version__
from .reports.plotting import plot_initial_state, plot_training_results


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

    Train a reinforcement learning model on time-resolved neutron reflectometry data
    to learn parameter evolution over time.
    """
    try:
        click.echo(f"🚀 Time-resolved NR modeling (timeref v{__version__})")

        config = workflow.WorkflowConfig(
            initial_state_file=initial_state,
            final_state_file=final_state,
            data_location=data,
            output_dir=output_dir,
            preview=preview,
            reverse=reverse,
            n_steps=steps,
            allow_mixing=allow_mixing,
        )

        # Setup output directory
        output_path = setup_output_directory(output_dir)
        click.echo(f"📁 Output directory: {output_path}")

        # Load data
        time_resolved_data = workflow.load_data(data)
        click.echo(f"✅ Loaded data with {len(time_resolved_data)} time points")

        click.echo(f"🏁 Initial state: {initial_state}")
        click.echo(f"🎯 Final state: {final_state}")

        # Register and create environment
        click.echo("🏗️  Setting up RL environment...")
        env = workflow.create_env(config)

        # Plot initial state
        plot_path = plot_initial_state(env.unwrapped, output_path)
        click.echo(f"📈 Initial state plot saved: {plot_path}")

        click.echo("📊 Environment info:")
        click.echo(f"   - Number of time points: {len(time_resolved_data)}")
        click.echo(f"   - Direction: {'reverse' if reverse else 'forward'}")

        # Show parameter info
        if hasattr(env.unwrapped, "par_labels"):
            click.echo(f"   - Trainable parameters: {len(env.unwrapped.par_labels)}")
            for i, label in enumerate(env.unwrapped.par_labels):
                click.echo(f"     {i + 1}. {label}")

        if preview:
            click.echo("👀 Preview mode: Environment setup complete, skipping training")
            return

        # Training
        if not config.evaluate:
            click.echo(f"🤖 Starting training for {steps} steps...")
            model = workflow.learn(env, config)
            click.echo("✅ Model trained successfully.")
        else:
            click.echo("🤖 Loading trained model for evaluation...")
            model = workflow.load_model(config.model_path, env)
            click.echo("✅ Model loaded successfully.")

        # Test trained model and generate results
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
            plot_path = plot_training_results(
                env.unwrapped,
                episode_rewards=episode_rewards,
                episode_actions=episode_actions,
                time_points=time_points,
                reverse=reverse,
                steps=steps,
                trainable_params=trainable_params,
                output_path=output_path,
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
