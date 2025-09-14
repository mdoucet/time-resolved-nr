"""
Command Line Interface for time-resolved neutron reflectometry RL training.

This module provides a CLI for training reinforcement learning models on
time-resolved neutron reflectometry data using the SLDEnv environment.
"""

import logging
from pathlib import Path
import click

from . import workflow
from . import __version__
from .reports.plotting import plot_initial_state


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
@click.option(
    "--evaluate/--no-evaluate",
    default=False,
    help="Evaluate a trained model instead of training",
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
    evaluate,
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
            evaluate=evaluate,
        )

        # Register and create environment
        env = workflow.create_env(config)

        # Plot initial state
        plot_initial_state(env.unwrapped, config.output_dir, show=preview)
        if preview:
            click.echo("👀 Preview mode: Environment setup only")
            return

        # Training
        if not config.evaluate:
            model = workflow.learn(env, config)
        else:
            model = workflow.load_model(config)

        # Test trained model and generate results
        workflow.evaluate_model(env.unwrapped, model, config.output_dir)

        click.echo("✅ Done!")
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if verbose:
            import traceback

            click.echo(traceback.format_exc(), err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
