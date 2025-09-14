"""
Workflow orchestration for time-resolved neutron reflectometry RL training.

This module defines the high-level workflow for training and evaluating RL models
on time-resolved neutron reflectometry data.
"""

import os
import json
import logging
from pydantic import BaseModel, Field
from typing import List
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium.utils.env_checker import check_env

from . import __version__
from .rl_model import SLDEnv
from .reports.plotting import plot_training_results


class WorkflowConfig(BaseModel):
    """Configuration for the RL training workflow."""

    model_name: str = Field("sac", description="RL model name")
    initial_state_file: Path | None = Field(
        None, description="Path to the initial state file"
    )
    final_state_file: Path | None = Field(
        None, description="Path to the final state file"
    )
    data_location: Path = Field("", description="Location of the time-resolved data")
    output_dir: Path = Field("./output", description="Directory to save outputs")
    preview: bool = Field(False, description="Preview mode without training")
    reverse: bool = Field(False, description="Train in reverse time order")
    evaluate: bool = Field(False, description="Evaluate a trained model")
    q_resolution: float = Field(0.028, description="Q-resolution for the probe")
    n_steps: int = Field(2048, description="Number of training steps")
    allow_mixing: bool = Field(
        False, description="Allow mixing between states during training"
    )
    verbose: bool = Field(False, description="Enable verbose logging")


def create_env(config: WorkflowConfig) -> SLDEnv:
    """Create the training environment."""
    logging.info("🏗️  Setting up RL environment...")

    time_resolved_data = load_data(config.data_location)

    logging.info("📊 Environment info:")
    logging.info(f"   - Number of time points: {len(time_resolved_data)}")
    logging.info(f"   - Initial state: {config.initial_state_file}")
    logging.info(f"   - Final state: {config.final_state_file}")
    logging.info(f"   - Direction: {'reverse' if config.reverse else 'forward'}")
    logging.info(f"   - Allow mixing: {config.allow_mixing}")

    # Register the custom environment
    gym.register(
        id="timeref/SLDEnv-v1",
        entry_point=SLDEnv,
    )
    env = gym.make(
        "timeref/SLDEnv-v1",
        initial_state_file=config.initial_state_file,
        final_state_file=config.final_state_file,
        data=time_resolved_data,
        reverse=config.reverse,
        allow_mixing=config.allow_mixing,
    )
    check_env(env.unwrapped)
    env.reset()

    if hasattr(env.unwrapped, "par_labels"):
        logging.info(f"   - Trainable parameters: {len(env.unwrapped.par_labels)}")
        for i, label in enumerate(env.unwrapped.par_labels):
            logging.info(f"     {i + 1}. {label}")

    return env


def learn(env: SLDEnv, config: WorkflowConfig) -> SAC:
    """Train the RL model."""
    logging.info(f"🤖 Starting training for {config.n_steps} steps...")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SAC("MlpPolicy", env, verbose=1 if config.verbose else 0)

    # Create nice name for log directory
    fwd_bck = "fwd" if not config.reverse else "bck"
    log_dir = output_dir / f"logs-{config.model_name}-{fwd_bck}"

    progress_callback = CheckpointCallback(
        save_freq=1000,
        save_path=log_dir,
        name_prefix=f"rl_model-{fwd_bck}",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # Train the model
    model.learn(
        total_timesteps=config.n_steps, progress_bar=True, callback=progress_callback
    )
    # Save trained model
    model_path = output_dir / f"model-{config.model_name}-{fwd_bck}"
    model.save(model_path)
    logging.info(f"📄 Trained model saved in: {model_path}")

    # Save training metadata
    metadata = {
        "version": __version__,
        "data_input": str(config.data_location),
        "initial_state": str(config.initial_state_file),
        "final_state": str(config.final_state_file),
        "reverse": config.reverse,
        "allow_mixing": config.allow_mixing,
        "training_steps": config.n_steps,
        "num_time_points": len(env.unwrapped.data),
        "trainable_parameters": len(env.unwrapped.par_labels),
        "parameter_labels": env.unwrapped.par_labels,
    }

    metadata_path = os.path.join(config.output_dir, "training_metadata.json")
    with open(metadata_path, "w") as fd:
        json.dump(metadata, fd, indent=2)

    logging.info(f"🎉 Model trained successfully! Saved in: {output_dir}")
    return model


def load_model(config: WorkflowConfig) -> SAC:
    """Load a trained RL model."""
    fwd_bck = "fwd" if not config.reverse else "bck"
    model_path = config.output_dir / f"model-{config.model_name}-{fwd_bck}"

    model = SAC.load(model_path)
    logging.info(f"🤖 Model loaded from {model_path}")
    return model


def evaluate_model(env: SLDEnv, model: SAC, output_dir: Path) -> dict:
    """Evaluate a trained RL model and save the results."""
    eval_results = run_model(env, model)

    plot_training_results(
        env,
        results=eval_results,
        output_path=output_dir,
    )


def run_model(env: SLDEnv, model: SAC) -> dict:
    """Run the trained model in the environment and collect results."""

    logging.info("🧪 Running the trained model")
    n_times = len(env.data)
    obs, info = env.reset()

    # Run a full episode with the trained model
    episode_rewards = []
    episode_actions = []
    time_points = []
    chi2 = []

    done = False

    for i in range(n_times):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        episode_rewards.append(reward)
        episode_actions.append(action.copy())
        chi2.append(env.chi2)
        time_points.append(env.time_stamp)

        done = terminated or truncated
        if done:
            break

    results = {
        "episode_rewards": episode_rewards,
        "episode_actions": episode_actions,
        "time_points": time_points,
        "chi2": chi2,
        "total_reward": sum(episode_rewards),
        "final_observation": obs,
        "info": info,
        "parameter_labels": env.par_labels,
    }
    return results


def load_data(data_location: str) -> List:
    """Process and load data from the specified location."""
    if os.path.isfile(data_location):
        time_resolved_data = load_data_from_json(data_location)
    elif os.path.isdir(data_location):
        time_resolved_data = load_data_from_directory(data_location)
    else:
        raise ValueError(
            f"Data input '{data_location}' is not a valid file or directory"
        )
    return time_resolved_data


def load_data_from_json(data_path: str) -> List:
    """Load time-resolved data from JSON file."""
    with open(data_path, "r") as fd:
        data_dict = json.load(fd)
        if "data" in data_dict:
            return data_dict["data"]
        else:
            raise ValueError(f"JSON file {data_path} does not contain 'data' key")


def load_data_from_directory(data_dir: str) -> List:
    """Load time-resolved data from directory of JSON files."""
    # Placeholder for handling directory of files
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise ValueError(f"Directory {data_dir} does not exist")

    # Look for time-resolved data files
    data_files = list(data_dir.glob("r*_t*.txt"))
    return []
