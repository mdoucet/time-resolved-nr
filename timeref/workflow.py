"""
Workflow orchestration for time-resolved neutron reflectometry RL training.

This module defines the high-level workflow for training and evaluating RL models
on time-resolved neutron reflectometry data.
"""

import os
import json
import logging
from pydantic import BaseModel, Field
from typing import Optional, List, Callable
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium.utils.env_checker import check_env

from .rl_model import SLDEnv


class WorkflowConfig(BaseModel):
    """Configuration for the RL training workflow."""

    model_name: str = Field("sac", description="RL model name")
    initial_state_file: str = Field(..., description="Path to the initial state file")
    final_state_file: str = Field(..., description="Path to the final state file")
    data_location: str = Field(..., description="Location of the time-resolved data")
    output_dir: str | None = Field(..., description="Directory to save outputs")
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
    # Get data
    time_resolved_data = load_data(config.data_location)

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
    return env


def learn(env: SLDEnv, config: WorkflowConfig) -> SAC:
    """Train the RL model."""
    model = SAC("MlpPolicy", env, verbose=1 if config.verbose else 0)

    # Create nice name for log directory
    if config.output_dir is not None:
        fwd_bck = "fwd" if not config.reverse else "bck"
        log_dir = os.path.join(config.output_dir, f"logs-{config.model_name}-{fwd_bck}")

        progress_callback = CheckpointCallback(
            save_freq=1000,
            save_path=log_dir,
            name_prefix=f"rl_model-{fwd_bck}",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
    else:
        progress_callback = None

    # Train the model
    model.learn(
        total_timesteps=config.n_steps, progress_bar=True, callback=progress_callback
    )
    # Save trained model
    if config.output_dir is not None:
        model_path = os.path.join(
            config.output_dir, f"model-{config.model_name}-{fwd_bck}.zip"
        )
        model.save(model_path)
        logging.info(f"Trained model saved: {model_path}")
    return model


def load_model(model_path: str, env: SLDEnv) -> SAC:
    """Load a trained RL model."""
    if not os.path.isfile(model_path):
        raise ValueError(f"Model file {model_path} does not exist")
    model = SAC.load(model_path, env=env)
    return model


def run_model(env: SLDEnv, model: SAC) -> dict:
    """Run the trained model in the environment and collect results."""
    episode_reward = 0
    n_times = len(env.data)
    obs, info = env.reset()

    # Run a full episode with the trained model
    episode_rewards = []
    episode_actions = []
    time_points = []
    actions = []
    chi2 = []

    done = False

    for i in range(n_times):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        episode_rewards.append(reward)
        episode_actions.append(action.copy())
        time_points.append(env.unwrapped.time_stamp)

        done = terminated or truncated
        if done:
            break

    results = {
        "episode_rewards": episode_rewards,
        "episode_actions": episode_actions,
        "time_points": time_points,
        "final_observation": obs,
        "info": info,
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
