import os
import json
import numpy as np
import pytest

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import SAC

from timeref import rl_model


class TestIntegration:
    """Integration tests for the SLDEnv class"""

    def test_integration(self):
        """Test the integration of all components"""

        dirname, _ = os.path.split(__file__)
        data_dir = os.path.join(dirname, "..", "data")

        initial_state_expt_file = os.path.join(data_dir, "189237-v2-expt.json")

        final_state_expt_file = os.path.join(data_dir, "189246-v2-expt.json")

        data_file = os.path.join(data_dir, "r189245-time-resolved.json")

        with open(data_file) as fd:
            m = json.load(fd)
            _data = m["data"]
            print("Number of times: %s" % len(_data))

        # Register the environment so we can create it with gym.make()
        gym.register(
            id="rl_model/SLDEnv-v1",
            entry_point=rl_model.SLDEnv,
        )
        # Create an instance of our custom environment
        env = gym.make(
            "rl_model/SLDEnv-v1",
            initial_state_file=initial_state_expt_file,
            final_state_file=final_state_expt_file,
            data=_data,
            reverse=False,
        )

        # use the Gymnasium 'check_env' function to check the environment
        # - returns nothing if the environment is verified as ok
        check_env(env.unwrapped)

        # initialize the environment
        state, info = env.reset()
        assert state == 0
        assert info == {}

        # Test an action
        action = env.unwrapped.normalized_parameters

        # take the action and get the information from the environment
        _, reward, _, _, info = env.step(action)
        assert np.isclose(reward, -1.3509, atol=1e-4)
        assert np.isclose(action[0], 0.22399, atol=1e-3)
        assert np.isclose(action[1], 0.30617, atol=1e-3)
        assert np.isclose(action[2], 0.58477, atol=1e-3)

        env.unwrapped.plot(errors=True)

        # Short training to see if everything works
        model = SAC("MlpPolicy", env, use_sde=False, verbose=0)
        model.learn(10)

