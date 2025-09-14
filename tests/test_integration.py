import os
import numpy as np
from pathlib import Path

from stable_baselines3 import SAC

from timeref import workflow
from timeref.reports.plotting import plot_initial_state, plot_training_results


class TestIntegration:
    """Integration tests for the SLDEnv class"""

    def test_integration(self):
        """Test the integration of all components"""

        dirname, _ = os.path.split(__file__)
        data_dir = os.path.join(dirname, "..", "data")

        initial_state_expt_file = os.path.join(data_dir, "189237-v2-expt.json")

        final_state_expt_file = os.path.join(data_dir, "189246-v2-expt.json")

        data_file = os.path.join(data_dir, "r189245-time-resolved.json")

        # Create workflow configuration
        workflow_config = workflow.WorkflowConfig(
            model_name="sac",
            initial_state_file=initial_state_expt_file,
            final_state_file=final_state_expt_file,
            data_location=data_file,
            output_dir="/tmp",
            reverse=False,
            q_resolution=0.028,
            n_steps=10,
        )

        # Create environment
        env = workflow.create_env(workflow_config)

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

        output_path = Path("/tmp")
        plot_initial_state(env.unwrapped, output_path=output_path, show=False)

        # Short training to see if everything works
        model = workflow.learn(env, workflow_config)
        assert isinstance(model, SAC)

        results = workflow.run_model(env.unwrapped, model)

        plot_training_results(env.unwrapped, results=results, output_path=output_path)
