"""
Tests for plotting functions in the reports module.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

from timeref.reports.plotting import plot_parameter_evolution, plot_reflectivity_evolution


class TestPlotParameterEvolution(unittest.TestCase):
    """Test the plot_parameter_evolution function."""

    def setUp(self):
        """Set up test data."""
        # Mock environment
        self.mock_env = Mock()
        self.mock_env.par_labels = ["interface_width", "polymer_thickness", "polymer_sld"]
        self.mock_env.parameters = np.array([1.0, 2.0, 3.0])
        self.mock_env.end_parameters = np.array([1.5, 2.5, 3.5])
        self.mock_env.reverse = False
        
        # Mock the conversion methods
        converted_params = np.array([
            [1.0, 1.1, 1.2, 1.3],  # param 1 over time
            [2.0, 2.1, 2.2, 2.3],  # param 2 over time
            [3.0, 3.1, 3.2, 3.3],  # param 3 over time
        ])
        converted_uncertainties = np.array([
            [0.1, 0.1, 0.1, 0.1],  # uncertainties for param 1
            [0.1, 0.1, 0.1, 0.1],  # uncertainties for param 2
            [0.1, 0.1, 0.1, 0.1],  # uncertainties for param 3
        ])
        
        self.mock_env.convert_action_to_parameters.return_value = converted_params.T
        self.mock_env.convert_action_uncertainties_to_parameters.return_value = converted_uncertainties.T
        
        # Mock results from run_model
        self.mock_results = {
            "episode_actions": np.array([
                [1.0, 2.0, 3.0],  # Time point 1
                [1.1, 2.1, 3.1],  # Time point 2
                [1.2, 2.2, 3.2],  # Time point 3
                [1.3, 2.3, 3.3],  # Time point 4
            ]),
            "actions_uncertainties": np.array([
                [0.1, 0.1, 0.1],  # Uncertainties for time point 1
                [0.1, 0.1, 0.1],  # Uncertainties for time point 2
                [0.1, 0.1, 0.1],  # Uncertainties for time point 3
                [0.1, 0.1, 0.1],  # Uncertainties for time point 4
            ]),
            "time_points": np.array([0, 100, 200, 300]),
            "parameter_labels": ["interface_width", "polymer_thickness", "polymer_sld"],
            "episode_rewards": [0.5, 0.6, 0.7, 0.8],
            "chi2": [10.0, 8.0, 6.0, 4.0]
        }

    @patch('timeref.reports.plotting.plt')
    @patch('builtins.print')  # Suppress print statements
    def test_plot_parameter_evolution_basic(self, mock_print, mock_plt):
        """Test basic functionality of plot_parameter_evolution."""
        # Mock matplotlib
        mock_plt.subplots.return_value = (MagicMock(), MagicMock())
        mock_plt.subplot.return_value = MagicMock()
        
        # Call the function
        result = plot_parameter_evolution(
            self.mock_env,
            self.mock_results,
            show=True
        )
        
        # Verify plotting calls were made
        mock_plt.subplots.assert_called_once()
        mock_plt.errorbar.assert_called()  # Changed from plot to errorbar
        mock_plt.xlabel.assert_called_with("Time Steps")
        
        # Should return None when no output_path
        self.assertIsNone(result)

    @patch('timeref.reports.plotting.plt')
    @patch('builtins.print')  # Suppress print statements
    def test_plot_parameter_evolution_with_output(self, mock_print, mock_plt):
        """Test plot_parameter_evolution with output path."""
        # Mock matplotlib
        mock_plt.subplots.return_value = (MagicMock(), MagicMock())
        mock_plt.subplot.return_value = MagicMock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # Call the function
            result = plot_parameter_evolution(
                self.mock_env,
                self.mock_results,
                output_path=output_path
            )
            
            # Verify file path is returned
            expected_path = output_path / "parameter_evolution.png"
            self.assertEqual(result, expected_path)
            
            # Verify save was called
            mock_plt.savefig.assert_called_with(
                expected_path, 
                dpi=150, 
                bbox_inches="tight"
            )

    @patch('timeref.reports.plotting.plt')
    @patch('builtins.print')  # Suppress print statements
    def test_plot_parameter_evolution_basic_functionality(self, mock_print, mock_plt):
        """Test basic functionality without bayesian fits."""
        # Mock matplotlib
        mock_plt.subplots.return_value = (MagicMock(), MagicMock())
        mock_plt.subplot.return_value = MagicMock()
        
        # Call the function without bayesian fits
        result = plot_parameter_evolution(
            self.mock_env,
            self.mock_results,
            show=True
        )
        
        # Verify plotting calls were made
        mock_plt.subplots.assert_called_once()
        mock_plt.errorbar.assert_called()  # Changed from plot to errorbar
        mock_plt.xlabel.assert_called_with("Time Steps")
        
        self.assertIsNone(result)

    @patch('timeref.reports.plotting.plt')
    @patch('builtins.print')  # Suppress print statements
    def test_plot_parameter_evolution_custom_parameters(self, mock_print, mock_plt):
        """Test plot_parameter_evolution with custom initial/final parameters."""
        # Mock matplotlib
        mock_plt.subplots.return_value = (MagicMock(), MagicMock())
        mock_plt.subplot.return_value = MagicMock()
        
        initial_params = np.array([0.5, 1.5, 2.5])
        final_params = np.array([1.5, 2.5, 3.5])
        
        # Call the function
        result = plot_parameter_evolution(
            self.mock_env,
            self.mock_results,
            initial_parameters=initial_params,
            final_parameters=final_params,
            show=True
        )
        
        # Verify plotting calls were made
        mock_plt.subplots.assert_called_once()
        mock_plt.errorbar.assert_called()  # Changed from plot to errorbar
        
        self.assertIsNone(result)


class TestPlotReflectivityEvolution(unittest.TestCase):
    """Test the plot_reflectivity_evolution function."""

    def setUp(self):
        """Set up test data."""
        # Mock environment
        self.mock_env = Mock()
        self.mock_env.reset.return_value = (Mock(), Mock())
        self.mock_env.step.return_value = (Mock(), 0.5, False, False, Mock())
        self.mock_env.data = ["data1", "data2", "data3", "data4"]
        
        # Mock model
        self.mock_model = Mock()
        self.mock_model.predict.return_value = (np.array([1.0, 2.0, 3.0]), None)
        
        # Mock results from run_model
        self.mock_results = {
            "episode_actions": [
                np.array([1.0, 2.0, 3.0]),  # Time point 1
                np.array([1.1, 2.1, 3.1]),  # Time point 2
                np.array([1.2, 2.2, 3.2]),  # Time point 3
                np.array([1.3, 2.3, 3.3]),  # Time point 4
            ],
            "time_points": [0, 100, 200, 300],
            "parameter_labels": ["interface_width", "polymer_thickness", "polymer_sld"],
            "episode_rewards": [0.5, 0.6, 0.7, 0.8],
            "chi2": [10.0, 8.0, 6.0, 4.0]
        }

    @patch('timeref.reports.plotting.plt')
    @patch('timeref.reports.plotting.plot_sld_env_state')
    def test_plot_reflectivity_evolution_basic(self, mock_plot_sld, mock_plt):
        """Test basic functionality of plot_reflectivity_evolution."""
        # Mock matplotlib
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.get_legend_handles_labels.return_value = ([], [])  # Empty legend
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        # Call the function
        result = plot_reflectivity_evolution(
            self.mock_env,
            self.mock_model,
            self.mock_results,
            show=True
        )
        
        # Verify environment was reset
        self.mock_env.reset.assert_called_once()
        
        # Verify matplotlib calls
        mock_plt.subplots.assert_called_once()
        # Note: xlim is no longer called in the updated function
        
        # Verify plot_sld_env_state was called for each time point
        self.assertGreater(mock_plot_sld.call_count, 0)
        
        # Should return None when no output_path
        self.assertIsNone(result)

    @patch('timeref.reports.plotting.plt')
    @patch('timeref.reports.plotting.plot_sld_env_state')
    def test_plot_reflectivity_evolution_with_output(self, mock_plot_sld, mock_plt):
        """Test plot_reflectivity_evolution with output path."""
        # Mock matplotlib
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.get_legend_handles_labels.return_value = ([], [])  # Empty legend
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # Call the function
            result = plot_reflectivity_evolution(
                self.mock_env,
                self.mock_model,
                self.mock_results,
                output_path=output_path
            )
            
            # Verify file path is returned
            expected_path = output_path / "reflectivity_evolution.png"
            self.assertEqual(result, expected_path)
            
            # Verify save was called
            mock_plt.savefig.assert_called_with(
                expected_path, 
                dpi=150, 
                bbox_inches="tight"
            )

    @patch('timeref.reports.plotting.plt')
    @patch('timeref.reports.plotting.plot_sld_env_state')
    def test_plot_reflectivity_evolution_custom_params(self, mock_plot_sld, mock_plt):
        """Test plot_reflectivity_evolution with custom parameters."""
        # Mock matplotlib
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.get_legend_handles_labels.return_value = ([], [])  # Empty legend
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        # Call the function with custom parameters (q_range removed)
        result = plot_reflectivity_evolution(
            self.mock_env,
            self.mock_model,
            self.mock_results,
            figsize=(8, 12),
            show=True
        )
        
        # Verify matplotlib calls with custom parameters
        mock_plt.subplots.assert_called_with(dpi=120, figsize=(8, 12))
        
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
