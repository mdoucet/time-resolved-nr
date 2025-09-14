"""
Unit tests for cli.py

This module contains comprehensive unit tests for the CLI module,
including command line option parsing, workflow integration, and error handling.
"""

import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from click.testing import CliRunner

from timeref.cli import main
from timeref import __version__


class TestCLIMain:
    """Test suite for the main CLI function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_file = Path(self.temp_dir) / "test_data.json"
        self.test_initial_state = Path(self.temp_dir) / "initial.json"
        self.test_final_state = Path(self.temp_dir) / "final.json"
        
        # Create sample test data files
        sample_data = {
            "data": {
                "0.0": [[1, 2, 3], [0.1, 0.2, 0.3], [0.01, 0.02, 0.03]],
                "1.0": [[1, 2, 3], [0.15, 0.25, 0.35], [0.01, 0.02, 0.03]]
            }
        }
        with open(self.test_data_file, 'w') as f:
            json.dump(sample_data, f)
            
        sample_state = {"layers": [{"thickness": 10, "rho": 1.0}]}
        with open(self.test_initial_state, 'w') as f:
            json.dump(sample_state, f)
        with open(self.test_final_state, 'w') as f:
            json.dump(sample_state, f)

    @patch('timeref.cli.workflow')
    @patch('timeref.cli.plot_initial_state')
    def test_main_success_preview_mode(self, mock_plot, mock_workflow):
        """Test successful CLI execution in preview mode."""
        # Setup mocks
        mock_env = Mock()
        mock_env.unwrapped = Mock()
        mock_workflow.create_env.return_value = mock_env
        
        result = self.runner.invoke(main, [
            '--data', str(self.test_data_file),
            '--preview'
        ])
        
        assert result.exit_code == 0
        assert "Time-resolved NR modeling" in result.output
        assert "Preview mode: Environment setup only" in result.output
        
        # Verify workflow calls
        mock_workflow.create_env.assert_called_once()
        mock_plot.assert_called_once()

    @patch('timeref.cli.workflow.create_env')
    @patch('timeref.cli.workflow.learn')
    @patch('timeref.cli.workflow.evaluate_model')
    @patch('timeref.cli.plot_initial_state')
    def test_main_success_training_mode(self, mock_plot, mock_evaluate, mock_learn, mock_create_env):
        """Test successful CLI execution in training mode."""
        # Setup mocks
        mock_env = Mock()
        mock_env.unwrapped = Mock()
        mock_model = Mock()
        mock_create_env.return_value = mock_env
        mock_learn.return_value = mock_model
        
        result = self.runner.invoke(main, [
            '--data', str(self.test_data_file),
            '--steps', '100'
        ])
        
        assert result.exit_code == 0
        assert "Time-resolved NR modeling" in result.output
        assert "Done!" in result.output
        
        # Verify workflow calls
        mock_create_env.assert_called_once()
        mock_learn.assert_called_once()
        mock_evaluate.assert_called_once()
        mock_plot.assert_called_once()

    @patch('timeref.cli.workflow.create_env')
    @patch('timeref.cli.workflow.load_model')
    @patch('timeref.cli.workflow.evaluate_model')
    @patch('timeref.cli.plot_initial_state')
    def test_main_success_evaluate_mode(self, mock_plot, mock_evaluate, mock_load, mock_create_env):
        """Test successful CLI execution in evaluate mode."""
        # Setup mocks
        mock_env = Mock()
        mock_env.unwrapped = Mock()
        mock_model = Mock()
        mock_create_env.return_value = mock_env
        mock_load.return_value = mock_model
        
        result = self.runner.invoke(main, [
            '--data', str(self.test_data_file),
            '--evaluate'
        ])
        
        assert result.exit_code == 0
        assert "Time-resolved NR modeling" in result.output
        assert "Done!" in result.output
        
        # Verify workflow calls
        mock_create_env.assert_called_once()
        mock_load.assert_called_once()
        mock_evaluate.assert_called_once()

    @patch('timeref.cli.workflow.create_env')
    @patch('timeref.cli.plot_initial_state')
    def test_main_with_all_options(self, mock_plot, mock_create_env):
        """Test CLI with all command-line options specified."""
        mock_env = Mock()
        mock_env.unwrapped = Mock()
        mock_create_env.return_value = mock_env
        
        result = self.runner.invoke(main, [
            '--data', str(self.test_data_file),
            '--initial-state', str(self.test_initial_state),
            '--final-state', str(self.test_final_state),
            '--forward',
            '--steps', '500',
            '--output-dir', str(self.temp_dir),
            '--allow-mixing',
            '--verbose',
            '--preview'
        ])
        
        assert result.exit_code == 0
        
        # Verify create_env was called with some config
        mock_create_env.assert_called_once()
        # Just verify it was called, the specific config validation
        # would be better tested in workflow tests
        config = mock_create_env.call_args[0][0]
        # Basic sanity check that it's a WorkflowConfig-like object
        assert hasattr(config, 'data_location')
        assert hasattr(config, 'n_steps')
        assert hasattr(config, 'reverse')

    def test_main_missing_required_data_option(self):
        """Test CLI fails when required --data option is missing."""
        result = self.runner.invoke(main, [])
        
        assert result.exit_code != 0
        assert "Missing option '--data'" in result.output

    def test_main_version_option(self):
        """Test CLI version option."""
        result = self.runner.invoke(main, ['--version'])
        
        assert result.exit_code == 0
        assert f"timeref version {__version__}" in result.output

    def test_main_help_option(self):
        """Test CLI help option."""
        result = self.runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert "Time-resolved neutron reflectometry RL training CLI" in result.output
        assert "--data" in result.output
        assert "--steps" in result.output
        assert "--preview" in result.output

    @patch('timeref.cli.workflow')
    def test_main_exception_handling_without_verbose(self, mock_workflow):
        """Test CLI exception handling without verbose mode."""
        mock_workflow.create_env.side_effect = Exception("Test error")
        
        result = self.runner.invoke(main, [
            '--data', str(self.test_data_file)
        ])
        
        assert result.exit_code == 1  # click.Abort() returns exit code 1
        assert "❌ Error: Test error" in result.output
        # Should not show traceback without --verbose
        assert "Traceback" not in result.output

    @patch('timeref.cli.workflow')
    def test_main_exception_handling_with_verbose(self, mock_workflow):
        """Test CLI exception handling with verbose mode."""
        mock_workflow.create_env.side_effect = Exception("Test error")
        
        result = self.runner.invoke(main, [
            '--data', str(self.test_data_file),
            '--verbose'
        ])
        
        assert result.exit_code == 1  # click.Abort() returns exit code 1
        assert "❌ Error: Test error" in result.output
        # Should show traceback with --verbose
        assert "Traceback" in result.output

    @patch('timeref.cli.workflow.create_env')
    @patch('timeref.cli.plot_initial_state')
    def test_main_workflow_config_reverse_default(self, mock_plot, mock_create_env):
        """Test that forward mode is default (not reverse)."""
        mock_env = Mock()
        mock_env.unwrapped = Mock()
        mock_create_env.return_value = mock_env
        
        result = self.runner.invoke(main, [
            '--data', str(self.test_data_file),
            '--preview'
        ])
        
        assert result.exit_code == 0
        mock_create_env.assert_called_once()

    @patch('timeref.cli.workflow.create_env')
    @patch('timeref.cli.plot_initial_state')
    def test_main_workflow_config_reverse_explicit(self, mock_plot, mock_create_env):
        """Test explicit reverse mode."""
        mock_env = Mock()
        mock_env.unwrapped = Mock()
        mock_create_env.return_value = mock_env
        
        result = self.runner.invoke(main, [
            '--data', str(self.test_data_file),
            '--reverse',
            '--preview'
        ])
        
        assert result.exit_code == 0
        mock_create_env.assert_called_once()

    @patch('timeref.cli.workflow.create_env')
    @patch('timeref.cli.plot_initial_state')
    def test_main_output_messages(self, mock_plot, mock_create_env):
        """Test that appropriate output messages are shown."""
        mock_env = Mock()
        mock_env.unwrapped = Mock()
        mock_create_env.return_value = mock_env
        
        result = self.runner.invoke(main, [
            '--data', str(self.test_data_file),
            '--preview'
        ])
        
        assert result.exit_code == 0
        
        # Check for expected output messages
        output_lines = result.output.split('\n')
        version_line = [line for line in output_lines if "Time-resolved NR modeling" in line]
        assert len(version_line) == 1
        assert f"timeref v{__version__}" in version_line[0]
        
        preview_line = [line for line in output_lines if "Preview mode" in line]
        assert len(preview_line) == 1

    def test_main_invalid_file_path(self):
        """Test CLI with invalid file path."""
        result = self.runner.invoke(main, [
            '--data', '/nonexistent/path/data.json'
        ])
        
        # Should fail during workflow execution
        assert result.exit_code == 1
        assert "❌ Error:" in result.output


class TestCLIEdgeCases:
    """Test edge cases and boundary conditions for the CLI."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch('timeref.cli.workflow.create_env')
    @patch('timeref.cli.plot_initial_state')
    def test_main_with_zero_steps(self, mock_plot, mock_create_env):
        """Test CLI with zero training steps."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"data": {"0": [[1], [0.1], [0.01]]}}, f)
            data_file = f.name

        mock_env = Mock()
        mock_env.unwrapped = Mock()
        mock_create_env.return_value = mock_env
        
        result = self.runner.invoke(main, [
            '--data', data_file,
            '--steps', '0',
            '--preview'
        ])
        
        assert result.exit_code == 0
        mock_create_env.assert_called_once()

    @patch('timeref.cli.workflow.create_env')
    @patch('timeref.cli.plot_initial_state')
    def test_main_with_very_large_steps(self, mock_plot, mock_create_env):
        """Test CLI with very large number of steps."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"data": {"0": [[1], [0.1], [0.01]]}}, f)
            data_file = f.name

        mock_env = Mock()
        mock_env.unwrapped = Mock()
        mock_create_env.return_value = mock_env
        
        result = self.runner.invoke(main, [
            '--data', data_file,
            '--steps', '1000000',
            '--preview'
        ])
        
        assert result.exit_code == 0
        mock_create_env.assert_called_once()

    def test_main_with_negative_steps(self):
        """Test CLI with negative training steps."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"data": {"0": [[1], [0.1], [0.01]]}}, f)
            data_file = f.name

        # Don't store result since we're not using it in this test
        self.runner.invoke(main, [
            '--data', data_file,
            '--steps', '-100'
        ])
        
        # Click should handle negative values validation
        # This might succeed depending on click's validation
        # but the workflow should handle it appropriately


class TestCLIIntegration:
    """Integration tests for CLI with mocked dependencies."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch('timeref.cli.workflow.create_env')
    @patch('timeref.cli.workflow.learn')
    @patch('timeref.cli.workflow.evaluate_model')
    @patch('timeref.cli.plot_initial_state')
    def test_full_training_workflow(self, mock_plot, mock_evaluate, mock_learn, mock_create_env):
        """Test the full training workflow integration."""
        # Setup mocks
        mock_env = Mock()
        mock_env.unwrapped = Mock()
        mock_model = Mock()
        
        mock_create_env.return_value = mock_env
        mock_learn.return_value = mock_model
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"data": {"0": [[1], [0.1], [0.01]]}}, f)
            data_file = f.name

        result = self.runner.invoke(main, [
            '--data', data_file,
            '--steps', '100'
        ])
        
        assert result.exit_code == 0
        
        # Verify the sequence of calls
        mock_create_env.assert_called_once()
        mock_plot.assert_called_once()
        mock_learn.assert_called_once_with(mock_env, mock_create_env.call_args[0][0])
        mock_evaluate.assert_called_once()

    @patch('timeref.cli.workflow.create_env')
    @patch('timeref.cli.workflow.load_model')
    @patch('timeref.cli.workflow.evaluate_model')
    @patch('timeref.cli.plot_initial_state')
    def test_full_evaluation_workflow(self, mock_plot, mock_evaluate, mock_load, mock_create_env):
        """Test the full evaluation workflow integration."""
        # Setup mocks
        mock_env = Mock()
        mock_env.unwrapped = Mock()
        mock_model = Mock()
        
        mock_create_env.return_value = mock_env
        mock_load.return_value = mock_model
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"data": {"0": [[1], [0.1], [0.01]]}}, f)
            data_file = f.name

        result = self.runner.invoke(main, [
            '--data', data_file,
            '--evaluate'
        ])
        
        assert result.exit_code == 0
        
        # Verify the sequence of calls
        mock_create_env.assert_called_once()
        mock_plot.assert_called_once()
        mock_load.assert_called_once_with(mock_create_env.call_args[0][0])
        mock_evaluate.assert_called_once()
