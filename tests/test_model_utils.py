"""
Unit tests for model_utils.py

This module contains comprehensive unit tests for all functions in model_utils.py,
including edge cases and error conditions.
"""

import json
import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

# Import the module under test
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model_utils import (
    sample_from_json, 
    expt_from_json_file, 
    calculate_reflectivity,
    ERR_MIN_ROUGH,
    ERR_MIN_THICK,
    ERR_MIN_RHO
)

from .fixtures import (
    sample_model_expt_json,
    sample_model_err_json,
    minimal_model_expt_json
)


class TestSampleFromJson:
    """Test cases for sample_from_json function"""

    @patch('model_utils.SLD')
    @patch('model_utils.Slab')
    def test_sample_from_json_basic(self, mock_slab, mock_sld, sample_model_expt_json):
        """Test basic functionality with valid JSON data"""
        # Mock the SLD and Slab objects
        mock_material = Mock()
        mock_sld.return_value = mock_material
        
        mock_slab_instance = Mock()
        mock_slab.return_value = mock_slab_instance
        
        # Mock the | operator for combining slabs
        mock_slab_instance.__or__ = Mock(return_value=mock_slab_instance)
        
        # Mock the material properties
        mock_slab_instance.material = Mock()
        mock_slab_instance.material.rho = Mock()
        mock_slab_instance.material.irho = Mock()
        mock_slab_instance.thickness = Mock()
        mock_slab_instance.interface = Mock()
        
        # Call the function
        result = sample_from_json(sample_model_expt_json)
        
        # Verify SLD was called with correct parameters for each layer
        assert mock_sld.call_count == 2  # Two layers in test data
        
        # Verify first layer call
        first_call = mock_sld.call_args_list[0]
        assert first_call[1]['name'] == 'substrate'
        assert first_call[1]['rho'] == 4.5e-6
        assert first_call[1]['irho'] == 1.0e-8
        
        # Verify Slab was called
        assert mock_slab.call_count == 2
        
        # Verify the result is properly combined
        assert result is not None

    @patch('model_utils.SLD')
    @patch('model_utils.Slab')
    def test_sample_from_json_with_errors(self, mock_slab, mock_sld, 
                                         sample_model_expt_json, sample_model_err_json):
        """Test functionality with error data provided"""
        # Mock the SLD and Slab objects
        mock_material = Mock()
        mock_sld.return_value = mock_material
        
        mock_slab_instance = Mock()
        mock_slab.return_value = mock_slab_instance
        
        # Mock the | operator for combining slabs
        mock_slab_instance.__or__ = Mock(return_value=mock_slab_instance)
        
        # Mock the material properties with dev method
        mock_param = Mock()
        mock_param.dev = Mock()
        mock_param.range = Mock()
        mock_param.fixed = False
        
        mock_slab_instance.material = Mock()
        mock_slab_instance.material.rho = mock_param
        mock_slab_instance.material.irho = mock_param
        mock_slab_instance.thickness = mock_param
        mock_slab_instance.interface = mock_param
        
        # Call the function with error data
        result = sample_from_json(
            sample_model_expt_json, 
            model_err_json=sample_model_err_json,
            prior_scale=1.0
        )
        
        # Verify dev was called for parameters with error data
        assert mock_param.dev.called
        assert result is not None

    @patch('model_utils.SLD')
    @patch('model_utils.Slab')
    def test_sample_from_json_prior_scale_zero(self, mock_slab, mock_sld,
                                              sample_model_expt_json, sample_model_err_json):
        """Test with prior_scale=0 should not apply error distributions"""
        mock_material = Mock()
        mock_sld.return_value = mock_material
        
        mock_slab_instance = Mock()
        mock_slab.return_value = mock_slab_instance
        
        # Mock the | operator for combining slabs
        mock_slab_instance.__or__ = Mock(return_value=mock_slab_instance)
        
        mock_param = Mock()
        mock_param.dev = Mock()
        mock_param.range = Mock()
        mock_param.fixed = False
        
        mock_slab_instance.material = Mock()
        mock_slab_instance.material.rho = mock_param
        mock_slab_instance.material.irho = mock_param
        mock_slab_instance.thickness = mock_param
        mock_slab_instance.interface = mock_param
        
        # Call with prior_scale=0
        result = sample_from_json(
            sample_model_expt_json,
            model_err_json=sample_model_err_json,
            prior_scale=0
        )
        
        # Should use range instead of dev
        assert mock_param.range.called
        assert not mock_param.dev.called
        assert result is not None

    @patch('model_utils.SLD')
    @patch('model_utils.Slab')
    def test_sample_from_json_set_ranges_true(self, mock_slab, mock_sld, sample_model_expt_json):
        """Test with set_ranges=True"""
        mock_material = Mock()
        mock_sld.return_value = mock_material
        
        mock_slab_instance = Mock()
        mock_slab.return_value = mock_slab_instance
        
        # Mock the | operator for combining slabs
        mock_slab_instance.__or__ = Mock(return_value=mock_slab_instance)
        
        mock_param = Mock()
        mock_param.range = Mock()
        mock_param.fixed = False
        
        mock_slab_instance.material = Mock()
        mock_slab_instance.material.rho = mock_param
        mock_slab_instance.material.irho = mock_param
        mock_slab_instance.thickness = mock_param
        mock_slab_instance.interface = mock_param
        
        result = sample_from_json(sample_model_expt_json, set_ranges=True)
        
        # When set_ranges=True, parameters should not be fixed
        assert not mock_param.fixed
        assert result is not None

    @patch('model_utils.SLD')
    @patch('model_utils.Slab')
    def test_sample_from_json_fixed_parameters(self, mock_slab, mock_sld, minimal_model_expt_json):
        """Test with all parameters fixed"""
        mock_material = Mock()
        mock_sld.return_value = mock_material
        
        mock_slab_instance = Mock()
        mock_slab.return_value = mock_slab_instance
        
        mock_slab_instance.material = Mock()
        mock_slab_instance.material.rho = Mock()
        mock_slab_instance.material.irho = Mock()
        mock_slab_instance.thickness = Mock()
        mock_slab_instance.interface = Mock()
        
        result = sample_from_json(minimal_model_expt_json)
        
        # Should not call range or dev methods for fixed parameters
        assert not mock_slab_instance.material.rho.range.called
        assert not mock_slab_instance.material.irho.range.called
        assert result is not None

    def test_error_constants(self):
        """Test that error constants are properly defined"""
        assert ERR_MIN_ROUGH == 1
        assert ERR_MIN_THICK == 1
        assert ERR_MIN_RHO == 0.2


class TestExptFromJsonFile:
    """Test cases for expt_from_json_file function"""

    @patch('model_utils.serialize.deserialize')
    @patch('model_utils.json.loads')
    @patch('builtins.open', new_callable=mock_open)
    def test_expt_from_json_file_basic(self, mock_file, mock_json_loads, mock_deserialize):
        """Test basic file loading without probe override"""
        # Setup mocks
        mock_json_loads.return_value = {"mock": "data"}
        mock_experiment = Mock()
        mock_deserialize.return_value = mock_experiment
        
        mock_file.return_value.read.return_value = '{"mock": "data"}'
        
        # Call function
        result = expt_from_json_file("test_file.json")
        
        # Verify file operations
        mock_file.assert_called_once_with("test_file.json", "rt")
        mock_json_loads.assert_called_once_with('{"mock": "data"}')
        mock_deserialize.assert_called_once_with({"mock": "data"}, migration=True)
        
        assert result == mock_experiment

    @patch('model_utils.Experiment')
    @patch('model_utils.serialize.deserialize')
    @patch('model_utils.json.loads')
    @patch('builtins.open', new_callable=mock_open)
    def test_expt_from_json_file_with_probe(self, mock_file, mock_json_loads, 
                                           mock_deserialize, mock_experiment_class):
        """Test file loading with probe override"""
        # Setup mocks
        mock_json_loads.return_value = {"mock": "data"}
        mock_original_experiment = Mock()
        mock_original_experiment.sample = Mock()
        mock_deserialize.return_value = mock_original_experiment
        
        mock_new_experiment = Mock()
        mock_experiment_class.return_value = mock_new_experiment
        
        mock_probe = Mock()
        
        mock_file.return_value.read.return_value = '{"mock": "data"}'
        
        # Call function with probe
        result = expt_from_json_file("test_file.json", probe=mock_probe)
        
        # Verify new experiment was created with provided probe
        mock_experiment_class.assert_called_once_with(
            probe=mock_probe, 
            sample=mock_original_experiment.sample
        )
        
        assert result == mock_new_experiment

    @patch('builtins.open', side_effect=FileNotFoundError("File not found"))
    def test_expt_from_json_file_not_found(self, mock_file):
        """Test behavior when file is not found"""
        with pytest.raises(FileNotFoundError):
            expt_from_json_file("nonexistent_file.json")

    @patch('model_utils.json.loads', side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
    @patch('builtins.open', new_callable=mock_open)
    def test_expt_from_json_file_invalid_json(self, mock_file, mock_json_loads):
        """Test behavior with invalid JSON"""
        mock_file.return_value.read.return_value = 'invalid json'
        
        with pytest.raises(json.JSONDecodeError):
            expt_from_json_file("invalid.json")


class TestCalculateReflectivity:
    """Test cases for calculate_reflectivity function"""

    @patch('model_utils.expt_from_json_file')
    def test_calculate_reflectivity_basic(self, mock_expt_from_json):
        """Test basic reflectivity calculation"""
        # Setup mock experiment
        mock_experiment = Mock()
        mock_experiment.reflectivity.return_value = (Mock(), [0.1, 0.05, 0.02])
        mock_expt_from_json.return_value = mock_experiment
        
        # Test data
        q_values = [0.01, 0.02, 0.03]
        
        # Call function
        result = calculate_reflectivity("test_model.json", q_values)
        
        # Verify experiment loading
        mock_expt_from_json.assert_called_once_with("test_model.json", q_values, q_resolution=0.025)
        
        # Verify reflectivity calculation
        mock_experiment.reflectivity.assert_called_once()
        
        # Verify result
        assert result == [0.1, 0.05, 0.02]

    @patch('model_utils.expt_from_json_file')
    def test_calculate_reflectivity_custom_resolution(self, mock_expt_from_json):
        """Test reflectivity calculation with custom resolution"""
        mock_experiment = Mock()
        mock_experiment.reflectivity.return_value = (Mock(), [0.2, 0.1])
        mock_expt_from_json.return_value = mock_experiment
        
        q_values = [0.01, 0.02]
        custom_resolution = 0.05
        
        result = calculate_reflectivity("test_model.json", q_values, q_resolution=custom_resolution)
        
        mock_expt_from_json.assert_called_once_with(
            "test_model.json", 
            q_values, 
            q_resolution=custom_resolution
        )
        
        assert result == [0.2, 0.1]

    @patch('model_utils.expt_from_json_file', side_effect=FileNotFoundError("File not found"))
    def test_calculate_reflectivity_file_not_found(self, mock_expt_from_json):
        """Test behavior when model file is not found"""
        with pytest.raises(FileNotFoundError):
            calculate_reflectivity("nonexistent.json", [0.01, 0.02])


class TestIntegration:
    """Integration tests for model_utils functions"""

    @patch('model_utils.SLD')
    @patch('model_utils.Slab')
    @patch('model_utils.serialize.deserialize')
    @patch('model_utils.json.loads')
    @patch('builtins.open', new_callable=mock_open)
    def test_workflow_sample_and_experiment(self, mock_file, mock_json_loads, 
                                          mock_deserialize, mock_slab, mock_sld,
                                          sample_model_expt_json):
        """Test a typical workflow using both sample_from_json and expt_from_json_file"""
        # Setup mocks for sample creation
        mock_material = Mock()
        mock_sld.return_value = mock_material
        
        mock_slab_instance = Mock()
        mock_slab.return_value = mock_slab_instance
        mock_slab_instance.material = Mock()
        
        # Mock the | operator for combining slabs
        mock_slab_instance.__or__ = Mock(return_value=mock_slab_instance)
        
        # Setup mocks for experiment loading
        mock_json_loads.return_value = sample_model_expt_json
        mock_experiment = Mock()
        mock_deserialize.return_value = mock_experiment
        mock_file.return_value.read.return_value = json.dumps(sample_model_expt_json)
        
        # Create sample
        sample = sample_from_json(sample_model_expt_json)
        
        # Load experiment
        experiment = expt_from_json_file("test.json")
        
        # Verify both operations succeeded
        assert sample is not None
        assert experiment is not None
        assert mock_sld.called
        assert mock_deserialize.called
