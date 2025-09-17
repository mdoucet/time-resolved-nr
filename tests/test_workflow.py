"""
Tests for workflow functions.
"""

import unittest
import tempfile
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, mock_open

from timeref.workflow import load_data_from_directory, package_json_data


class TestLoadDataFromDirectory(unittest.TestCase):
    """Test the load_data_from_directory function."""

    def setUp(self):
        """Set up test data."""
        # Sample data that would be in a txt file
        self.sample_data = np.array([
            [0.1, 0.2, 0.3],  # q values
            [1.0, 0.8, 0.6],  # reflectivity values
            [0.1, 0.1, 0.1],  # error values
        ])

    def test_load_data_from_directory_nonexistent_directory(self):
        """Test load_data_from_directory with non-existent directory."""
        nonexistent_path = Path("/nonexistent/directory")
        
        with self.assertRaises(ValueError) as context:
            load_data_from_directory(nonexistent_path)
        
        self.assertIn("Directory", str(context.exception))
        self.assertIn("does not exist", str(context.exception))

    def test_load_data_from_directory_file_instead_of_directory(self):
        """Test load_data_from_directory when given a file instead of directory."""
        with tempfile.NamedTemporaryFile() as temp_file:
            file_path = Path(temp_file.name)
            
            with self.assertRaises(ValueError) as context:
                load_data_from_directory(file_path)
            
            self.assertIn("Directory", str(context.exception))
            self.assertIn("does not exist", str(context.exception))

    @patch('timeref.workflow.package_json_data')
    def test_load_data_from_directory_single_run_success(self, mock_package):
        """Test successful loading with single run number."""
        mock_package.return_value = ([100, 200, 300], [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]])
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock data files for run 5
            (temp_path / "r5_t100.txt").touch()
            (temp_path / "r5_t200.txt").touch()
            (temp_path / "r5_t300.txt").touch()
            
            result = load_data_from_directory(temp_path)
            
            # Verify package_json_data was called
            mock_package.assert_called_once()
            call_args = mock_package.call_args[0]
            
            # Check that the correct files were passed
            self.assertEqual(len(call_args[0]), 3)  # 3 data files
            self.assertEqual(call_args[1], temp_path)  # data_location
            self.assertTrue(str(call_args[2]).endswith("r5-time-resolved.json"))  # output file
            
            # Check return value
            self.assertEqual(result, [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]])

    @patch('timeref.workflow.logging')
    def test_load_data_from_directory_multiple_runs_error(self, mock_logging):
        """Test error handling with multiple run numbers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create files for different run numbers
            (temp_path / "r3_t100.txt").touch()
            (temp_path / "r5_t200.txt").touch()  # Different run number
            
            result = load_data_from_directory(temp_path)
            
            # Should return None and log error
            self.assertIsNone(result)
            mock_logging.error.assert_called()
            error_call = mock_logging.error.call_args[0][0]
            self.assertIn("Multiple run numbers", error_call)

    @patch('timeref.workflow.logging')
    def test_load_data_from_directory_invalid_filename_format(self, mock_logging):
        """Test handling of files with invalid filename format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create files with invalid format
            (temp_path / "invalid_format.txt").touch()
            (temp_path / "r_t100.txt").touch()  # Missing run number
            (temp_path / "rabc_t100.txt").touch()  # Non-numeric run number
            
            result = load_data_from_directory(temp_path)
            
            # Should return None due to no valid run numbers
            self.assertIsNone(result)
            
            # Should have logged errors for invalid formats
            self.assertGreater(mock_logging.error.call_count, 0)

    def test_load_data_from_directory_no_matching_files(self):
        """Test directory with no matching files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create files that don't match the pattern
            (temp_path / "data.txt").touch()
            (temp_path / "results.json").touch()
            
            result = load_data_from_directory(temp_path)
            
            # Should return None as no valid run numbers found
            self.assertIsNone(result)

    @patch('timeref.workflow.package_json_data')
    def test_load_data_from_directory_files_sorted(self, mock_package):
        """Test that files are sorted before processing."""
        mock_package.return_value = ([100, 200], [[[1, 2]], [[3, 4]]])
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create files in non-sorted order
            (temp_path / "r1_t200.txt").touch()
            (temp_path / "r1_t100.txt").touch()
            
            load_data_from_directory(temp_path)
            
            # Verify files were sorted
            call_args = mock_package.call_args[0]
            file_names = [f.name for f in call_args[0]]
            self.assertEqual(file_names, ["r1_t100.txt", "r1_t200.txt"])


class TestPackageJsonData(unittest.TestCase):
    """Test the package_json_data function."""

    def setUp(self):
        """Set up test data."""
        self.sample_data_1 = np.array([
            [0.1, 0.2, 0.3],  # q values
            [1.0, 0.8, 0.6],  # reflectivity values  
            [0.1, 0.1, 0.1],  # error values
        ])
        
        self.sample_data_2 = np.array([
            [0.1, 0.2, 0.3],  # q values
            [0.9, 0.7, 0.5],  # reflectivity values
            [0.1, 0.1, 0.1],  # error values
        ])

    @patch('numpy.loadtxt')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_package_json_data_success(self, mock_json_dump, mock_file_open, mock_loadtxt):
        """Test successful packaging of data files."""
        # Mock numpy.loadtxt to return our sample data
        mock_loadtxt.side_effect = [self.sample_data_1, self.sample_data_2]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "output.json"
            
            # Create mock file objects
            file1 = Path("r3_t100.txt")
            file2 = Path("r3_t200.txt")
            data_files = [file1, file2]
            
            times, data = package_json_data(data_files, temp_path, output_path)
            
            # Verify times were extracted correctly
            self.assertEqual(times, [100, 200])
            
            # Verify data was transposed and converted to list
            expected_data = [
                self.sample_data_1.T.tolist(),
                self.sample_data_2.T.tolist()
            ]
            self.assertEqual(data, expected_data)
            
            # Verify numpy.loadtxt was called correctly
            self.assertEqual(mock_loadtxt.call_count, 2)
            
            # Verify JSON was written
            mock_json_dump.assert_called_once()
            json_data = mock_json_dump.call_args[0][0]
            self.assertEqual(json_data["times"], [100, 200])
            self.assertEqual(json_data["data"], expected_data)

    @patch('numpy.loadtxt')
    def test_package_json_data_single_file(self, mock_loadtxt):
        """Test packaging with single data file."""
        mock_loadtxt.return_value = self.sample_data_1
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "output.json"
            
            file1 = Path("r1_t150.txt")
            data_files = [file1]
            
            times, data = package_json_data(data_files, temp_path, output_path)
            
            self.assertEqual(times, [150])
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0], self.sample_data_1.T.tolist())

    @patch('numpy.loadtxt')
    def test_package_json_data_complex_timestamps(self, mock_loadtxt):
        """Test extraction of various timestamp formats."""
        mock_loadtxt.side_effect = [self.sample_data_1, self.sample_data_2]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "output.json"
            
            # Test various timestamp formats
            file1 = Path("r5_t0.txt")    # timestamp 0
            file2 = Path("r5_t1000.txt") # timestamp 1000
            data_files = [file1, file2]
            
            times, data = package_json_data(data_files, temp_path, output_path)
            
            self.assertEqual(times, [0, 1000])
            self.assertEqual(len(data), 2)

    def test_package_json_data_file_creation(self):
        """Test that JSON file is actually created and readable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "test_output.json"
            
            # Create actual test data files
            data_file1 = temp_path / "r1_t100.txt"
            data_file2 = temp_path / "r1_t200.txt"
            
            # Write sample data to files
            np.savetxt(data_file1, self.sample_data_1)
            np.savetxt(data_file2, self.sample_data_2)
            
            data_files = [data_file1, data_file2]
            
            times, data = package_json_data(data_files, temp_path, output_path)
            
            # Verify file was created
            self.assertTrue(output_path.exists())
            
            # Verify file contents
            with open(output_path, 'r') as f:
                saved_data = json.load(f)
            
            self.assertEqual(saved_data["times"], [100, 200])
            self.assertEqual(len(saved_data["data"]), 2)

    @patch('numpy.loadtxt')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_package_json_data_empty_file_list(self, mock_json_dump, mock_file_open, mock_loadtxt):
        """Test package_json_data with empty file list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "output.json"
            
            data_files = []
            
            times, data = package_json_data(data_files, temp_path, output_path)
            
            self.assertEqual(times, [])
            self.assertEqual(data, [])
            
            # Verify JSON was still written (with empty data)
            mock_json_dump.assert_called_once()
            json_data = mock_json_dump.call_args[0][0]
            self.assertEqual(json_data["times"], [])
            self.assertEqual(json_data["data"], [])


if __name__ == '__main__':
    unittest.main()
