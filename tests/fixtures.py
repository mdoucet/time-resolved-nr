"""
Test fixtures for model_utils tests
"""
import pytest


@pytest.fixture
def sample_model_expt_json():
    """Sample experiment JSON data structure"""
    return {
        "sample": {
            "layers": [
                {
                    "type": "layer",
                    "name": "substrate",
                    "thickness": {
                        "value": 100.0,
                        "fixed": False,
                        "name": "substrate_thickness",
                        "bounds": {"limits": [50.0, 200.0]}
                    },
                    "interface": {
                        "value": 5.0,
                        "fixed": False,
                        "name": "substrate_interface",
                        "bounds": {"limits": [1.0, 10.0]}
                    },
                    "material": {
                        "rho": {
                            "value": 4.5e-6,
                            "fixed": False,
                            "name": "substrate_rho",
                            "bounds": {"limits": [3.0e-6, 6.0e-6]}
                        },
                        "irho": {
                            "value": 1.0e-8,
                            "fixed": False,
                            "name": "substrate_irho",
                            "bounds": {"limits": [0.0, 1.0e-7]}
                        }
                    },
                    "magnetism": {}
                },
                {
                    "type": "layer",
                    "name": "film",
                    "thickness": {
                        "value": 50.0,
                        "fixed": True,
                        "name": "film_thickness",
                        "bounds": {"limits": [20.0, 100.0]}
                    },
                    "interface": {
                        "value": 3.0,
                        "fixed": True,
                        "name": "film_interface",
                        "bounds": {"limits": [1.0, 8.0]}
                    },
                    "material": {
                        "rho": {
                            "value": 2.5e-6,
                            "fixed": False,
                            "name": "film_rho",
                            "bounds": {"limits": [1.0e-6, 4.0e-6]}
                        },
                        "irho": {
                            "value": 5.0e-9,
                            "fixed": True,
                            "name": "film_irho",
                            "bounds": {"limits": [0.0, 1.0e-8]}
                        }
                    },
                    "magnetism": {}
                }
            ]
        }
    }


@pytest.fixture
def sample_model_err_json():
    """Sample error JSON data structure"""
    return {
        "substrate_rho": {"std": 0.1e-6},
        "substrate_irho": {"std": 0.5e-9},
        "substrate_thickness": {"std": 5.0},
        "substrate_interface": {"std": 0.5},
        "film_rho": {"std": 0.2e-6},
        "film_irho": {"std": 0.2e-9},
        "film_thickness": {"std": 2.0},
        "film_interface": {"std": 0.3}
    }


@pytest.fixture
def minimal_model_expt_json():
    """Minimal experiment JSON with one layer"""
    return {
        "sample": {
            "layers": [
                {
                    "type": "layer",
                    "name": "single_layer",
                    "thickness": {
                        "value": 10.0,
                        "fixed": True,
                        "name": "single_thickness",
                        "bounds": {"limits": [5.0, 20.0]}
                    },
                    "interface": {
                        "value": 2.0,
                        "fixed": True,
                        "name": "single_interface",
                        "bounds": {"limits": [1.0, 5.0]}
                    },
                    "material": {
                        "rho": {
                            "value": 1.0e-6,
                            "fixed": True,
                            "name": "single_rho",
                            "bounds": {"limits": [0.5e-6, 2.0e-6]}
                        },
                        "irho": {
                            "value": 0.0,
                            "fixed": True,
                            "name": "single_irho",
                            "bounds": {"limits": [0.0, 1.0e-8]}
                        }
                    },
                    "magnetism": {}
                }
            ]
        }
    }


@pytest.fixture
def mock_serialized_experiment():
    """Mock serialized experiment structure for testing deserialization"""
    return {
        "type": "Experiment", 
        "sample": {
            "type": "Stack",
            "layers": []
        },
        "probe": {
            "type": "QProbe"
        }
    }
