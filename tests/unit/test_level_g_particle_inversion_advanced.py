"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for level G particle inversion models - advanced tests.

This module tests the particle parameter inversion for 7D phase field theory,
including parameter inversion from observable properties.

Physical Meaning:
    Tests the inversion of model parameters from observable particle properties.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from bhlff.models.level_g.validation import ParticleInversion


class TestParticleInversion:
    """
    Test particle parameter inversion - advanced tests.

    Physical Meaning:
        Tests the inversion of fundamental model parameters from
        observable particle properties.
    """

    def test_regularization_computation(self):
        """Test regularization computation."""
        observables = {
            "tail": 1.0,
            "jr": 0.5,
            "Achi": 0.3,
            "peaks": 2,
            "mobility": 0.8,
            "Meff": 1.2,
        }

        priors = {
            "beta": [0.6, 1.4],
            "layers_count": [1, 3],
            "eta": [0.1, 0.3],
            "gamma": [0.1, 0.8],
            "tau": [0.5, 2.0],
            "q": [-2, 2],
        }

        inversion = ParticleInversion(observables, priors)

        # Test with parameters within bounds
        params_within_bounds = {
            "beta": 1.0,
            "layers_count": 2,
            "eta": 0.2,
            "gamma": 0.5,
            "tau": 1.0,
            "q": 1,
        }

        regularization1 = inversion._compute_regularization(params_within_bounds)
        assert regularization1 >= 0

        # Test with parameters outside bounds
        params_outside_bounds = {
            "beta": 2.0,  # Outside bounds
            "layers_count": 2,
            "eta": 0.2,
            "gamma": 0.5,
            "tau": 1.0,
            "q": 1,
        }

        regularization2 = inversion._compute_regularization(params_outside_bounds)
        assert regularization2 > regularization1  # Should be higher for out-of-bounds

    def test_gradients_computation(self):
        """Test gradients computation."""
        observables = {
            "tail": 1.0,
            "jr": 0.5,
            "Achi": 0.3,
            "peaks": 2,
            "mobility": 0.8,
            "Meff": 1.2,
        }

        priors = {
            "beta": [0.6, 1.4],
            "layers_count": [1, 3],
            "eta": [0.1, 0.3],
            "gamma": [0.1, 0.8],
            "tau": [0.5, 2.0],
            "q": [-2, 2],
        }

        inversion = ParticleInversion(observables, priors)

        params = {
            "beta": 1.0,
            "layers_count": 2,
            "eta": 0.2,
            "gamma": 0.5,
            "tau": 1.0,
            "q": 1,
        }

        gradients = inversion._compute_gradients(params)

        assert "beta" in gradients
        assert "layers_count" in gradients
        assert "eta" in gradients
        assert "gamma" in gradients
        assert "tau" in gradients
        assert "q" in gradients

        # Check that gradients are finite
        for key, value in gradients.items():
            assert np.isfinite(value)

    def test_parameter_uncertainties_computation(self):
        """Test parameter uncertainties computation."""
        observables = {
            "tail": 1.0,
            "jr": 0.5,
            "Achi": 0.3,
            "peaks": 2,
            "mobility": 0.8,
            "Meff": 1.2,
        }

        priors = {
            "beta": [0.6, 1.4],
            "layers_count": [1, 3],
            "eta": [0.1, 0.3],
            "gamma": [0.1, 0.8],
            "tau": [0.5, 2.0],
            "q": [-2, 2],
        }

        inversion = ParticleInversion(observables, priors)

        params = {
            "beta": 1.0,
            "layers_count": 2,
            "eta": 0.2,
            "gamma": 0.5,
            "tau": 1.0,
            "q": 1,
        }

        uncertainties = inversion._compute_parameter_uncertainties(params)

        assert "beta" in uncertainties
        assert "layers_count" in uncertainties
        assert "eta" in uncertainties
        assert "gamma" in uncertainties
        assert "tau" in uncertainties
        assert "q" in uncertainties

        # Check that uncertainties are positive
        for key, value in uncertainties.items():
            assert value > 0
            assert np.isfinite(value)

    def test_inversion_optimization(self):
        """Test inversion optimization."""
        observables = {
            "tail": 1.0,
            "jr": 0.5,
            "Achi": 0.3,
            "peaks": 2,
            "mobility": 0.8,
            "Meff": 1.2,
        }

        priors = {
            "beta": [0.6, 1.4],
            "layers_count": [1, 3],
            "eta": [0.1, 0.3],
            "gamma": [0.1, 0.8],
            "tau": [0.5, 2.0],
            "q": [-2, 2],
        }

        optimization_params = {
            "max_iterations": 10,  # Small number for testing
            "tolerance": 1e-6,
            "learning_rate": 0.01,
        }

        inversion = ParticleInversion(
            observables, priors, optimization_params=optimization_params
        )
        results = inversion.invert_parameters()

        assert "optimized_parameters" in results
        assert "final_loss" in results
        assert "convergence_info" in results
        assert "parameter_uncertainties" in results

        # Check that optimized parameters are within bounds
        optimized_params = results["optimized_parameters"]
        for param_name, param_value in optimized_params.items():
            if param_name in priors:
                prior_range = priors[param_name]
                if isinstance(prior_range, list) and len(prior_range) == 2:
                    min_val, max_val = prior_range
                    assert min_val <= param_value <= max_val

        # Check that final loss is finite
        assert np.isfinite(results["final_loss"])
        assert results["final_loss"] >= 0
