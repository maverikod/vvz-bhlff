"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for level G particle inversion models - basic tests.

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
    Test particle parameter inversion - basic tests.

    Physical Meaning:
        Tests the inversion of fundamental model parameters from
        observable particle properties.
    """

    def test_inversion_initialization(self):
        """Test inversion initialization."""
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

        loss_weights = {
            "tail": 1.0,
            "jr": 1.0,
            "Achi": 0.5,
            "peaks": 0.5,
            "mobility": 0.5,
            "Meff": 1.0,
        }

        optimization_params = {
            "max_iterations": 100,
            "tolerance": 1e-6,
            "learning_rate": 0.01,
        }

        inversion = ParticleInversion(
            observables, priors, loss_weights, optimization_params
        )

        assert inversion.observables == observables
        assert inversion.priors == priors
        assert inversion.loss_weights == loss_weights
        assert inversion.optimization_params == optimization_params

    def test_parameter_initialization(self):
        """Test parameter initialization."""
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
        initial_params = inversion._initialize_parameters()

        assert "beta" in initial_params
        assert "layers_count" in initial_params
        assert "eta" in initial_params
        assert "gamma" in initial_params
        assert "tau" in initial_params
        assert "q" in initial_params

        # Check that parameters are within prior ranges
        assert 0.6 <= initial_params["beta"] <= 1.4
        assert 1 <= initial_params["layers_count"] <= 3
        assert 0.1 <= initial_params["eta"] <= 0.3
        assert 0.1 <= initial_params["gamma"] <= 0.8
        assert 0.5 <= initial_params["tau"] <= 2.0
        assert -2 <= initial_params["q"] <= 2

    def test_loss_function_computation(self):
        """Test loss function computation."""
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

        # Test with some parameters
        params = {
            "beta": 1.0,
            "layers_count": 2,
            "eta": 0.2,
            "gamma": 0.5,
            "tau": 1.0,
            "q": 1,
        }

        loss = inversion._compute_loss(params)

        assert loss >= 0
        assert np.isfinite(loss)

    def test_model_predictions_computation(self):
        """Test model predictions computation."""
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

        predictions = inversion._compute_model_predictions(params)

        assert "tail" in predictions
        assert "jr" in predictions
        assert "Achi" in predictions
        assert "peaks" in predictions
        assert "mobility" in predictions
        assert "Meff" in predictions

        # Check that predictions are finite
        for key, value in predictions.items():
            assert np.isfinite(value)

    def test_distance_metric_computation(self):
        """Test distance metric computation."""
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

        # Test distance metrics
        distance1 = inversion._compute_distance_metric(1.0, 1.0, "tail")
        distance2 = inversion._compute_distance_metric(1.0, 0.5, "tail")
        distance3 = inversion._compute_distance_metric(2, 2, "peaks")
        distance4 = inversion._compute_distance_metric(2, 3, "peaks")

        assert distance1 == 0.0  # Perfect match
        assert distance2 > 0.0  # Mismatch
        assert distance3 == 0.0  # Perfect match for integer metric
        assert distance4 > 0.0  # Mismatch for integer metric
