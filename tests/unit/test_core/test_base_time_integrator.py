"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for BaseTimeIntegrator.

This module contains unit tests for the BaseTimeIntegrator class
in the 7D BVP framework, focusing on parameter validation and
abstract base class functionality.

Physical Meaning:
    Tests the abstract base class functionality and parameter validation
    for temporal integrators in the 7D BVP framework.

Mathematical Foundation:
    Tests validate the integration of the dynamic equation:
    ∂a/∂t + ν(-Δ)^β a + λa = s(x,φ,t)
    with various source configurations and parameter combinations.
"""

import numpy as np
import pytest
from typing import Dict, Any, Tuple

from bhlff.core.time import BaseTimeIntegrator, BVPEnvelopeIntegrator
from bhlff.core.domain.domain_7d_bvp import Domain7DBVP
from bhlff.core.domain.parameters_7d_bvp import Parameters7DBVP


class TestBaseTimeIntegrator:
    """
    Unit tests for BaseTimeIntegrator.

    Physical Meaning:
        Tests the abstract base class functionality and parameter validation
        for temporal integrators in the 7D BVP framework.
    """

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain7DBVP(L_spatial=1.0, N_spatial=8, N_phase=4, T=1.0, N_t=8)

    @pytest.fixture
    def parameters_basic(self):
        """Basic parameters for testing."""
        return Parameters7DBVP(
            mu=1.0,
            beta=1.0,
            lambda_param=0.1,
            nu=1.0,
            precision="float64",
            tolerance=1e-12,
        )

    def test_parameter_validation(self, domain_7d, parameters_basic):
        """
        Test parameter validation.

        Physical Meaning:
            Validates that the integrator correctly validates physical
            parameters and raises appropriate errors for invalid values.
        """
        # Test valid parameters
        integrator = BVPEnvelopeIntegrator(domain_7d, parameters_basic)
        assert integrator.is_initialized

        # Test invalid nu (negative) - this will fail in Parameters7DBVP constructor
        with pytest.raises(
            ValueError, match="nu must be positive"
        ):
            invalid_params = Parameters7DBVP(
                mu=1.0,
                beta=1.0,
                lambda_param=0.1,
                nu=-1.0,
                precision="float64",
                tolerance=1e-12,
            )

        # Test invalid beta (out of range) - this will fail in Parameters7DBVP constructor
        with pytest.raises(ValueError, match="beta must be in \\(0,2\\)"):
            invalid_params = Parameters7DBVP(
                mu=1.0,
                beta=2.5,
                lambda_param=0.1,
                nu=1.0,
                precision="float64",
                tolerance=1e-12,
            )

        # Test invalid lambda (negative) - this will fail in Parameters7DBVP constructor
        with pytest.raises(
            ValueError, match="lambda_param must be non-negative"
        ):
            invalid_params = Parameters7DBVP(
                mu=1.0,
                beta=1.0,
                lambda_param=-0.1,
                nu=1.0,
                precision="float64",
                tolerance=1e-12,
            )

    def test_abstract_methods(self, domain_7d, parameters_basic):
        """
        Test that abstract methods raise NotImplementedError.

        Physical Meaning:
            Validates that the abstract base class properly enforces
            implementation of required methods in subclasses.
        """
        # Create instance of abstract class (should raise error)
        with pytest.raises(TypeError):
            BaseTimeIntegrator(domain_7d, parameters_basic)

    def test_domain_validation(self, parameters_basic):
        """
        Test domain validation.

        Physical Meaning:
            Validates that the integrator correctly validates the
            computational domain and raises appropriate errors.
        """
        # Test invalid domain (negative L_spatial) - this will fail in Domain7DBVP constructor
        with pytest.raises(ValueError, match="L_spatial must be positive"):
            invalid_domain = Domain7DBVP(L_spatial=-1.0, N_spatial=8, N_phase=4, T=1.0, N_t=8)

        # Test invalid domain (N_spatial zero) - this will fail in Domain7DBVP constructor
        with pytest.raises(ValueError, match="N_spatial must be positive"):
            invalid_domain = Domain7DBVP(L_spatial=1.0, N_spatial=0, N_phase=4, T=1.0, N_t=8)

    def test_initialization_state(self, domain_7d, parameters_basic):
        """
        Test initialization state tracking.

        Physical Meaning:
            Validates that the integrator correctly tracks its
            initialization state and prevents operations before
            proper initialization.
        """
        integrator = BVPEnvelopeIntegrator(domain_7d, parameters_basic)
        assert integrator.is_initialized

        # Test that operations work after initialization
        current_field = np.random.random(domain_7d.shape) + 1j * np.random.random(
            domain_7d.shape
        )
        source_field = np.random.random(domain_7d.shape) + 1j * np.random.random(
            domain_7d.shape
        )
        result = integrator.step(current_field, source_field, 0.01)
        assert result is not None
        assert result.shape == domain_7d.shape
