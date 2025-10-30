"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for FractionalLaplacian with Parameters7DBVP support.

This module tests the enhanced FractionalLaplacian initialization
that supports both float parameters and Parameters7DBVP objects.
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.operators.fractional_laplacian import FractionalLaplacian
from bhlff.core.domain.domain_7d_bvp import Domain7DBVP
from bhlff.core.domain.parameters_7d_bvp import Parameters7DBVP


class TestFractionalLaplacianParameters:
    """
    Tests for FractionalLaplacian with Parameters7DBVP support.

    Physical Meaning:
        Tests the enhanced FractionalLaplacian initialization that supports
        both direct float parameters and Parameters7DBVP objects for
        improved integration with the 7D BVP framework.
    """

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain7DBVP(L_spatial=1.0, N_spatial=8, N_phase=4, T=1.0, N_t=8)

    @pytest.fixture
    def parameters_7d(self):
        """Create Parameters7DBVP for testing."""
        return Parameters7DBVP(
            mu=1.0,
            beta=1.5,
            lambda_param=0.1,
            precision="float64",
            tolerance=1e-12,
        )

    def test_parameters_object_initialization(self, domain_7d, parameters_7d):
        """
        Test initialization with Parameters7DBVP object.

        Physical Meaning:
            Tests that FractionalLaplacian can be initialized with
            a Parameters7DBVP object, extracting beta and lambda_param
            automatically.
        """
        laplacian = FractionalLaplacian(domain_7d, parameters_7d)

        # Check that parameters were extracted correctly
        assert laplacian.beta == parameters_7d.beta
        assert laplacian.lambda_param == parameters_7d.lambda_param
        assert laplacian.domain == domain_7d

    def test_float_initialization_backward_compatibility(self, domain_7d):
        """
        Test backward compatibility with float initialization.

        Physical Meaning:
            Tests that the original float-based initialization
            still works for backward compatibility.
        """
        laplacian = FractionalLaplacian(domain_7d, beta=1.5, lambda_param=0.1)

        # Check that parameters were set correctly
        assert laplacian.beta == 1.5
        assert laplacian.lambda_param == 0.1
        assert laplacian.domain == domain_7d

    def test_parameters_object_with_custom_lambda(self, domain_7d, parameters_7d):
        """
        Test Parameters object with custom lambda_param override.

        Physical Meaning:
            Tests that lambda_param can be overridden when using
            Parameters object initialization.
        """
        custom_lambda = 0.5
        laplacian = FractionalLaplacian(
            domain_7d, parameters_7d, lambda_param=custom_lambda
        )

        # Check that custom lambda was used
        assert laplacian.beta == parameters_7d.beta
        assert laplacian.lambda_param == custom_lambda

    def test_parameters_object_without_lambda_attribute(self, domain_7d):
        """
        Test Parameters object without lambda_param attribute.

        Physical Meaning:
            Tests graceful handling when Parameters object doesn't
            have lambda_param attribute.
        """

        # Create a mock parameters object without lambda_param
        class MockParameters:
            def __init__(self):
                self.beta = 1.5

        mock_params = MockParameters()
        laplacian = FractionalLaplacian(domain_7d, mock_params, lambda_param=0.2)

        # Check that default lambda_param was used
        assert laplacian.beta == 1.5
        assert laplacian.lambda_param == 0.2

    def test_apply_method_with_parameters_object(self, domain_7d, parameters_7d):
        """
        Test apply method works with Parameters object initialization.

        Physical Meaning:
            Tests that the apply method functions correctly
            when initialized with Parameters object.
        """
        laplacian = FractionalLaplacian(domain_7d, parameters_7d)

        # Create test field
        field = np.random.random(domain_7d.shape)

        # Apply fractional Laplacian
        result = laplacian.apply(field)

        # Check that result is valid
        assert isinstance(result, np.ndarray)
        assert result.shape == field.shape
        assert np.isfinite(result).all()

    def test_spectral_coefficients_with_parameters_object(
        self, domain_7d, parameters_7d
    ):
        """
        Test spectral coefficients with Parameters object initialization.

        Physical Meaning:
            Tests that spectral coefficients are computed correctly
            when initialized with Parameters object.
        """
        laplacian = FractionalLaplacian(domain_7d, parameters_7d)

        # Get spectral coefficients
        coeffs = laplacian.get_spectral_coefficients()

        # Check that coefficients are valid
        assert isinstance(coeffs, np.ndarray)
        assert coeffs.shape == domain_7d.shape
        assert np.all(coeffs >= 0)  # Should be non-negative
        assert np.isfinite(coeffs).all()

    def test_fractional_order_retrieval(self, domain_7d, parameters_7d):
        """
        Test fractional order retrieval with Parameters object.

        Physical Meaning:
            Tests that the fractional order can be retrieved correctly
            when initialized with Parameters object.
        """
        laplacian = FractionalLaplacian(domain_7d, parameters_7d)

        # Get fractional order
        beta = laplacian.get_fractional_order()

        # Check that it matches the parameters
        assert beta == parameters_7d.beta

    def test_validation_with_parameters_object(self, domain_7d):
        """
        Test parameter validation with Parameters object.

        Physical Meaning:
            Tests that parameter validation works correctly
            when using Parameters object initialization.
        """

        # Test with invalid beta using mock object (since Parameters7DBVP validates beta)
        class MockInvalidParameters:
            def __init__(self):
                self.beta = 2.5  # Invalid: beta > 2
                self.lambda_param = 0.1

        invalid_params = MockInvalidParameters()

        with pytest.raises(
            ValueError, match="Fractional order beta must be in \\(0,2\\)"
        ):
            FractionalLaplacian(domain_7d, invalid_params)

    def test_validation_with_float_parameters(self, domain_7d):
        """
        Test parameter validation with float parameters.

        Physical Meaning:
            Tests that parameter validation works correctly
            when using float parameter initialization.
        """
        # Test with invalid beta
        with pytest.raises(
            ValueError, match="Fractional order beta must be in \\(0,2\\)"
        ):
            FractionalLaplacian(domain_7d, beta=2.5, lambda_param=0.1)

        # Test with beta = 0
        with pytest.raises(
            ValueError, match="Fractional order beta must be in \\(0,2\\)"
        ):
            FractionalLaplacian(domain_7d, beta=0.0, lambda_param=0.1)

    def test_consistency_between_initialization_methods(self, domain_7d, parameters_7d):
        """
        Test consistency between Parameters object and float initialization.

        Physical Meaning:
            Tests that both initialization methods produce
            equivalent results for the same parameters.
        """
        # Initialize with Parameters object
        laplacian_params = FractionalLaplacian(domain_7d, parameters_7d)

        # Initialize with float parameters
        laplacian_float = FractionalLaplacian(
            domain_7d, parameters_7d.beta, parameters_7d.lambda_param
        )

        # Check that both have same parameters
        assert laplacian_params.beta == laplacian_float.beta
        assert laplacian_params.lambda_param == laplacian_float.lambda_param

        # Check that spectral coefficients are identical
        coeffs_params = laplacian_params.get_spectral_coefficients()
        coeffs_float = laplacian_float.get_spectral_coefficients()

        np.testing.assert_array_almost_equal(coeffs_params, coeffs_float, decimal=12)

        # Check that apply method produces identical results
        field = np.random.random(domain_7d.shape)
        result_params = laplacian_params.apply(field)
        result_float = laplacian_float.apply(field)

        np.testing.assert_array_almost_equal(result_params, result_float, decimal=12)

    def test_fft_solver_integration(self, domain_7d, parameters_7d):
        """
        Test integration with FFTSolver7DBasic.

        Physical Meaning:
            Tests that the enhanced FractionalLaplacian works
            correctly with FFTSolver7DBasic when using Parameters7DBVP.
        """
        from bhlff.core.fft.fft_solver_7d_basic import FFTSolver7DBasic

        # Create solver with Parameters7DBVP
        solver = FFTSolver7DBasic(domain_7d, parameters_7d)

        # Check that fractional_laplacian was created correctly
        assert hasattr(solver, "fractional_laplacian")
        assert solver.fractional_laplacian.beta == parameters_7d.beta
        assert solver.fractional_laplacian.lambda_param == parameters_7d.lambda_param

        # Test that solver works
        source = np.random.random(domain_7d.shape)
        solution = solver.solve(source)

        assert isinstance(solution, np.ndarray)
        assert solution.shape == domain_7d.shape
        assert np.isfinite(solution).all()
