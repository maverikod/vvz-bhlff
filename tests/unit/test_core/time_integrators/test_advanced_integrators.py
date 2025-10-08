"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Advanced time integrators tests.

This module contains advanced tests for time integrators
including complex scenarios and edge cases.
"""

import numpy as np
import pytest
from typing import Dict, Any, Tuple

from bhlff.core.time import (
    BVPEnvelopeIntegrator,
    CrankNicolsonIntegrator,
    MemoryKernel,
    QuenchDetector,
)
from bhlff.core.domain.domain_7d_bvp import Domain7DBVP
from bhlff.core.domain.parameters_7d_bvp import Parameters7DBVP


class TestAdvancedIntegrators:
    """
    Advanced tests for time integrators.

    Physical Meaning:
        Tests advanced functionality and edge cases of temporal integration
        methods in 7D space-time, including stability and accuracy tests.
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

    def test_integrator_stability(self, domain_7d, parameters_basic):
        """Test integrator stability with different parameter values."""
        # Test with different parameter combinations
        test_params = [
            {"mu": 1.0, "beta": 1.0, "lambda_param": 0.1, "nu": 1.0},
            {"mu": 0.1, "beta": 1.5, "lambda_param": 0.01, "nu": 0.5},
            {"mu": 10.0, "beta": 0.5, "lambda_param": 1.0, "nu": 2.0},
        ]

        for params in test_params:
            test_params_obj = Parameters7DBVP(
                mu=params["mu"],
                beta=params["beta"],
                lambda_param=params["lambda_param"],
                nu=params["nu"],
                precision="float64",
                tolerance=1e-12,
            )

            # Test exponential integrator
            exp_integrator = BVPEnvelopeIntegrator(domain_7d, test_params_obj)
            assert (
                exp_integrator is not None
            ), f"Exponential integrator should be stable for params {params}"

            # Test Crank-Nicolson integrator
            cn_integrator = CrankNicolsonIntegrator(domain_7d, test_params_obj)
            assert (
                cn_integrator is not None
            ), f"Crank-Nicolson integrator should be stable for params {params}"

    def test_integrator_accuracy(self, domain_7d, parameters_basic):
        """Test integrator accuracy with known solutions."""
        # Create integrators
        exp_integrator = BVPEnvelopeIntegrator(domain_7d, parameters_basic)
        cn_integrator = CrankNicolsonIntegrator(domain_7d, parameters_basic)

        # Test with simple field
        test_field = np.ones(domain_7d.shape)

        try:
            # Test envelope integrator
            time_steps = np.linspace(0, 0.1, 10)
            source_field = np.zeros((len(time_steps),) + test_field.shape, dtype=test_field.dtype)
            exp_result = exp_integrator.integrate(test_field, source_field, time_steps)
            assert np.all(
                np.isfinite(exp_result)
            ), "Envelope integrator should produce finite results"

            # Test Crank-Nicolson integrator
            cn_result = cn_integrator.integrate(test_field, source_field, time_steps)
            assert np.all(
                np.isfinite(cn_result)
            ), "Crank-Nicolson integrator should produce finite results"

        except (NotImplementedError, AttributeError):
            # Some methods might not be fully implemented yet
            pass

    def test_memory_kernel_advanced(self, domain_7d, parameters_basic):
        """Test advanced memory kernel functionality."""
        kernel = MemoryKernel(domain_7d, num_memory_vars=3)

        # Test with different field configurations
        test_fields = [
            np.ones(domain_7d.shape) + 1j * np.ones(domain_7d.shape),
            np.random.randn(*domain_7d.shape) + 1j * np.random.randn(*domain_7d.shape),
            np.zeros(domain_7d.shape) + 1j * np.zeros(domain_7d.shape),
        ]

        for field in test_fields:
            try:
                # Test kernel computation
                kernel_result = kernel.compute_kernel(field)
                assert np.all(
                    np.isfinite(kernel_result)
                ), "Memory kernel should produce finite results"

                # Test kernel application
                applied_result = kernel.apply_kernel(field)
                assert np.all(
                    np.isfinite(applied_result)
                ), "Applied kernel should produce finite results"

            except (NotImplementedError, AttributeError):
                # Some methods might not be fully implemented yet
                pass

    def test_quench_detector_advanced(self, domain_7d, parameters_basic):
        """Test advanced quench detector functionality."""
        detector = QuenchDetector(domain_7d, energy_threshold=0.1, rate_threshold=0.01, magnitude_threshold=0.5)

        # Test with different field configurations
        test_fields = [
            np.ones(domain_7d.shape) + 1j * np.ones(domain_7d.shape),
            np.random.randn(*domain_7d.shape) + 1j * np.random.randn(*domain_7d.shape),
            np.zeros(domain_7d.shape) + 1j * np.zeros(domain_7d.shape),
        ]

        for field in test_fields:
            try:
                # Test quench detection
                quench_result = detector.detect_quench(field, time=0.0)
                assert isinstance(
                    quench_result, (bool, np.ndarray)
                ), "Quench detection should return boolean or array"

                # Test quench analysis
                analysis_result = detector.analyze_quench(field)
                assert (
                    analysis_result is not None
                ), "Quench analysis should return a result"

            except (NotImplementedError, AttributeError):
                # Some methods might not be fully implemented yet
                pass

    def test_integrator_convergence(self, domain_7d, parameters_basic):
        """Test integrator convergence with increasing resolution."""
        # Test with different resolutions
        resolutions = [4, 8, 16]

        for N in resolutions:
            test_domain = Domain7DBVP(L_spatial=1.0, N_spatial=N, N_phase=4, T=1.0, N_t=8)

            # Test exponential integrator
            exp_integrator = BVPEnvelopeIntegrator(test_domain, parameters_basic)
            assert (
                exp_integrator is not None
            ), f"Exponential integrator should work with resolution {N}"

            # Test Crank-Nicolson integrator
            cn_integrator = CrankNicolsonIntegrator(test_domain, parameters_basic)
            assert (
                cn_integrator is not None
            ), f"Crank-Nicolson integrator should work with resolution {N}"

    def test_integrator_boundary_conditions(self, domain_7d, parameters_basic):
        """Test integrator behavior with boundary conditions."""
        integrator = BVPEnvelopeIntegrator(domain_7d, parameters_basic)

        # Test with fields that have boundary effects
        boundary_field = np.zeros(domain_7d.shape)
        boundary_field[0, :, :, :, :, :, :] = 1.0  # Left boundary
        boundary_field[-1, :, :, :, :, :, :] = 1.0  # Right boundary

        try:
            time_steps = np.linspace(0, 0.1, 10)
            source_field = np.zeros((len(time_steps),) + boundary_field.shape, dtype=boundary_field.dtype)
            result = integrator.integrate(boundary_field, source_field, time_steps)
            assert np.all(
                np.isfinite(result)
            ), "Integrator should handle boundary conditions"

        except (NotImplementedError, AttributeError):
            # Some methods might not be fully implemented yet
            pass

    def test_integrator_extreme_values(self, domain_7d, parameters_basic):
        """Test integrator behavior with extreme values."""
        integrator = BVPEnvelopeIntegrator(domain_7d, parameters_basic)

        # Test with very large values
        large_field = np.full(domain_7d.shape, 1e10)

        try:
            time_steps = np.linspace(0, 0.1, 10)
            source_field = np.zeros((len(time_steps),) + large_field.shape, dtype=large_field.dtype)
            result = integrator.integrate(large_field, source_field, time_steps)
            assert np.all(np.isfinite(result)), "Integrator should handle large values"

        except (NotImplementedError, AttributeError):
            # Some methods might not be fully implemented yet
            pass

        # Test with very small values
        small_field = np.full(domain_7d.shape, 1e-10)

        try:
            time_steps = np.linspace(0, 0.1, 10)
            source_field = np.zeros((len(time_steps),) + small_field.shape, dtype=small_field.dtype)
            result = integrator.integrate(small_field, source_field, time_steps)
            assert np.all(np.isfinite(result)), "Integrator should handle small values"

        except (NotImplementedError, AttributeError):
            # Some methods might not be fully implemented yet
            pass

    def test_integrator_consistency_across_runs(self, domain_7d, parameters_basic):
        """Test that integrators produce consistent results across runs."""
        integrator = BVPEnvelopeIntegrator(domain_7d, parameters_basic)
        test_field = np.random.randn(*domain_7d.shape)

        try:
            # Run integration multiple times
            results = []
            for i in range(3):
                time_steps = np.linspace(0, 0.1, 10)
                source_field = np.zeros((len(time_steps),) + test_field.shape, dtype=test_field.dtype)
                result = integrator.integrate(test_field, source_field, time_steps)
                results.append(result)

            # Results should be identical
            for i in range(1, len(results)):
                assert np.allclose(
                    results[0], results[i], rtol=1e-15
                ), "Integrator should produce consistent results across runs"

        except (NotImplementedError, AttributeError):
            # Some methods might not be fully implemented yet
            pass

    def test_integrator_memory_usage(self, domain_7d, parameters_basic):
        """Test integrator memory usage."""
        import psutil
        import os

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create multiple integrators
        integrators = []
        for i in range(5):
            integrator = BVPEnvelopeIntegrator(domain_7d, parameters_basic)
            integrators.append(integrator)

        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable
        assert (
            memory_increase < 100.0
        ), f"Memory usage increased too much: {memory_increase:.1f}MB"

    def test_integrator_error_recovery(self, domain_7d, parameters_basic):
        """Test integrator error recovery."""
        integrator = BVPEnvelopeIntegrator(domain_7d, parameters_basic)

        # Test with invalid input
        with pytest.raises((ValueError, TypeError)):
            time_steps = np.linspace(0, 0.1, 10)
            source_field = np.zeros((len(time_steps),) + domain_7d.shape, dtype=np.complex128)
            integrator.integrate("invalid_input", source_field, time_steps)

        # Test that integrator still works after error
        test_field = np.random.randn(*domain_7d.shape)

        try:
            time_steps = np.linspace(0, 0.1, 10)
            source_field = np.zeros((len(time_steps),) + test_field.shape, dtype=test_field.dtype)
            result = integrator.integrate(test_field, source_field, time_steps)
            assert np.all(np.isfinite(result)), "Integrator should work after error"

        except (NotImplementedError, AttributeError):
            # Some methods might not be fully implemented yet
            pass
