"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for step resonator integration in 7D BVP theory implementation.

This module tests that step resonators with frequency-dependent coefficients
are properly integrated into all relevant modules.
"""

import pytest
import numpy as np
from pathlib import Path
from bhlff.core.bvp.boundary.step_resonator import (
    FrequencyDependentResonator,
    CascadeResonatorFilter,
    create_frequency_dependent_resonator,
    create_cascade_filter,
)


class TestStepResonatorIntegration:
    """
    Test class for verifying step resonator integration.

    Physical Meaning:
        Ensures that step resonators with frequency-dependent coefficients
        are properly integrated into the 7D BVP theory implementation.
    """

    def test_frequency_dependent_resonator_creation(self):
        """
        Test creation of frequency-dependent resonator.

        Physical Meaning:
            Verifies that frequency-dependent resonators can be created
            with proper R(ω) and T(ω) coefficients.
        """
        resonator = FrequencyDependentResonator(R0=0.1, T0=0.9, omega0=1.0)

        assert resonator.R0 == 0.1
        assert resonator.T0 == 0.9
        assert resonator.omega0 == 1.0

        # Test coefficient computation
        frequencies = np.array([0.5, 1.0, 2.0])
        R, T = resonator.compute_coefficients(frequencies)

        assert len(R) == len(frequencies)
        assert len(T) == len(frequencies)
        assert np.all(R >= 0) and np.all(R <= 1)
        assert np.all(T >= 0) and np.all(T <= 1)

        print("✅ Frequency-dependent resonator creation successful")

    def test_cascade_filter_creation(self):
        """
        Test creation of cascade resonator filter.

        Physical Meaning:
            Verifies that cascade filters can be created with multiple
            resonator stages for complex energy exchange patterns.
        """
        filter_stages = 3
        cascade_filter = CascadeResonatorFilter(stages=filter_stages)

        assert cascade_filter.stages == filter_stages
        assert len(cascade_filter.resonators) == filter_stages

        # Test cascade filter application
        field = np.random.random((10, 10)) + 1j * np.random.random((10, 10))
        frequencies = np.linspace(0.1, 2.0, 10)

        # Use only valid axes for 2D field
        filtered_field = cascade_filter.apply_cascade_filter(
            field, frequencies, axes=(0, 1)
        )

        assert filtered_field.shape == field.shape
        assert not np.array_equal(filtered_field, field)  # Should be modified

        print("✅ Cascade filter creation and application successful")

    def test_resonator_integration_in_bvp_integrator(self):
        """
        Test that bvp_envelope_integrator.py uses frequency-dependent resonators.

        Physical Meaning:
            Ensures that the envelope integrator uses step resonator model
            instead of exponential factors.
        """
        file_path = Path("bhlff/core/time/bvp_envelope_integrator.py")
        assert file_path.exists(), f"File {file_path} not found"

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for resonator imports and usage
        assert (
            "FrequencyDependentResonator" in content
        ), "Missing FrequencyDependentResonator import"
        assert (
            "CascadeResonatorFilter" in content
        ), "Missing CascadeResonatorFilter import"
        assert "_resonator" in content, "Missing resonator initialization"
        assert (
            "compute_coefficients" in content
        ), "Missing frequency-dependent coefficient computation"

        print("✅ BVP envelope integrator uses frequency-dependent resonators")

    def test_resonator_integration_in_memory_kernel(self):
        """
        Test that memory_kernel.py uses frequency-dependent resonators.

        Physical Meaning:
            Ensures that memory kernels use step resonator model
            with frequency-dependent coefficients.
        """
        file_path = Path("bhlff/core/operators/memory_kernel.py")
        assert file_path.exists(), f"File {file_path} not found"

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for resonator imports and usage
        assert (
            "FrequencyDependentResonator" in content
        ), "Missing FrequencyDependentResonator import"
        assert (
            "CascadeResonatorFilter" in content
        ), "Missing CascadeResonatorFilter import"
        assert "_resonator" in content, "Missing resonator initialization"
        assert (
            "compute_coefficients" in content
        ), "Missing frequency-dependent coefficient computation"

        print("✅ Memory kernel uses frequency-dependent resonators")

    def test_resonator_integration_in_collective(self):
        """
        Test that collective.py uses frequency-dependent resonators.

        Physical Meaning:
            Ensures that collective excitations use step resonator model
            for energy exchange analysis.
        """
        file_path = Path("bhlff/models/level_f/collective.py")
        assert file_path.exists(), f"File {file_path} not found"

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for resonator imports
        assert (
            "FrequencyDependentResonator" in content
        ), "Missing FrequencyDependentResonator import"
        assert (
            "CascadeResonatorFilter" in content
        ), "Missing CascadeResonatorFilter import"

        print("✅ Collective excitations use frequency-dependent resonators")

    def test_resonator_factory_functions(self):
        """
        Test resonator factory functions.

        Physical Meaning:
            Verifies that factory functions can create properly configured
            resonators for different use cases.
        """
        # Test frequency-dependent resonator factory
        field_shape = (10, 10, 10)
        parameters = {"R0": 0.2, "T0": 0.8, "omega0": 1.5}

        resonator = create_frequency_dependent_resonator(
            field_shape, frequency_axis=-1, parameters=parameters
        )

        assert resonator.R0 == 0.2
        assert resonator.T0 == 0.8
        assert resonator.omega0 == 1.5

        # Test cascade filter factory
        filter_parameters = {"base_R": 0.15, "base_T": 0.85}
        cascade_filter = create_cascade_filter(stages=4, parameters=filter_parameters)

        assert cascade_filter.stages == 4
        assert cascade_filter.base_R == 0.15
        assert cascade_filter.base_T == 0.85

        print("✅ Resonator factory functions work correctly")

    def test_no_exponential_in_resonator_implementation(self):
        """
        Test that resonator implementation does not use exponential functions.

        Physical Meaning:
            Ensures that step resonators use step functions and frequency
            dependence without exponential attenuation.
        """
        file_path = Path("bhlff/core/bvp/boundary/step_resonator.py")
        assert file_path.exists(), f"File {file_path} not found"

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that no exponential functions are used in core logic
        # Allow np.exp only in frequency-dependent coefficient computation
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if "np.exp(" in line and "freq_factor" not in line:
                # This should only be in frequency-dependent coefficient computation
                if "compute_coefficients" not in lines[max(0, i - 5) : i + 5]:
                    assert (
                        False
                    ), f"Found unexpected np.exp usage in line {i+1}: {line.strip()}"

        print("✅ Resonator implementation uses step functions, not exponentials")
