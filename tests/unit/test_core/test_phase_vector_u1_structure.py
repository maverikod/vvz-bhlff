"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for U(1)³ Phase Vector structure.

This module provides comprehensive unit tests for the U(1)³ Phase Vector
implementation, ensuring physical consistency and theoretical correctness.

Physical Meaning:
    Tests validate U(1)³ phase vector:
    - Three independent U(1) phase components
    - Phase coherence maintenance
    - Topological charge quantization
    - Electroweak current generation

Mathematical Foundation:
    Tests U(1)³ phase structure: Θ = (Θ₁, Θ₂, Θ₃)
    and validates phase coherence and topological properties.

Example:
    >>> pytest tests/unit/test_core/test_phase_vector_u1_structure.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.bvp.phase_vector.phase_vector import PhaseVector
from bhlff.core.bvp.phase_vector.phase_components import PhaseComponents
from bhlff.core.bvp.phase_vector.electroweak_coupling import ElectroweakCoupling


class TestPhaseVectorU1Structure:
    """U(1)³ Phase Vector structure unit tests."""

    @pytest.fixture
    def domain_7d_small(self):
        """Create small 7D domain for testing."""
        return Domain(L=1.0, N=2, dimensions=7)

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain(L=1.0, N=2, dimensions=7)

    @pytest.fixture
    def phase_vector_config(self):
        """Create phase vector configuration."""
        return {
            "phase_components": {
                "amplitude_1": 1.0,
                "amplitude_2": 0.8,
                "amplitude_3": 0.6,
                "frequency_1": 1.0,
                "frequency_2": 1.5,
                "frequency_3": 2.0,
            },
            "su2_coupling": {"coupling_strength": 0.1},
            "electroweak": {
                "em_coupling": 1.0,
                "weak_coupling": 0.1,
                "mixing_angle": 0.23,
                "gauge_coupling": 0.65,
            },
        }

    @pytest.fixture
    def phase_vector(self, domain_7d_small, phase_vector_config):
        """Create PhaseVector instance."""
        return PhaseVector(domain_7d_small, phase_vector_config)

    def test_phase_vector_initialization(self, phase_vector, domain_7d_small):
        """
        Test PhaseVector initialization.

        Physical Meaning:
            Validates that PhaseVector is properly initialized with
            three U(1) phase components and electroweak coupling.
        """
        assert phase_vector.domain == domain_7d_small
        assert phase_vector.config is not None
        assert phase_vector.constants is not None
        assert hasattr(phase_vector, "_phase_components")
        assert hasattr(phase_vector, "_electroweak_coupling")
        assert hasattr(phase_vector, "coupling_matrix")

        # Check SU(2) coupling matrix
        assert phase_vector.coupling_matrix.shape == (3, 3)
        assert np.allclose(np.diag(phase_vector.coupling_matrix), 1.0)

    def test_phase_components_structure(self, phase_vector):
        """
        Test U(1)³ phase components structure.

        Physical Meaning:
            Validates that three independent U(1) phase components
            are properly initialized and managed.
        """
        components = phase_vector.get_phase_components()

        # Should have exactly 3 components
        assert len(components) == 3

        # Each component should be a complex array
        for i, component in enumerate(components):
            assert isinstance(component, np.ndarray)
            assert component.dtype == complex
            assert component.shape == phase_vector.domain.shape

    def test_total_phase_computation(self, phase_vector):
        """
        Test total phase computation.

        Physical Meaning:
            Validates that total phase is correctly computed from
            the three U(1) components with proper coupling.
        """
        total_phase = phase_vector.get_total_phase()

        assert isinstance(total_phase, np.ndarray)
        assert total_phase.shape == phase_vector.domain.shape
        assert total_phase.dtype == complex

    def test_phase_decomposition(self, phase_vector):
        """
        Test phase structure decomposition.

        Physical Meaning:
            Validates that phase structure can be decomposed into
            amplitude and three phase components.
        """
        # Create test envelope
        envelope = np.ones(phase_vector.domain.shape, dtype=complex)
        envelope *= np.exp(1j * np.pi / 4)  # Add phase

        amplitude, phases = phase_vector.decompose_phase_structure(envelope)

        # Check amplitude
        assert isinstance(amplitude, np.ndarray)
        assert amplitude.shape == phase_vector.domain.shape
        assert np.all(amplitude >= 0)  # Amplitude should be non-negative

        # Check phases
        assert isinstance(phases, list)
        assert len(phases) == 3  # Three U(1) components

        for phase in phases:
            assert isinstance(phase, np.ndarray)
            assert phase.shape == phase_vector.domain.shape
            # Phases should be in [0, 2π)
            assert np.all(phase >= 0)
            assert np.all(phase < 2 * np.pi)

    def test_topological_charge_computation(self, phase_vector):
        """
        Test topological charge computation.

        Physical Meaning:
            Validates that topological charge is correctly computed
            and is quantized according to the theory.
        """
        # Create test envelope with known topology
        envelope = np.ones(phase_vector.domain.shape, dtype=complex)

        # Add phase winding for non-zero topological charge
        if phase_vector.domain.dimensions >= 1:
            # Create proper coordinate arrays for 7D domain
            if phase_vector.domain.dimensions == 7:
                # For 7D: ℝ³ₓ × 𝕋³_φ × ℝₜ
                x = np.linspace(
                    -phase_vector.domain.L / 2,
                    phase_vector.domain.L / 2,
                    phase_vector.domain.N,
                )
                # Apply phase winding only to first spatial dimension
                envelope = envelope * np.exp(
                    1j
                    * 2
                    * np.pi
                    * x[:, None, None, None, None, None, None]
                    / phase_vector.domain.L
                )
            else:
                x = np.linspace(
                    -phase_vector.domain.L / 2,
                    phase_vector.domain.L / 2,
                    phase_vector.domain.N,
                )
                envelope *= np.exp(1j * 2 * np.pi * x / phase_vector.domain.L)

        topological_charge = phase_vector.compute_topological_charge(envelope)

        assert isinstance(topological_charge, float)
        assert np.isfinite(topological_charge)
        # Topological charge should be quantized (close to integer)
        assert np.isclose(topological_charge, np.round(topological_charge), atol=1e-6)

    def test_phase_coherence_computation(self, phase_vector):
        """
        Test phase coherence computation.

        Physical Meaning:
            Validates that phase coherence is correctly computed
            and is in the range [0, 1].
        """
        # Create test envelope
        envelope = np.ones(phase_vector.domain.shape, dtype=complex)
        envelope *= np.exp(1j * np.pi / 4)  # Add phase

        coherence = phase_vector.compute_phase_coherence(envelope)

        assert isinstance(coherence, float)
        assert 0 <= coherence <= 1  # Coherence should be in [0, 1]
        assert np.isfinite(coherence)

    def test_electroweak_currents_computation(self, phase_vector):
        """
        Test electroweak currents computation.

        Physical Meaning:
            Validates that electroweak currents are correctly computed
            as functionals of the envelope.
        """
        # Create test envelope
        envelope = np.ones(phase_vector.domain.shape, dtype=complex)
        envelope *= np.exp(1j * np.pi / 4)  # Add phase

        currents = phase_vector.compute_electroweak_currents(envelope)

        # Check current structure
        assert isinstance(currents, dict)
        assert "em_current" in currents
        assert "weak_current" in currents
        assert "mixed_current" in currents

        # Check current properties
        for current_name, current in currents.items():
            assert isinstance(current, np.ndarray)
            assert current.shape == phase_vector.domain.shape
            assert np.all(np.isfinite(current))

    def test_su2_coupling_management(self, phase_vector):
        """
        Test SU(2) coupling management.

        Physical Meaning:
            Validates that SU(2) coupling strength can be properly
            managed and updated.
        """
        # Get initial coupling strength
        initial_strength = phase_vector.get_su2_coupling_strength()
        assert isinstance(initial_strength, float)
        assert initial_strength > 0

        # Update coupling strength
        new_strength = 0.2
        phase_vector.set_su2_coupling_strength(new_strength)

        # Check updated strength
        updated_strength = phase_vector.get_su2_coupling_strength()
        assert np.isclose(updated_strength, new_strength)

    def test_electroweak_coefficients_management(self, phase_vector):
        """
        Test electroweak coefficients management.

        Physical Meaning:
            Validates that electroweak coupling coefficients can be
            properly managed and updated.
        """
        # Get initial coefficients
        initial_coeffs = phase_vector.get_electroweak_coefficients()
        assert isinstance(initial_coeffs, dict)
        assert "em_coupling" in initial_coeffs
        assert "weak_coupling" in initial_coeffs
        assert "mixing_angle" in initial_coeffs
        assert "gauge_coupling" in initial_coeffs

        # Update coefficients
        new_coeffs = {
            "em_coupling": 1.5,
            "weak_coupling": 0.15,
            "mixing_angle": 0.25,
            "gauge_coupling": 0.75,
        }
        phase_vector.set_electroweak_coefficients(new_coeffs)

        # Check updated coefficients
        updated_coeffs = phase_vector.get_electroweak_coefficients()
        for key, value in new_coeffs.items():
            assert np.isclose(updated_coeffs[key], value)

    def test_phase_components_update(self, phase_vector):
        """
        Test phase components update from envelope.

        Physical Meaning:
            Validates that phase components can be updated from
            a solved BVP envelope field.
        """
        # Create test envelope
        envelope = np.ones(phase_vector.domain.shape, dtype=complex)
        envelope *= np.exp(1j * np.pi / 4)  # Add phase

        # Update phase components
        phase_vector.update_phase_components(envelope)

        # Check that components were updated
        components = phase_vector.get_phase_components()
        assert len(components) == 3

        for component in components:
            assert isinstance(component, np.ndarray)
            assert component.shape == phase_vector.domain.shape

    def test_7d_phase_structure(self, domain_7d, phase_vector_config):
        """
        Test 7D phase structure.

        Physical Meaning:
            Validates that U(1)³ phase structure works correctly
            in 7D space-time according to the theory.
        """
        phase_vector_7d = PhaseVector(domain_7d, phase_vector_config)

        # Test 7D phase components
        components = phase_vector_7d.get_phase_components()
        assert len(components) == 3

        for component in components:
            assert component.shape == domain_7d.shape
            assert component.ndim == 7  # 7D structure

        # Test 7D total phase
        total_phase = phase_vector_7d.get_total_phase()
        assert total_phase.shape == domain_7d.shape
        assert total_phase.ndim == 7

    def test_phase_vector_repr(self, phase_vector):
        """
        Test PhaseVector string representation.

        Physical Meaning:
            Validates that PhaseVector has proper string representation
            for debugging and logging.
        """
        repr_str = repr(phase_vector)
        assert isinstance(repr_str, str)
        assert "PhaseVector" in repr_str
        assert "domain" in repr_str
        assert "su2_coupling" in repr_str
        assert "em_coupling" in repr_str


class TestPhaseComponents:
    """PhaseComponents unit tests."""

    @pytest.fixture
    def domain_7d_small(self):
        """Create small 7D domain for testing."""
        return Domain(L=1.0, N=2, dimensions=7)

    @pytest.fixture
    def phase_components_config(self):
        """Create phase components configuration."""
        return {
            "phase_components": {
                "amplitude_1": 1.0,
                "amplitude_2": 0.8,
                "amplitude_3": 0.6,
                "frequency_1": 1.0,
                "frequency_2": 1.5,
                "frequency_3": 2.0,
            }
        }

    @pytest.fixture
    def phase_components(self, domain_7d_small, phase_components_config):
        """Create PhaseComponents instance."""
        return PhaseComponents(domain_7d_small, phase_components_config)

    def test_phase_components_initialization(self, phase_components, domain_7d_small):
        """
        Test PhaseComponents initialization.

        Physical Meaning:
            Validates that PhaseComponents is properly initialized
            with three U(1) phase components.
        """
        assert phase_components.domain == domain_7d_small
        assert phase_components.config is not None
        assert len(phase_components.theta_components) == 3

        for component in phase_components.theta_components:
            assert isinstance(component, np.ndarray)
            assert component.shape == domain_7d_small.shape
            assert component.dtype == complex

    def test_phase_components_get_components(self, phase_components):
        """
        Test getting phase components.

        Physical Meaning:
            Validates that phase components can be retrieved
            as a list of three U(1) components.
        """
        components = phase_components.get_components()

        assert isinstance(components, list)
        assert len(components) == 3

        for component in components:
            assert isinstance(component, np.ndarray)
            assert component.shape == phase_components.domain.shape

    def test_phase_components_total_phase(self, phase_components):
        """
        Test total phase computation.

        Physical Meaning:
            Validates that total phase is correctly computed
            from the three U(1) components.
        """
        total_phase = phase_components.get_total_phase()

        assert isinstance(total_phase, np.ndarray)
        assert total_phase.shape == phase_components.domain.shape
        assert total_phase.dtype == complex

    def test_phase_components_coherence(self, phase_components):
        """
        Test phase coherence computation.

        Physical Meaning:
            Validates that phase coherence is correctly computed
            across the three U(1) components.
        """
        coherence = phase_components.compute_phase_coherence()

        assert isinstance(coherence, np.ndarray)
        assert coherence.shape == phase_components.domain.shape
        assert np.all(coherence >= 0)  # Coherence should be non-negative
        assert np.all(coherence <= 1)  # Coherence should be <= 1


class TestElectroweakCoupling:
    """ElectroweakCoupling unit tests."""

    @pytest.fixture
    def electroweak_config(self):
        """Create electroweak configuration."""
        return {
            "electroweak": {
                "em_coupling": 1.0,
                "weak_coupling": 0.1,
                "mixing_angle": 0.23,
                "gauge_coupling": 0.65,
            }
        }

    @pytest.fixture
    def electroweak_coupling(self, electroweak_config):
        """Create ElectroweakCoupling instance."""
        return ElectroweakCoupling(electroweak_config)

    def test_electroweak_coupling_initialization(self, electroweak_coupling):
        """
        Test ElectroweakCoupling initialization.

        Physical Meaning:
            Validates that ElectroweakCoupling is properly initialized
            with correct coefficients.
        """
        assert electroweak_coupling.config is not None
        assert hasattr(electroweak_coupling, "electroweak_coefficients")

        coeffs = electroweak_coupling.electroweak_coefficients
        assert "em_coupling" in coeffs
        assert "weak_coupling" in coeffs
        assert "mixing_angle" in coeffs
        assert "gauge_coupling" in coeffs

    def test_electroweak_coefficients_management(self, electroweak_coupling):
        """
        Test electroweak coefficients management.

        Physical Meaning:
            Validates that electroweak coupling coefficients can be
            properly managed and updated.
        """
        # Get initial coefficients
        initial_coeffs = electroweak_coupling.get_electroweak_coefficients()
        assert isinstance(initial_coeffs, dict)

        # Update coefficients
        new_coeffs = {
            "em_coupling": 1.5,
            "weak_coupling": 0.15,
            "mixing_angle": 0.25,
            "gauge_coupling": 0.75,
        }
        electroweak_coupling.set_electroweak_coefficients(new_coeffs)

        # Check updated coefficients
        updated_coeffs = electroweak_coupling.get_electroweak_coefficients()
        for key, value in new_coeffs.items():
            assert np.isclose(updated_coeffs[key], value)

    def test_weinberg_angle_management(self, electroweak_coupling):
        """
        Test Weinberg angle management.

        Physical Meaning:
            Validates that Weinberg mixing angle can be properly
            managed and updated.
        """
        # Get initial angle
        initial_angle = electroweak_coupling.get_weinberg_angle()
        assert isinstance(initial_angle, float)
        assert 0 <= initial_angle <= np.pi / 2

        # Update angle
        new_angle = 0.3
        electroweak_coupling.set_weinberg_angle(new_angle)

        # Check updated angle
        updated_angle = electroweak_coupling.get_weinberg_angle()
        assert np.isclose(updated_angle, new_angle)
