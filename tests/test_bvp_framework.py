"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP Framework integration tests.

This module implements comprehensive tests for the BVP framework integration
across all levels A-G, ensuring proper functionality of BVP envelope solver,
quench detection, impedance calculation, and U(1)³ phase vector structure.

Physical Meaning:
    Tests validate the Base High-Frequency Field (BVP) framework as the
    central backbone of the entire system, replacing classical patterns
    with BVP-modulational approach.

Mathematical Foundation:
    Tests validate BVP envelope equation ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    with quench detection, impedance calculation, and U(1)³ phase structure.

Example:
    >>> pytest tests/test_bvp_framework.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.bvp import BVPCore, QuenchDetector, BVPEnvelopeSolver, BVPImpedanceCalculator, PhaseVector
from bhlff.solvers.spectral import FFTSolver3D
from bhlff.solvers.integrators import TimeIntegrator


class TestBVPFramework:
    """Test suite for BVP framework integration."""

    @pytest.fixture
    def domain(self):
        """Create test domain."""
        return Domain(
            dimensions=3,
            size=(1.0, 1.0, 1.0),
            resolution=(64, 64, 64),
            boundary_conditions="periodic"
        )

    @pytest.fixture
    def bvp_config(self):
        """Create BVP configuration."""
        return {
            "carrier_frequency": 1.85e43,
            "envelope_equation": {
                "kappa_0": 1.0,
                "kappa_2": 0.1,
                "chi_prime": 1.0,
                "chi_double_prime_0": 0.01,
                "k0_squared": 1.0
            },
            "quench_detection": {
                "amplitude_threshold": 0.8,
                "detuning_threshold": 0.1,
                "gradient_threshold": 0.5
            },
            "impedance_calculation": {
                "frequency_range": [1e15, 1e20],
                "frequency_points": 1000,
                "boundary_conditions": "periodic"
            }
        }

    @pytest.fixture
    def bvp_core(self, domain, bvp_config):
        """Create BVP core instance."""
        return BVPCore(domain, bvp_config)

    def test_bvp_core_initialization(self, domain, bvp_config):
        """Test A0.1: BVP Core initialization."""
        bvp_core = BVPCore(domain, bvp_config)
        
        assert bvp_core.domain == domain
        assert bvp_core.config == bvp_config
        assert bvp_core.get_carrier_frequency() == 1.85e43
        assert bvp_core.get_phase_vector() is not None

    def test_bvp_envelope_solver(self, bvp_core, domain):
        """Test A0.1: BVP envelope solver validation."""
        # Create test source
        source = np.zeros(domain.shape)
        source[32, 32, 32] = 1.0  # Point source at center
        
        # Solve BVP envelope equation
        envelope = bvp_core.solve_envelope(source)
        
        # Validate envelope properties
        assert envelope.shape == domain.shape
        assert np.all(np.isfinite(envelope))
        assert np.max(np.abs(envelope)) > 0
        
        # Check envelope parameters
        params = bvp_core.get_envelope_parameters()
        assert "kappa_0" in params
        assert "kappa_2" in params
        assert "chi_prime" in params

    def test_bvp_quench_detection(self, bvp_core, domain):
        """Test A0.2: BVP quench detection validation."""
        # Create envelope with quench events
        envelope = np.zeros(domain.shape)
        envelope[32, 32, 32] = 1.0  # High amplitude at center
        envelope[16, 16, 16] = 0.9  # Another high amplitude
        
        # Detect quenches
        quenches = bvp_core.detect_quenches(envelope)
        
        # Validate quench detection results
        assert "quench_locations" in quenches
        assert "quench_types" in quenches
        assert "energy_dumped" in quenches
        
        # Check quench thresholds
        thresholds = bvp_core.get_quench_thresholds()
        assert "amplitude_threshold" in thresholds
        assert "detuning_threshold" in thresholds
        assert "gradient_threshold" in thresholds

    def test_bvp_u1_phase_vector(self, bvp_core):
        """Test A0.3: BVP U(1)³ phase vector validation."""
        phase_vector = bvp_core.get_phase_vector()
        
        # Get phase components
        phase_components = bvp_core.get_phase_components()
        assert len(phase_components) == 3  # Three U(1) components
        
        # Check total phase
        total_phase = bvp_core.get_total_phase()
        assert total_phase.shape == bvp_core.domain.shape
        
        # Check electroweak currents
        envelope = np.ones(bvp_core.domain.shape)
        currents = bvp_core.compute_electroweak_currents(envelope)
        assert "em_current" in currents
        assert "weak_current" in currents
        assert "mixed_current" in currents
        
        # Check phase coherence
        coherence = bvp_core.compute_phase_coherence()
        assert coherence.shape == bvp_core.domain.shape

    def test_bvp_impedance_calculation(self, bvp_core, domain):
        """Test A0.4: BVP impedance calculation validation."""
        # Create test envelope
        envelope = np.zeros(domain.shape)
        envelope[32, 32, 32] = 1.0
        
        # Compute impedance
        impedance = bvp_core.compute_impedance(envelope)
        
        # Validate impedance results
        assert "admittance" in impedance
        assert "reflection" in impedance
        assert "transmission" in impedance
        assert "peaks" in impedance
        
        # Check impedance parameters
        params = bvp_core.get_impedance_parameters()
        assert "frequency_range" in params
        assert "frequency_points" in params

    def test_bvp_fft_solver_integration(self, domain, bvp_config, bvp_core):
        """Test B1: BVP FFT solver integration."""
        # Create FFT solver with BVP integration
        fft_solver = FFTSolver3D(domain, bvp_config, bvp_core)
        
        # Test BVP envelope solving
        source = np.zeros(domain.shape)
        source[32, 32, 32] = 1.0
        
        envelope = fft_solver.solve_bvp_envelope(source)
        assert envelope.shape == domain.shape
        
        # Test quench detection
        quenches = fft_solver.detect_quenches(envelope)
        assert isinstance(quenches, dict)
        
        # Test impedance calculation
        impedance = fft_solver.compute_bvp_impedance(envelope)
        assert isinstance(impedance, dict)

    def test_bvp_time_integrator_integration(self, domain, bvp_config, bvp_core):
        """Test B2: BVP time integrator integration."""
        # Create time integrator with BVP integration
        class TestTimeIntegrator(TimeIntegrator):
            def step(self, field: np.ndarray, dt: float) -> np.ndarray:
                return field + dt * np.ones_like(field)
            
            def get_integrator_type(self) -> str:
                return "test"
        
        integrator = TestTimeIntegrator(domain, bvp_config, bvp_core)
        
        # Test quench detection
        envelope = np.ones(domain.shape)
        quenches = integrator.detect_quenches(envelope)
        assert isinstance(quenches, dict)
        
        # Test BVP core access
        assert integrator.get_bvp_core() == bvp_core

    def test_bvp_power_law_tails(self, bvp_core, domain):
        """Test B1: BVP envelope power law tails."""
        # Create point source
        source = np.zeros(domain.shape)
        source[32, 32, 32] = 1.0
        
        # Solve BVP envelope
        envelope = bvp_core.solve_envelope(source)
        
        # Check power law behavior (simplified test)
        center = np.array([32, 32, 32])
        distances = []
        amplitudes = []
        
        for i in range(domain.shape[0]):
            for j in range(domain.shape[1]):
                for k in range(domain.shape[2]):
                    if (i, j, k) != tuple(center):
                        dist = np.linalg.norm(np.array([i, j, k]) - center)
                        distances.append(dist)
                        amplitudes.append(np.abs(envelope[i, j, k]))
        
        # Basic validation (more sophisticated tests would be in level B)
        assert len(distances) > 0
        assert len(amplitudes) > 0
        assert np.max(amplitudes) > 0

    def test_bvp_monotonicity(self, bvp_core, domain):
        """Test B2: BVP envelope monotonicity."""
        # Create point source
        source = np.zeros(domain.shape)
        source[32, 32, 32] = 1.0
        
        # Solve BVP envelope
        envelope = bvp_core.solve_envelope(source)
        
        # Check monotonicity (simplified test)
        center = np.array([32, 32, 32])
        radial_profile = []
        
        for r in range(1, 20):
            count = 0
            total_amp = 0
            for i in range(domain.shape[0]):
                for j in range(domain.shape[1]):
                    for k in range(domain.shape[2]):
                        dist = np.linalg.norm(np.array([i, j, k]) - center)
                        if abs(dist - r) < 0.5:
                            total_amp += np.abs(envelope[i, j, k])
                            count += 1
            if count > 0:
                radial_profile.append(total_amp / count)
        
        # Basic monotonicity check
        if len(radial_profile) > 1:
            # Should generally decrease with distance
            assert radial_profile[0] > radial_profile[-1]

    def test_bvp_topological_charge(self, bvp_core):
        """Test B3: BVP topological charge."""
        phase_vector = bvp_core.get_phase_vector()
        
        # Test topological charge calculation
        # This would be more sophisticated in actual implementation
        total_phase = bvp_core.get_total_phase()
        assert total_phase.shape == bvp_core.domain.shape
        
        # Check SU(2) coupling
        coupling_strength = bvp_core.get_su2_coupling_strength()
        assert isinstance(coupling_strength, float)
        assert coupling_strength >= 0

    def test_bvp_zone_separation(self, bvp_core, domain):
        """Test B4: BVP zone separation."""
        # Create test envelope
        envelope = np.zeros(domain.shape)
        envelope[32, 32, 32] = 1.0
        
        # Compute impedance for zone analysis
        impedance = bvp_core.compute_impedance(envelope)
        
        # Basic zone separation validation
        assert "peaks" in impedance
        # More sophisticated zone analysis would be in level B

    def test_bvp_boundary_effects(self, bvp_core, domain):
        """Test C1: BVP boundary effects."""
        # Create envelope with boundary effects
        envelope = np.zeros(domain.shape)
        envelope[32, 32, 32] = 1.0
        
        # Test impedance calculation for boundary effects
        impedance = bvp_core.compute_impedance(envelope)
        
        # Validate boundary function calculation
        assert "admittance" in impedance
        assert "reflection" in impedance
        assert "transmission" in impedance

    def test_bvp_quench_memory(self, bvp_core, domain):
        """Test C3: BVP quench memory/pinning."""
        # Create envelope with quench events
        envelope = np.zeros(domain.shape)
        envelope[32, 32, 32] = 1.0
        
        # Detect quenches
        quenches = bvp_core.detect_quenches(envelope)
        
        # Validate quench memory effects
        assert "quench_locations" in quenches
        assert "energy_dumped" in quenches
        
        # Test quench threshold modification
        new_thresholds = {
            "amplitude_threshold": 0.9,
            "detuning_threshold": 0.2,
            "gradient_threshold": 0.6
        }
        bvp_core.set_quench_thresholds(new_thresholds)
        
        updated_thresholds = bvp_core.get_quench_thresholds()
        assert updated_thresholds["amplitude_threshold"] == 0.9

    def test_bvp_parameter_scaling(self, domain, bvp_config):
        """Test A2: BVP parameter scaling."""
        # Test with different scales
        config1 = bvp_config.copy()
        config2 = bvp_config.copy()
        config2["carrier_frequency"] = 2.0 * config1["carrier_frequency"]
        
        bvp_core1 = BVPCore(domain, config1)
        bvp_core2 = BVPCore(domain, config2)
        
        # Create same source
        source = np.zeros(domain.shape)
        source[32, 32, 32] = 1.0
        
        # Solve with both configurations
        envelope1 = bvp_core1.solve_envelope(source)
        envelope2 = bvp_core2.solve_envelope(source)
        
        # Basic scaling validation
        assert envelope1.shape == envelope2.shape
        assert np.all(np.isfinite(envelope1))
        assert np.all(np.isfinite(envelope2))

    def test_bvp_energy_balance(self, bvp_core, domain):
        """Test A0.5: BVP energy balance."""
        # Create test source
        source = np.zeros(domain.shape)
        source[32, 32, 32] = 1.0
        
        # Solve BVP envelope
        envelope = bvp_core.solve_envelope(source)
        
        # Basic energy balance validation
        total_energy = np.sum(np.abs(envelope)**2)
        assert total_energy > 0
        assert np.all(np.isfinite(envelope))
        
        # Check envelope parameters for energy calculation
        params = bvp_core.get_envelope_parameters()
        assert "kappa_0" in params
        assert "chi_prime" in params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
