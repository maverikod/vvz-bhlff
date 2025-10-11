"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test suite for step 1.1.1 exponential decay fix in gravitational models.

This module tests that exponential decay functions have been replaced
with step resonator models according to 7D BVP theory principles.
"""

import pytest
import numpy as np
from bhlff.models.level_g.gravity_einstein import PhaseEnvelopeBalanceSolver
from bhlff.models.level_g.gravity_waves import VBPGravitationalWavesCalculator


class TestStep111ExponentialDecayFix:
    """Test suite for step 1.1.1 exponential decay fix."""
    
    def test_no_exponential_decay_in_gravity_einstein(self):
        """Test that gravity_einstein.py uses step resonator instead of exponential decay."""
        # Read the file content
        with open("bhlff/models/level_g/gravity_einstein.py", "r") as f:
            content = f.read()
        
        # Check that exponential decay is not used in the main logic
        assert "np.exp(-k_magnitude / 10.0)" not in content, \
            "Found exponential decay in gravity_einstein.py - violates 7D BVP theory"
        
        # Check that step resonator method is present
        assert "_step_resonator_transmission" in content, \
            "Step resonator method not found in gravity_einstein.py"
        
        # Check that step resonator is used instead of exponential
        assert "self._step_resonator_transmission(k_magnitude)" in content, \
            "Step resonator not used in gravity_einstein.py"
    
    def test_no_exponential_decay_in_gravity_waves(self):
        """Test that gravity_waves.py uses step resonator instead of exponential decay."""
        # Read the file content
        with open("bhlff/models/level_g/gravity_waves.py", "r") as f:
            content = f.read()
        
        # Check that exponential decay is not used in the main logic
        assert "np.exp(-dt / self.params.get(\"damping_time\"" not in content, \
            "Found exponential temporal damping in gravity_waves.py - violates 7D BVP theory"
        
        assert "np.exp(-dx / self.params.get(\"spatial_damping\"" not in content, \
            "Found exponential spatial damping in gravity_waves.py - violates 7D BVP theory"
        
        # Check that step resonator methods are present
        assert "_step_resonator_boundary_condition" in content, \
            "Step resonator boundary condition method not found in gravity_waves.py"
        
        assert "_step_resonator_spatial_boundary" in content, \
            "Step resonator spatial boundary method not found in gravity_waves.py"
        
        # Check that step resonator methods are used
        assert "self._step_resonator_boundary_condition(dt)" in content, \
            "Step resonator boundary condition not used in gravity_waves.py"
        
        assert "self._step_resonator_spatial_boundary(dx)" in content, \
            "Step resonator spatial boundary not used in gravity_waves.py"
    
    def test_step_resonator_implementation(self):
        """Test that step resonator methods are properly implemented."""
        # Test parameters
        params = {
            "resonator_cutoff_frequency": 10.0,
            "transmission_coefficient": 0.9,
            "resonator_time_cutoff": 1.0,
            "resonator_spatial_cutoff": 1.0
        }
        
        # Create mock domain
        class MockDomain:
            def __init__(self):
                self.N = 64
                self.L = 1.0
        
        domain = MockDomain()
        
        # Test gravity_einstein.py step resonator
        solver = PhaseEnvelopeBalanceSolver(domain, params)
        
        # Test step resonator transmission
        k_magnitude = np.array([5.0, 15.0, 25.0])
        transmission = solver._step_resonator_transmission(k_magnitude)
        
        # Check that transmission is step function
        expected = np.array([0.9, 0.0, 0.0])  # 0.9 below cutoff, 0.0 above
        np.testing.assert_array_equal(transmission, expected)
        
        # Test gravity_waves.py step resonator
        wave_calc = VBPGravitationalWavesCalculator(domain, params)
        
        # Test temporal boundary condition
        dt_values = np.array([0.5, 1.5, 2.0])
        temporal_damping = [wave_calc._step_resonator_boundary_condition(dt) for dt in dt_values]
        expected_temporal = [0.9, 0.0, 0.0]  # 0.9 below cutoff, 0.0 above
        np.testing.assert_array_equal(temporal_damping, expected_temporal)
        
        # Test spatial boundary condition
        dx_values = np.array([0.5, 1.5, 2.0])
        spatial_damping = [wave_calc._step_resonator_spatial_boundary(dx) for dx in dx_values]
        expected_spatial = [0.9, 0.0, 0.0]  # 0.9 below cutoff, 0.0 above
        np.testing.assert_array_equal(spatial_damping, expected_spatial)
    
    def test_no_exponential_in_other_models(self):
        """Test that other model files don't use exponential decay."""
        # Check multi-particle models
        model_files = [
            "bhlff/models/level_f/multi_particle_potential.py",
            "bhlff/models/level_f/multi_particle_modes.py",
            "bhlff/models/level_c/resonators/resonator_analyzer.py",
            "bhlff/models/level_c/resonators/resonator_spectrum.py",
            "bhlff/models/level_c/memory/memory_evolution.py",
            "bhlff/models/level_c/memory/memory_analyzer.py"
        ]
        
        for file_path in model_files:
            try:
                with open(file_path, "r") as f:
                    content = f.read()
                
                # Check that exponential decay is not used in main logic
                assert "np.exp(-" not in content or "classical comparison" in content, \
                    f"Found exponential decay in {file_path} - violates 7D BVP theory"
                
            except FileNotFoundError:
                # File might not exist, skip
                pass
    
    def test_step_resonator_physics_correctness(self):
        """Test that step resonator model follows 7D BVP theory principles."""
        # Test parameters
        params = {
            "resonator_cutoff_frequency": 10.0,
            "transmission_coefficient": 0.9,
            "resonator_time_cutoff": 1.0,
            "resonator_spatial_cutoff": 1.0
        }
        
        # Create mock domain
        class MockDomain:
            def __init__(self):
                self.N = 64
                self.L = 1.0
        
        domain = MockDomain()
        
        # Test that step resonator follows 7D BVP theory
        solver = PhaseEnvelopeBalanceSolver(domain, params)
        
        # Test frequency response
        frequencies = np.linspace(0, 20, 100)
        transmission = solver._step_resonator_transmission(frequencies)
        
        # Check that transmission is 1.0 below cutoff, 0.0 above
        below_cutoff = frequencies < 10.0
        above_cutoff = frequencies >= 10.0
        
        assert np.all(transmission[below_cutoff] == 0.9), \
            "Step resonator should transmit below cutoff frequency"
        
        assert np.all(transmission[above_cutoff] == 0.0), \
            "Step resonator should not transmit above cutoff frequency"
        
        # Test that step resonator is discontinuous (step function)
        cutoff_index = np.where(frequencies >= 10.0)[0][0]
        assert transmission[cutoff_index-1] == 0.9, \
            "Step resonator should be 0.9 just below cutoff"
        assert transmission[cutoff_index] == 0.0, \
            "Step resonator should be 0.0 just above cutoff"
    
    def test_classical_comparison_preserved(self):
        """Test that classical comparison patterns are preserved for comparison."""
        # This test ensures that classical patterns used for comparison
        # are not removed, only marked appropriately
        
        # Check that test files still contain exponential functions for comparison
        test_files = [
            "tests/unit/test_level_g/test_vbp_gravitational_waves_physics.py",
            "tests/unit/test_level_g/test_vbp_gravitational_effects_integration.py"
        ]
        
        for file_path in test_files:
            try:
                with open(file_path, "r") as f:
                    content = f.read()
                
                # Test files should contain exponential functions for comparison
                # but they should be marked as classical comparison
                if "np.exp(" in content:
                    assert "classical comparison" in content or "for comparison" in content, \
                        f"Exponential functions in {file_path} should be marked as classical comparison"
                
            except FileNotFoundError:
                # File might not exist, skip
                pass