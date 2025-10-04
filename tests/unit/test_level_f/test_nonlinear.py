"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for NonlinearEffects class in Level F models.

This module contains comprehensive tests for the NonlinearEffects
class, including tests for nonlinear interactions, soliton solutions,
and stability analysis.

Physical Meaning:
    Tests verify that nonlinear effects are correctly
    implemented in multi-particle systems, including
    nonlinear modes, solitons, and stability analysis.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from bhlff.models.level_f.nonlinear import NonlinearEffects
from bhlff.models.level_f.multi_particle import MultiParticleSystem, Particle
from bhlff.core.domain import Domain


class TestNonlinearEffects:
    """
    Test cases for NonlinearEffects class.
    
    Physical Meaning:
        Tests verify the correct implementation of nonlinear
        effects including cubic, quartic, and sine-Gordon
        nonlinearities.
    """
    
    @pytest.fixture
    def domain(self):
        """Create test domain."""
        return Domain(L=20.0, N=16, N_phi=8, N_t=16, T=10.0, dimensions=7)
    
    @pytest.fixture
    def particles(self):
        """Create test particles."""
        return [
            Particle(position=np.array([5.0, 10.0, 10.0]), charge=1, phase=0.0),
            Particle(position=np.array([15.0, 10.0, 10.0]), charge=-1, phase=np.pi)
        ]
    
    @pytest.fixture
    def system(self, domain, particles):
        """Create test system."""
        return MultiParticleSystem(domain, particles, interaction_range=5.0)
    
    @pytest.fixture
    def nonlinear_params(self):
        """Create test nonlinear parameters."""
        return {
            'strength': 1.0,
            'order': 3,
            'type': 'cubic',
            'coupling': 'local'
        }
    
    @pytest.fixture
    def nonlinear(self, system, nonlinear_params):
        """Create test nonlinear effects."""
        return NonlinearEffects(system, nonlinear_params)
    
    def test_initialization(self, system, nonlinear_params):
        """
        Test nonlinear effects initialization.
        
        Physical Meaning:
            Verifies that the nonlinear effects model is
            correctly initialized with the specified parameters.
        """
        nonlinear = NonlinearEffects(system, nonlinear_params)
        
        assert nonlinear.system == system
        assert nonlinear.nonlinear_strength == 1.0
        assert nonlinear.nonlinear_order == 3
        assert nonlinear.nonlinear_type == 'cubic'
        assert nonlinear.coupling_type == 'local'
    
    def test_cubic_nonlinearity_setup(self, system):
        """
        Test cubic nonlinearity setup.
        
        Physical Meaning:
            Verifies that cubic nonlinearity is correctly
            set up with the appropriate potential and derivative.
        """
        params = {'strength': 1.0, 'order': 3, 'type': 'cubic'}
        nonlinear = NonlinearEffects(system, params)
        
        # Test nonlinear potential
        psi = np.array([0.1, 0.5, 1.0])
        potential = nonlinear.nonlinear_potential(psi)
        expected = 1.0 * np.abs(psi)**3
        np.testing.assert_allclose(potential, expected)
        
        # Test nonlinear derivative
        derivative = nonlinear.nonlinear_derivative(psi)
        expected = 3 * 1.0 * np.abs(psi) * np.sign(psi)
        np.testing.assert_allclose(derivative, expected)
    
    def test_quartic_nonlinearity_setup(self, system):
        """
        Test quartic nonlinearity setup.
        
        Physical Meaning:
            Verifies that quartic nonlinearity is correctly
            set up with the appropriate potential and derivative.
        """
        params = {'strength': 1.0, 'order': 4, 'type': 'quartic'}
        nonlinear = NonlinearEffects(system, params)
        
        # Test nonlinear potential
        psi = np.array([0.1, 0.5, 1.0])
        potential = nonlinear.nonlinear_potential(psi)
        expected = 1.0 * np.abs(psi)**4
        np.testing.assert_allclose(potential, expected)
        
        # Test nonlinear derivative
        derivative = nonlinear.nonlinear_derivative(psi)
        expected = 4 * 1.0 * np.abs(psi)**2 * np.sign(psi)
        np.testing.assert_allclose(derivative, expected)
    
    def test_sine_gordon_nonlinearity_setup(self, system):
        """
        Test sine-Gordon nonlinearity setup.
        
        Physical Meaning:
            Verifies that sine-Gordon nonlinearity is correctly
            set up with the appropriate potential and derivative.
        """
        params = {'strength': 1.0, 'type': 'sine_gordon'}
        nonlinear = NonlinearEffects(system, params)
        
        # Test nonlinear potential
        psi = np.array([0.1, 0.5, 1.0])
        potential = nonlinear.nonlinear_potential(psi)
        expected = 1.0 * (1 - np.cos(psi))
        np.testing.assert_allclose(potential, expected)
        
        # Test nonlinear derivative
        derivative = nonlinear.nonlinear_derivative(psi)
        expected = 1.0 * np.sin(psi)
        np.testing.assert_allclose(derivative, expected)
    
    def test_add_nonlinear_interactions(self, nonlinear):
        """
        Test adding nonlinear interactions.
        
        Physical Meaning:
            Verifies that nonlinear interactions are correctly
            added to the system.
        """
        new_params = {
            'strength': 2.0,
            'order': 4,
            'type': 'quartic'
        }
        
        nonlinear.add_nonlinear_interactions(new_params)
        
        # Check that parameters are updated
        assert nonlinear.nonlinear_strength == 2.0
        assert nonlinear.nonlinear_order == 4
        assert nonlinear.nonlinear_type == 'quartic'
    
    def test_find_nonlinear_modes(self, nonlinear):
        """
        Test finding nonlinear modes.
        
        Physical Meaning:
            Verifies that nonlinear modes are correctly
            identified in the system.
        """
        modes = nonlinear.find_nonlinear_modes()
        
        # Check that modes are returned
        assert 'linear_frequencies' in modes
        assert 'nonlinear_frequencies' in modes
        assert 'amplitudes' in modes
        assert 'stability' in modes
        assert 'bifurcations' in modes
        
        # Check that frequencies are returned
        assert isinstance(modes['linear_frequencies'], np.ndarray)
        assert isinstance(modes['nonlinear_frequencies'], np.ndarray)
        
        # Check that frequencies are non-negative
        assert np.all(modes['linear_frequencies'] >= 0)
        assert np.all(modes['nonlinear_frequencies'] >= 0)
    
    def test_find_soliton_solutions(self, nonlinear):
        """
        Test finding soliton solutions.
        
        Physical Meaning:
            Verifies that soliton solutions are correctly
            identified in the system.
        """
        solitons = nonlinear.find_soliton_solutions()
        
        # Check that solitons are returned
        assert 'solitons' in solitons
        assert 'profiles' in solitons
        assert 'velocities' in solitons
        assert 'stability' in solitons
        
        # Check that solitons are found
        assert len(solitons['solitons']) > 0
        assert len(solitons['profiles']) > 0
        assert len(solitons['velocities']) > 0
        assert len(solitons['stability']) > 0
    
    def test_sine_gordon_solitons(self, system):
        """
        Test sine-Gordon soliton solutions.
        
        Physical Meaning:
            Verifies that sine-Gordon solitons are correctly
            identified.
        """
        params = {'strength': 1.0, 'type': 'sine_gordon'}
        nonlinear = NonlinearEffects(system, params)
        
        solitons = nonlinear._find_sine_gordon_solitons()
        
        # Check that solitons are returned
        assert len(solitons) > 0
        
        # Check that solitons have required properties
        for soliton in solitons:
            assert 'type' in soliton
            assert 'velocity' in soliton
            assert 'amplitude' in soliton
            assert 'width' in soliton
            assert 'position' in soliton
            assert 'stability' in soliton
    
    def test_cubic_solitons(self, system):
        """
        Test cubic soliton solutions.
        
        Physical Meaning:
            Verifies that cubic solitons are correctly
            identified.
        """
        params = {'strength': 1.0, 'type': 'cubic'}
        nonlinear = NonlinearEffects(system, params)
        
        solitons = nonlinear._find_cubic_solitons()
        
        # Check that solitons are returned
        assert len(solitons) > 0
        
        # Check that solitons have required properties
        for soliton in solitons:
            assert 'type' in soliton
            assert 'velocity' in soliton
            assert 'amplitude' in soliton
            assert 'width' in soliton
            assert 'position' in soliton
            assert 'stability' in soliton
    
    def test_quartic_solitons(self, system):
        """
        Test quartic soliton solutions.
        
        Physical Meaning:
            Verifies that quartic solitons are correctly
            identified.
        """
        params = {'strength': 1.0, 'type': 'quartic'}
        nonlinear = NonlinearEffects(system, params)
        
        solitons = nonlinear._find_quartic_solitons()
        
        # Check that solitons are returned
        assert len(solitons) > 0
        
        # Check that solitons have required properties
        for soliton in solitons:
            assert 'type' in soliton
            assert 'velocity' in soliton
            assert 'amplitude' in soliton
            assert 'width' in soliton
            assert 'position' in soliton
            assert 'stability' in soliton
    
    def test_soliton_profile_computation(self, nonlinear):
        """
        Test soliton profile computation.
        
        Physical Meaning:
            Verifies that soliton profiles are correctly
            computed.
        """
        soliton = {
            'type': 'kink',
            'velocity': 0.5,
            'amplitude': 2.0,
            'width': 1.0,
            'position': 0.0
        }
        
        profile = nonlinear._compute_soliton_profile(soliton)
        
        # Check that profile is returned
        assert isinstance(profile, np.ndarray)
        assert len(profile) > 0
        
        # Check that profile is finite
        assert np.all(np.isfinite(profile))
    
    def test_nonlinear_corrections(self, nonlinear):
        """
        Test nonlinear corrections computation.
        
        Physical Meaning:
            Verifies that nonlinear corrections are correctly
            computed for linear modes.
        """
        linear_modes = {
            'frequencies': np.array([1.0, 2.0, 3.0]),
            'amplitudes': np.array([[0.5, 0.3, 0.2], [0.4, 0.6, 0.1], [0.3, 0.2, 0.5]])
        }
        
        corrections = nonlinear._compute_nonlinear_corrections(linear_modes)
        
        # Check that corrections are returned
        assert 'frequencies' in corrections
        assert 'amplitudes' in corrections
        assert 'frequency_shifts' in corrections
        assert 'amplitude_corrections' in corrections
        
        # Check that frequencies are returned
        assert isinstance(corrections['frequencies'], np.ndarray)
        assert len(corrections['frequencies']) == len(linear_modes['frequencies'])
        
        # Check that amplitudes are returned
        assert isinstance(corrections['amplitudes'], np.ndarray)
        assert corrections['amplitudes'].shape == linear_modes['amplitudes'].shape
    
    def test_bifurcation_points(self, nonlinear):
        """
        Test bifurcation points identification.
        
        Physical Meaning:
            Verifies that bifurcation points are correctly
            identified in the system.
        """
        bifurcations = nonlinear._find_bifurcation_points()
        
        # Check that bifurcations are returned
        assert isinstance(bifurcations, list)
        
        # Check that bifurcations have required properties
        for bifurcation in bifurcations:
            assert 'parameter' in bifurcation
            assert 'value' in bifurcation
            assert 'type' in bifurcation
            assert 'stability' in bifurcation
    
    def test_nonlinear_stability_analysis(self, nonlinear):
        """
        Test nonlinear stability analysis.
        
        Physical Meaning:
            Verifies that nonlinear stability is correctly
            analyzed.
        """
        stability = nonlinear._analyze_nonlinear_stability()
        
        # Check that stability is returned
        assert 'is_stable' in stability
        assert 'stability_margin' in stability
        assert 'nonlinear_criteria' in stability
        
        # Check that stability is boolean
        assert isinstance(stability['is_stable'], (bool, np.bool_))
        assert isinstance(stability['nonlinear_criteria'], bool)
    
    def test_stability_check(self, nonlinear):
        """
        Test stability check.
        
        Physical Meaning:
            Verifies that the stability of nonlinear solutions
            is correctly checked.
        """
        stability = nonlinear.check_nonlinear_stability()
        
        # Check that stability is returned
        assert 'linear_stability' in stability
        assert 'nonlinear_stability' in stability
        assert 'growth_rates' in stability
        assert 'stability_regions' in stability
        
        # Check that growth rates are returned
        assert isinstance(stability['growth_rates'], np.ndarray)
        assert len(stability['growth_rates']) > 0
    
    def test_growth_rates_computation(self, nonlinear):
        """
        Test growth rates computation.
        
        Physical Meaning:
            Verifies that growth rates are correctly
            computed for instability analysis.
        """
        growth_rates = nonlinear._compute_growth_rates()
        
        # Check that growth rates are returned
        assert isinstance(growth_rates, np.ndarray)
        assert len(growth_rates) > 0
        
        # Check that growth rates are finite
        assert np.all(np.isfinite(growth_rates))
    
    def test_stability_region_identification(self, nonlinear):
        """
        Test stability region identification.
        
        Physical Meaning:
            Verifies that stability regions are correctly
            identified in parameter space.
        """
        regions = nonlinear._identify_stability_regions()
        
        # Check that regions are returned
        assert 'stable_regions' in regions
        assert 'unstable_regions' in regions
        
        # Check that regions have required properties
        for region_type in ['stable_regions', 'unstable_regions']:
            assert 'nonlinear_strength' in regions[region_type]
            assert 'nonlinear_order' in regions[region_type]
    
    def test_different_nonlinear_types(self, system):
        """
        Test different nonlinear types.
        
        Physical Meaning:
            Verifies that different nonlinear types
            work correctly.
        """
        nonlinear_types = ['cubic', 'quartic', 'sine_gordon']
        
        for nonlinear_type in nonlinear_types:
            params = {'strength': 1.0, 'type': nonlinear_type}
            nonlinear = NonlinearEffects(system, params)
            
            # Check that nonlinear type is set
            assert nonlinear.nonlinear_type == nonlinear_type
            
            # Check that nonlinear terms are set up
            assert hasattr(nonlinear, 'nonlinear_potential')
            assert hasattr(nonlinear, 'nonlinear_derivative')
    
    def test_parameter_dependence(self, system):
        """
        Test dependence on nonlinear parameters.
        
        Physical Meaning:
            Verifies that the system behavior changes
            correctly with nonlinear parameters.
        """
        strengths = [0.5, 1.0, 2.0]
        
        for strength in strengths:
            params = {'strength': strength, 'type': 'cubic'}
            nonlinear = NonlinearEffects(system, params)
            
            # Check that strength is set
            assert nonlinear.nonlinear_strength == strength
            
            # Check that nonlinear terms depend on strength
            psi = np.array([0.1, 0.5, 1.0])
            potential = nonlinear.nonlinear_potential(psi)
            expected = strength * np.abs(psi)**3
            np.testing.assert_allclose(potential, expected)
    
    def test_soliton_analysis(self, nonlinear):
        """
        Test soliton analysis.
        
        Physical Meaning:
            Verifies that soliton properties are correctly
            analyzed.
        """
        solitons = [
            {'type': 'kink', 'velocity': 0.5, 'amplitude': 2.0, 'width': 1.0, 'position': 0.0, 'stability': True},
            {'type': 'antikink', 'velocity': -0.5, 'amplitude': -2.0, 'width': 1.0, 'position': 0.0, 'stability': True}
        ]
        
        analysis = nonlinear._analyze_soliton_properties(solitons)
        
        # Check that analysis is returned
        assert 'profiles' in analysis
        assert 'velocities' in analysis
        assert 'stability' in analysis
        
        # Check that profiles are returned
        assert len(analysis['profiles']) == len(solitons)
        assert len(analysis['velocities']) == len(solitons)
        assert len(analysis['stability']) == len(solitons)
        
        # Check that profiles are finite
        for profile in analysis['profiles']:
            assert np.all(np.isfinite(profile))
    
    def test_error_handling(self, system):
        """
        Test error handling for invalid parameters.
        
        Physical Meaning:
            Verifies that the system handles invalid
            parameters gracefully.
        """
        # Test invalid nonlinear type
        with pytest.raises(ValueError):
            params = {'strength': 1.0, 'type': 'invalid_type'}
            NonlinearEffects(system, params)
    
    def test_linear_stability_analysis(self, nonlinear):
        """
        Test linear stability analysis.
        
        Physical Meaning:
            Verifies that linear stability is correctly
            analyzed.
        """
        stability = nonlinear._analyze_linear_stability()
        
        # Check that stability is returned
        assert 'is_stable' in stability
        assert 'stability_margin' in stability
        assert 'eigenvalues' in stability
        
        # Check that stability is boolean
        assert isinstance(stability['is_stable'], (bool, np.bool_))
        assert isinstance(stability['stability_margin'], (int, float, np.integer, np.floating))
        assert isinstance(stability['eigenvalues'], np.ndarray)
