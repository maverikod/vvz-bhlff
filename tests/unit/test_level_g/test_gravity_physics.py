"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical tests for gravitational effects in Level G experiments.

This module contains comprehensive physical tests for gravitational
effects models, verifying their behavior against theoretical predictions
and physical constraints.

Theoretical Background:
    Tests verify the physical correctness of gravitational calculations
    including Einstein equations, spacetime curvature, and gravitational
    waves.

Example:
    >>> pytest tests/unit/test_level_g/test_gravity_physics.py -v
"""

import pytest
import numpy as np
from bhlff.core.domain.domain import Domain
from bhlff.models.level_g.gravity import GravitationalEffectsModel
from bhlff.models.level_g.gravity_curvature import SpacetimeCurvatureCalculator
from bhlff.models.level_g.gravity_einstein import EinsteinEquationsSolver
from bhlff.models.level_g.gravity_waves import GravitationalWavesCalculator


class TestGravityPhysics:
    """
    Physical tests for gravitational effects models.
    
    Tests verify the physical correctness of gravitational calculations
    against known theoretical results and physical constraints.
    """

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain(
            L=4.0,  # Domain size
            N=64,   # Grid points
            dimensions=7
        )

    @pytest.fixture
    def gravity_params(self):
        """Create realistic gravity parameters."""
        return {
            "G": 6.67430e-11,  # Gravitational constant
            "c": 299792458.0,   # Speed of light
            "phase_gravity_coupling": 1.0,
            "resolution": 64,
            "domain_size": 4.0,
            "precision": 1e-12,
            "tolerance": 1e-12,
            "max_iterations": 1000,
            "field_mass_squared": 1.0,
            "update_factor": 0.01,
            "frequency_range": (1e-4, 1e3),
            "detection_sensitivity": 1e-21,
            "wave_speed": 299792458.0,
            "source_distance": 1e6
        }

    @pytest.fixture
    def mock_system(self, domain_7d):
        """Create mock system for testing."""
        class MockSystem:
            def __init__(self, domain):
                self.domain = domain
                self.phase_field = np.ones((64, 64, 64), dtype=complex)
        
        return MockSystem(domain_7d)

    def test_curvature_tensor_calculation(self, domain_7d, gravity_params):
        """
        Test Riemann tensor calculation.
        
        Physical Meaning:
            Verifies that the Riemann tensor is correctly calculated
            with proper symmetry properties and physical constraints.
        """
        curvature_calc = SpacetimeCurvatureCalculator(domain_7d, gravity_params)
        
        # Create test metric (Minkowski metric with small perturbation)
        metric = np.eye(4)
        metric[0, 0] = -1  # Time component
        metric += 0.01 * np.random.randn(4, 4)  # Small perturbation
        metric = 0.5 * (metric + metric.T)  # Ensure symmetry
        
        # Compute Riemann tensor
        riemann_tensor = curvature_calc.compute_riemann_tensor(metric)
        
        # Check tensor properties
        assert riemann_tensor.shape == (4, 4, 4, 4), "Riemann tensor should have correct shape"
        assert np.all(np.isfinite(riemann_tensor)), "Riemann tensor should be finite"
        
        # Check symmetry properties
        for mu in range(4):
            for nu in range(4):
                for rho in range(4):
                    for sigma in range(4):
                        # Antisymmetry in first two indices
                        assert abs(riemann_tensor[mu, nu, rho, sigma] + 
                                 riemann_tensor[nu, mu, rho, sigma]) < 1e-10, \
                               "Riemann tensor should be antisymmetric in first two indices"

    def test_ricci_tensor_calculation(self, domain_7d, gravity_params):
        """
        Test Ricci tensor calculation.
        
        Physical Meaning:
            Verifies that the Ricci tensor is correctly calculated
            from the Riemann tensor.
        """
        curvature_calc = SpacetimeCurvatureCalculator(domain_7d, gravity_params)
        
        # Create test metric
        metric = np.eye(4)
        metric[0, 0] = -1
        
        # Compute Riemann tensor
        riemann_tensor = curvature_calc.compute_riemann_tensor(metric)
        
        # Compute Ricci tensor
        ricci_tensor = curvature_calc.compute_ricci_tensor(riemann_tensor)
        
        # Check tensor properties
        assert ricci_tensor.shape == (4, 4), "Ricci tensor should have correct shape"
        assert np.all(np.isfinite(ricci_tensor)), "Ricci tensor should be finite"
        
        # Check symmetry
        assert np.allclose(ricci_tensor, ricci_tensor.T), "Ricci tensor should be symmetric"

    def test_scalar_curvature_calculation(self, domain_7d, gravity_params):
        """
        Test scalar curvature calculation.
        
        Physical Meaning:
            Verifies that the scalar curvature is correctly calculated
            from the Ricci tensor and metric.
        """
        curvature_calc = SpacetimeCurvatureCalculator(domain_7d, gravity_params)
        
        # Create test metric
        metric = np.eye(4)
        metric[0, 0] = -1
        
        # Compute Riemann and Ricci tensors
        riemann_tensor = curvature_calc.compute_riemann_tensor(metric)
        ricci_tensor = curvature_calc.compute_ricci_tensor(riemann_tensor)
        
        # Compute scalar curvature
        scalar_curvature = curvature_calc.compute_scalar_curvature(ricci_tensor, metric)
        
        # Check scalar properties
        assert isinstance(scalar_curvature, (int, float)), "Scalar curvature should be scalar"
        assert np.isfinite(scalar_curvature), "Scalar curvature should be finite"

    def test_einstein_equations_solution(self, domain_7d, gravity_params):
        """
        Test Einstein equations solution.
        
        Physical Meaning:
            Verifies that the Einstein equations are correctly solved
            with phase field sources.
        """
        einstein_solver = EinsteinEquationsSolver(domain_7d, gravity_params)
        
        # Create test phase field
        phase_field = np.ones((64, 64, 64), dtype=complex)
        
        # Solve Einstein equations
        metric = einstein_solver.solve_einstein_equations(phase_field)
        
        # Check metric properties
        assert metric.shape == (4, 4), "Metric should have correct shape"
        assert np.all(np.isfinite(metric)), "Metric should be finite"
        assert np.allclose(metric, metric.T), "Metric should be symmetric"
        
        # Check metric signature (Lorentzian)
        eigenvalues = np.linalg.eigvals(metric)
        negative_eigenvalues = np.sum(eigenvalues < 0)
        assert negative_eigenvalues == 1, "Metric should have Lorentzian signature"

    def test_energy_momentum_tensor(self, domain_7d, gravity_params):
        """
        Test energy-momentum tensor calculation.
        
        Physical Meaning:
            Verifies that the energy-momentum tensor is correctly
            calculated from the phase field.
        """
        einstein_solver = EinsteinEquationsSolver(domain_7d, gravity_params)
        
        # Create test phase field
        phase_field = np.ones((64, 64, 64), dtype=complex)
        
        # Compute energy-momentum tensor
        T_mu_nu = einstein_solver._compute_energy_momentum_tensor(phase_field)
        
        # Check tensor properties
        assert T_mu_nu.shape == (4, 4), "Energy-momentum tensor should have correct shape"
        assert np.all(np.isfinite(T_mu_nu)), "Energy-momentum tensor should be finite"
        assert np.allclose(T_mu_nu, T_mu_nu.T), "Energy-momentum tensor should be symmetric"

    def test_gravitational_waves_calculation(self, domain_7d, gravity_params):
        """
        Test gravitational waves calculation.
        
        Physical Meaning:
            Verifies that gravitational waves are correctly calculated
            from the spacetime metric.
        """
        waves_calc = GravitationalWavesCalculator(domain_7d, gravity_params)
        
        # Create test metric
        metric = np.eye(4)
        metric[0, 0] = -1
        metric += 0.01 * np.random.randn(4, 4)  # Small perturbation
        metric = 0.5 * (metric + metric.T)
        
        # Compute gravitational waves
        waves = waves_calc.compute_gravitational_waves(metric)
        
        # Check wave properties
        assert "strain_tensor" in waves, "Should contain strain tensor"
        assert "amplitude" in waves, "Should contain amplitude"
        assert "frequency_spectrum" in waves, "Should contain frequency spectrum"
        assert "polarization" in waves, "Should contain polarization"
        
        # Check strain tensor
        strain_tensor = waves["strain_tensor"]
        assert strain_tensor.shape == (4, 4), "Strain tensor should have correct shape"
        assert np.all(np.isfinite(strain_tensor)), "Strain tensor should be finite"
        
        # Check amplitude
        amplitude = waves["amplitude"]
        assert isinstance(amplitude, (int, float)), "Amplitude should be scalar"
        assert amplitude >= 0, "Amplitude should be non-negative"

    def test_polarization_modes(self, domain_7d, gravity_params):
        """
        Test gravitational wave polarization modes.
        
        Physical Meaning:
            Verifies that polarization modes are correctly calculated
            from the strain tensor.
        """
        waves_calc = GravitationalWavesCalculator(domain_7d, gravity_params)
        
        # Create test strain tensor
        strain_tensor = np.random.randn(4, 4)
        strain_tensor = 0.5 * (strain_tensor + strain_tensor.T)  # Symmetric
        
        # Compute polarization modes
        polarization = waves_calc._compute_polarization_modes(strain_tensor)
        
        # Check polarization modes
        assert "plus" in polarization, "Should contain plus polarization"
        assert "cross" in polarization, "Should contain cross polarization"
        assert "x_mode" in polarization, "Should contain x-mode polarization"
        assert "y_mode" in polarization, "Should contain y-mode polarization"
        
        # Check that all modes are finite
        for mode_name, mode_value in polarization.items():
            assert np.isfinite(mode_value), f"{mode_name} mode should be finite"

    def test_gravitational_effects_integration(self, mock_system, gravity_params):
        """
        Test integration of all gravitational effects.
        
        Physical Meaning:
            Verifies that all gravitational effects work together
            correctly in the main model.
        """
        gravity_model = GravitationalEffectsModel(mock_system, gravity_params)
        
        # Compute all gravitational effects
        effects = gravity_model.compute_gravitational_effects()
        
        # Check that all effects are present
        assert "curvature" in effects, "Should contain curvature analysis"
        assert "gravitational_waves" in effects, "Should contain gravitational waves"
        assert "einstein_effects" in effects, "Should contain Einstein effects"
        
        # Check curvature analysis
        curvature = effects["curvature"]
        assert "riemann_tensor" in curvature, "Should contain Riemann tensor"
        assert "ricci_tensor" in curvature, "Should contain Ricci tensor"
        assert "scalar_curvature" in curvature, "Should contain scalar curvature"
        assert "weyl_tensor" in curvature, "Should contain Weyl tensor"
        assert "curvature_invariants" in curvature, "Should contain curvature invariants"
        
        # Check gravitational waves
        waves = effects["gravitational_waves"]
        assert "strain_tensor" in waves, "Should contain strain tensor"
        assert "amplitude" in waves, "Should contain amplitude"
        assert "frequency_spectrum" in waves, "Should contain frequency spectrum"
        assert "polarization" in waves, "Should contain polarization"

    def test_curvature_invariants(self, domain_7d, gravity_params):
        """
        Test curvature invariants calculation.
        
        Physical Meaning:
            Verifies that curvature invariants are correctly calculated
            and are coordinate-independent.
        """
        curvature_calc = SpacetimeCurvatureCalculator(domain_7d, gravity_params)
        
        # Create test metric
        metric = np.eye(4)
        metric[0, 0] = -1
        metric += 0.01 * np.random.randn(4, 4)
        metric = 0.5 * (metric + metric.T)
        
        # Compute curvature tensors
        riemann_tensor = curvature_calc.compute_riemann_tensor(metric)
        ricci_tensor = curvature_calc.compute_ricci_tensor(riemann_tensor)
        scalar_curvature = curvature_calc.compute_scalar_curvature(ricci_tensor, metric)
        
        # Compute invariants
        invariants = curvature_calc.compute_curvature_invariants(
            riemann_tensor, ricci_tensor, scalar_curvature
        )
        
        # Check invariants
        assert "kretschmann" in invariants, "Should contain Kretschmann scalar"
        assert "ricci_squared" in invariants, "Should contain Ricci squared"
        assert "ricci_tensor_squared" in invariants, "Should contain Ricci tensor squared"
        assert "scalar_curvature" in invariants, "Should contain scalar curvature"
        
        # Check that invariants are finite
        for name, value in invariants.items():
            assert np.isfinite(value), f"Invariant {name} should be finite"

    def test_weyl_tensor_calculation(self, domain_7d, gravity_params):
        """
        Test Weyl tensor calculation.
        
        Physical Meaning:
            Verifies that the Weyl tensor is correctly calculated
            and represents the conformal curvature.
        """
        curvature_calc = SpacetimeCurvatureCalculator(domain_7d, gravity_params)
        
        # Create test metric
        metric = np.eye(4)
        metric[0, 0] = -1
        metric += 0.01 * np.random.randn(4, 4)
        metric = 0.5 * (metric + metric.T)
        
        # Compute curvature tensors
        riemann_tensor = curvature_calc.compute_riemann_tensor(metric)
        ricci_tensor = curvature_calc.compute_ricci_tensor(riemann_tensor)
        scalar_curvature = curvature_calc.compute_scalar_curvature(ricci_tensor, metric)
        
        # Compute Weyl tensor
        weyl_tensor = curvature_calc.compute_weyl_tensor(
            riemann_tensor, ricci_tensor, scalar_curvature, metric
        )
        
        # Check Weyl tensor properties
        assert weyl_tensor.shape == (4, 4, 4, 4), "Weyl tensor should have correct shape"
        assert np.all(np.isfinite(weyl_tensor)), "Weyl tensor should be finite"
        
        # Check symmetry properties
        for mu in range(4):
            for nu in range(4):
                for rho in range(4):
                    for sigma in range(4):
                        # Antisymmetry in first two indices
                        assert abs(weyl_tensor[mu, nu, rho, sigma] + 
                                 weyl_tensor[nu, mu, rho, sigma]) < 1e-10, \
                               "Weyl tensor should be antisymmetric in first two indices"
