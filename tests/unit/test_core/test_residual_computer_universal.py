"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for universal ResidualComputer class.

This module tests the universal ResidualComputer class that combines
functionality from both original ResidualComputer implementations,
ensuring backward compatibility and correct behavior for both
domain types and configurations.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from bhlff.core.bvp.residual_computer import ResidualComputer
from bhlff.core.bvp.residual_computer_base import ResidualComputerBase
from bhlff.core.domain import Domain
from bhlff.core.domain.domain_7d import Domain7D
from bhlff.core.bvp.bvp_constants import BVPConstants


class TestResidualComputerUniversal:
    """Test universal ResidualComputer class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Standard domain
        self.domain = Domain(L=1.0, N=32, dimensions=3)
        
        # 7D domain
        self.domain_7d = Domain7D(
            L_spatial=1.0, N_spatial=32, N_phase=16, T=1.0, N_t=64
        )
        
        # BVP constants
        self.constants = BVPConstants()
        
        # Config dict
        self.config = {
            "kappa_0": 1.0,
            "kappa_2": 0.1,
            "chi_prime": 1.0,
            "chi_double_prime_0": 0.1,
            "k0_squared": 1.0
        }
        
        # Test fields
        self.envelope_3d = np.random.rand(32, 32, 32) + 1j * np.random.rand(32, 32, 32)
        self.source_3d = np.random.rand(32, 32, 32) + 1j * np.random.rand(32, 32, 32)
        
        self.envelope_7d = np.random.rand(32, 32, 32, 16, 16, 16, 64) + \
                          1j * np.random.rand(32, 32, 32, 16, 16, 16, 64)
        self.source_7d = np.random.rand(32, 32, 32, 16, 16, 16, 64) + \
                        1j * np.random.rand(32, 32, 32, 16, 16, 16, 64)
    
    def test_inheritance(self):
        """Test that ResidualComputer inherits from ResidualComputerBase."""
        computer = ResidualComputer(self.domain, self.constants)
        assert isinstance(computer, ResidualComputerBase)
    
    def test_domain_type_detection_standard(self):
        """Test domain type detection for standard domain."""
        computer = ResidualComputer(self.domain, self.constants)
        assert computer.domain_type == "standard"
    
    def test_domain_type_detection_7d(self):
        """Test domain type detection for 7D domain."""
        computer = ResidualComputer(self.domain_7d, self.config)
        assert computer.domain_type == "7d_bvp"
    
    def test_standard_approach_initialization(self):
        """Test initialization with standard approach (Domain + BVPConstants)."""
        computer = ResidualComputer(self.domain, self.constants)
        
        assert computer.domain == self.domain
        assert computer.config_or_constants == self.constants
        assert computer.domain_type == "standard"
        
        # Check that parameters are set up
        assert hasattr(computer, 'kappa_0')
        assert hasattr(computer, 'kappa_2')
        assert hasattr(computer, 'chi_prime')
        assert hasattr(computer, 'chi_double_prime_0')
        assert hasattr(computer, 'k0_squared')
    
    def test_7d_approach_initialization(self):
        """Test initialization with 7D approach (Domain7D + config)."""
        computer = ResidualComputer(self.domain_7d, self.config)
        
        assert computer.domain == self.domain_7d
        assert computer.config_or_constants == self.config
        assert computer.domain_type == "7d_bvp"
        
        # Check that parameters are None (handled by external objects)
        assert computer.kappa_0 is None
        assert computer.kappa_2 is None
        assert computer.chi_prime is None
        assert computer.chi_double_prime_0 is None
        assert computer.k0_squared is None
    
    def test_standard_approach_residual_computation(self):
        """Test residual computation with standard approach."""
        computer = ResidualComputer(self.domain, self.constants)
        
        # Mock the domain attributes needed for computation
        computer.domain.dx = 0.03125  # 1.0 / 32
        computer.domain.dphi = 0.392699  # 2*pi / 16
        computer.domain.dt = 0.015625  # 1.0 / 64
        
        residual = computer.compute_residual(self.envelope_3d, self.source_3d)
        
        assert residual.shape == self.envelope_3d.shape
        assert np.isfinite(residual).all()
        assert residual.dtype == complex
    
    def test_7d_approach_residual_computation(self):
        """Test residual computation with 7D approach."""
        computer = ResidualComputer(self.domain_7d, self.config)
        
        # Mock derivative operators
        derivative_operators = Mock()
        derivative_operators.apply_spatial_gradient.return_value = np.zeros_like(self.envelope_7d)
        derivative_operators.apply_spatial_divergence.return_value = np.zeros_like(self.envelope_7d)
        derivative_operators.apply_phase_gradient.return_value = np.zeros_like(self.envelope_7d)
        derivative_operators.apply_phase_divergence.return_value = np.zeros_like(self.envelope_7d)
        
        # Mock nonlinear terms
        nonlinear_terms = Mock()
        nonlinear_terms.compute_stiffness.return_value = np.ones_like(self.envelope_7d)
        nonlinear_terms.compute_susceptibility.return_value = np.ones_like(self.envelope_7d)
        nonlinear_terms.k0 = 1.0
        
        residual = computer.compute_residual(
            self.envelope_7d, self.source_7d, 
            derivative_operators, nonlinear_terms
        )
        
        assert residual.shape == self.envelope_7d.shape
        assert np.isfinite(residual).all()
        assert residual.dtype == complex
    
    def test_residual_norm_computation(self):
        """Test residual norm computation."""
        computer = ResidualComputer(self.domain, self.constants)
        
        test_residual = np.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j])
        expected_norm = np.sqrt(1**2 + 2**2 + 3**2 + 4**2 + 5**2 + 6**2)
        
        norm = computer.compute_residual_norm(test_residual)
        
        assert abs(norm - expected_norm) < 1e-10
    
    def test_standard_approach_component_analysis(self):
        """Test component analysis with standard approach."""
        computer = ResidualComputer(self.domain, self.constants)
        
        # Mock the domain attributes
        computer.domain.dx = 0.03125
        computer.domain.dphi = 0.392699
        computer.domain.dt = 0.015625
        
        analysis = computer.analyze_residual_components(self.envelope_3d, self.source_3d)
        
        assert "divergence_norm" in analysis
        assert "susceptibility_norm" in analysis
        assert "source_norm" in analysis
        assert "total_residual_norm" in analysis
        assert "component_ratios" in analysis
        
        # Check that all norms are non-negative
        assert analysis["divergence_norm"] >= 0
        assert analysis["susceptibility_norm"] >= 0
        assert analysis["source_norm"] >= 0
        assert analysis["total_residual_norm"] >= 0
    
    def test_7d_approach_component_analysis(self):
        """Test component analysis with 7D approach."""
        computer = ResidualComputer(self.domain_7d, self.config)
        
        # Mock derivative operators
        derivative_operators = Mock()
        derivative_operators.apply_spatial_gradient.return_value = np.zeros_like(self.envelope_7d)
        derivative_operators.apply_spatial_divergence.return_value = np.zeros_like(self.envelope_7d)
        derivative_operators.apply_phase_gradient.return_value = np.zeros_like(self.envelope_7d)
        derivative_operators.apply_phase_divergence.return_value = np.zeros_like(self.envelope_7d)
        
        # Mock nonlinear terms
        nonlinear_terms = Mock()
        nonlinear_terms.compute_stiffness.return_value = np.ones_like(self.envelope_7d)
        nonlinear_terms.compute_susceptibility.return_value = np.ones_like(self.envelope_7d)
        nonlinear_terms.k0 = 1.0
        
        analysis = computer.analyze_residual_components(
            self.envelope_7d, self.source_7d,
            derivative_operators, nonlinear_terms
        )
        
        assert "spatial_divergence_norm" in analysis
        assert "phase_divergence_norm" in analysis
        assert "susceptibility_norm" in analysis
        assert "source_norm" in analysis
        assert "total_residual_norm" in analysis
        assert "component_ratios" in analysis
        
        # Check that all norms are non-negative
        assert analysis["spatial_divergence_norm"] >= 0
        assert analysis["phase_divergence_norm"] >= 0
        assert analysis["susceptibility_norm"] >= 0
        assert analysis["source_norm"] >= 0
        assert analysis["total_residual_norm"] >= 0
    
    def test_repr(self):
        """Test string representation."""
        computer = ResidualComputer(self.domain, self.constants)
        repr_str = repr(computer)
        
        assert "ResidualComputer" in repr_str
        assert "domain=" in repr_str
    
    def test_config_dict_fallback(self):
        """Test initialization with config dict for standard domain."""
        computer = ResidualComputer(self.domain, self.config)
        
        assert computer.domain_type == "standard"
        assert computer.kappa_0 == 1.0
        assert computer.kappa_2 == 0.1
        assert computer.chi_prime == 1.0
        assert computer.chi_double_prime_0 == 0.1
        assert computer.k0_squared == 1.0
    
    def test_error_handling_invalid_domain(self):
        """Test error handling for invalid domain."""
        with pytest.raises(AttributeError):
            # This should fail because the domain doesn't have required attributes
            computer = ResidualComputer(self.domain, self.constants)
            computer.compute_residual(self.envelope_3d, self.source_3d)
    
    def test_physical_meaning_preservation(self):
        """Test that physical meaning is preserved in both approaches."""
        # Standard approach
        computer_std = ResidualComputer(self.domain, self.constants)
        assert computer_std.domain_type == "standard"
        
        # 7D approach
        computer_7d = ResidualComputer(self.domain_7d, self.config)
        assert computer_7d.domain_type == "7d_bvp"
        
        # Both should be ResidualComputer instances
        assert isinstance(computer_std, ResidualComputer)
        assert isinstance(computer_7d, ResidualComputer)
        
        # Both should inherit from base class
        assert isinstance(computer_std, ResidualComputerBase)
        assert isinstance(computer_7d, ResidualComputerBase)
