"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Comprehensive unit tests for domain module.

This module provides comprehensive unit tests for the domain module,
covering all classes and methods to achieve high test coverage.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from bhlff.core.domain import Domain, Field, Parameters
from bhlff.core.domain.domain_7d import Domain7D


class TestDomain:
    """Comprehensive tests for Domain class."""

    @pytest.fixture
    def domain(self):
        """Create domain for testing."""
        return Domain(
            L=1.0,
            N=16,
            dimensions=7,
            N_phi=8,
            N_t=32,
            T=1.0
        )

    def test_domain_initialization(self, domain):
        """Test domain initialization."""
        assert domain.L == 1.0
        assert domain.N == 16
        assert domain.dimensions == 7
        assert domain.N_phi == 8
        assert domain.N_t == 32
        assert domain.T == 1.0

    def test_domain_properties(self, domain):
        """Test domain properties."""
        assert domain.dx == 1.0 / 16
        assert domain.dt == 1.0 / 32
        assert domain.dphi == 2 * np.pi / 8
        assert domain.shape == (16, 16, 16, 8, 8, 8, 32)

    def test_domain_coordinates(self, domain):
        """Test coordinate generation."""
        x = domain.get_coordinates(0)
        assert len(x) == 16
        assert x[0] == -0.5
        assert x[-1] == 0.5 - domain.dx

    def test_domain_phase_coordinates(self, domain):
        """Test phase coordinate generation."""
        phi = domain.get_phase_coordinates(0)
        assert len(phi) == 8
        assert phi[0] == 0.0
        assert phi[-1] == 2 * np.pi - domain.dphi

    def test_domain_time_coordinates(self, domain):
        """Test time coordinate generation."""
        t = domain.get_time_coordinates()
        assert len(t) == 32
        assert t[0] == 0.0
        assert t[-1] == 1.0 - domain.dt

    def test_domain_meshgrid(self, domain):
        """Test meshgrid generation."""
        X, Y, Z = domain.get_meshgrid()
        assert X.shape == (16, 16, 16)
        assert Y.shape == (16, 16, 16)
        assert Z.shape == (16, 16, 16)

    def test_domain_phase_meshgrid(self, domain):
        """Test phase meshgrid generation."""
        PHI1, PHI2, PHI3 = domain.get_phase_meshgrid()
        assert PHI1.shape == (8, 8, 8)
        assert PHI2.shape == (8, 8, 8)
        assert PHI3.shape == (8, 8, 8)

    def test_domain_validation(self):
        """Test domain validation."""
        with pytest.raises(ValueError):
            Domain(L=-1.0, N=16, dimensions=7)
        
        with pytest.raises(ValueError):
            Domain(L=1.0, N=0, dimensions=7)
        
        with pytest.raises(ValueError):
            Domain(L=1.0, N=16, dimensions=0)

    def test_domain_repr(self, domain):
        """Test domain string representation."""
        repr_str = repr(domain)
        assert "Domain" in repr_str
        assert "L=1.0" in repr_str
        assert "N=16" in repr_str


class TestDomain7D:
    """Comprehensive tests for Domain7D class."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain7D(
            L=1.0,
            N=8,
            N_phi=4,
            N_t=8,
            T=1.0,
            dimensions=7
        )

    def test_domain_7d_initialization(self, domain_7d):
        """Test 7D domain initialization."""
        assert domain_7d.L == 1.0
        assert domain_7d.N == 8
        assert domain_7d.N_phi == 4
        assert domain_7d.N_t == 8
        assert domain_7d.T == 1.0
        assert domain_7d.dimensions == 3

    def test_domain_7d_properties(self, domain_7d):
        """Test 7D domain properties."""
        assert domain_7d.dx == 1.0 / 8
        assert domain_7d.dt == 1.0 / 8
        assert domain_7d.dphi == 2 * np.pi / 4
        assert domain_7d.shape == (8, 8, 8, 4, 4, 4, 8)

    def test_domain_7d_coordinates(self, domain_7d):
        """Test 7D coordinate generation."""
        x = domain_7d.get_coordinates(0)
        assert len(x) == 8
        
        phi = domain_7d.get_phase_coordinates(0)
        assert len(phi) == 4
        
        t = domain_7d.get_time_coordinates()
        assert len(t) == 8

    def test_domain_7d_meshgrid(self, domain_7d):
        """Test 7D meshgrid generation."""
        X, Y, Z = domain_7d.get_meshgrid()
        assert X.shape == (8, 8, 8)
        
        PHI1, PHI2, PHI3 = domain_7d.get_phase_meshgrid()
        assert PHI1.shape == (4, 4, 4)

    def test_domain_7d_validation(self):
        """Test 7D domain validation."""
        with pytest.raises(ValueError):
            Domain7D(L=-1.0, N=8, N_phi=4, N_t=8, T=1.0, dimensions=7)
        
        with pytest.raises(ValueError):
            Domain7D(L=1.0, N=0, N_phi=4, N_t=8, T=1.0, dimensions=7)

    def test_domain_7d_repr(self, domain_7d):
        """Test 7D domain string representation."""
        repr_str = repr(domain_7d)
        assert "Domain7D" in repr_str


class TestField:
    """Comprehensive tests for Field class."""

    @pytest.fixture
    def domain(self):
        """Create domain for testing."""
        return Domain(
            L=1.0,
            N=8,
            dimensions=7,
            N_phi=4,
            N_t=8,
            T=1.0
        )

    @pytest.fixture
    def field(self, domain):
        """Create field for testing."""
        return Field(domain)

    def test_field_initialization(self, field, domain):
        """Test field initialization."""
        assert field.domain == domain
        assert field.data.shape == domain.shape
        assert np.all(field.data == 0.0)

    def test_field_set_data(self, field):
        """Test setting field data."""
        test_data = np.ones(field.domain.shape)
        field.set_data(test_data)
        assert np.all(field.data == test_data)

    def test_field_get_data(self, field):
        """Test getting field data."""
        test_data = np.random.random(field.domain.shape)
        field.set_data(test_data)
        retrieved_data = field.get_data()
        assert np.all(retrieved_data == test_data)

    def test_field_energy(self, field):
        """Test field energy calculation."""
        # Set test data
        test_data = np.ones(field.domain.shape)
        field.set_data(test_data)
        
        energy = field.get_energy()
        expected_energy = np.sum(test_data**2) * field.domain.dx**3 * field.domain.dphi**3 * field.domain.dt
        assert abs(energy - expected_energy) < 1e-10

    def test_field_norm(self, field):
        """Test field norm calculation."""
        test_data = np.ones(field.domain.shape)
        field.set_data(test_data)
        
        norm = field.get_norm()
        expected_norm = np.sqrt(np.sum(test_data**2) * field.domain.dx**3 * field.domain.dphi**3 * field.domain.dt)
        assert abs(norm - expected_norm) < 1e-10

    def test_field_gradient(self, field):
        """Test field gradient calculation."""
        # Create test field with known gradient
        x = field.domain.get_coordinates(0)
        y = field.domain.get_coordinates(1)
        z = field.domain.get_coordinates(2)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        test_data = X + Y + Z
        
        # Broadcast to full 7D shape
        test_data_7d = test_data[:, :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        field.set_data(test_data_7d)
        
        gradient = field.get_gradient()
        assert gradient.shape == (3, 8, 8, 8, 4, 4, 4, 8)
        assert np.allclose(gradient[0], 1.0, atol=1e-10)
        assert np.allclose(gradient[1], 1.0, atol=1e-10)
        assert np.allclose(gradient[2], 1.0, atol=1e-10)

    def test_field_laplacian(self, field):
        """Test field Laplacian calculation."""
        # Create test field with known Laplacian
        x = field.domain.get_coordinates(0)
        y = field.domain.get_coordinates(1)
        z = field.domain.get_coordinates(2)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        test_data = X**2 + Y**2 + Z**2
        
        # Broadcast to full 7D shape
        test_data_7d = test_data[:, :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        field.set_data(test_data_7d)
        
        laplacian = field.get_laplacian()
        assert laplacian.shape == (8, 8, 8, 4, 4, 4, 8)
        # Laplacian of x² + y² + z² = 6
        assert np.allclose(laplacian, 6.0, atol=1e-10)

    def test_field_validation(self, field):
        """Test field validation."""
        with pytest.raises(ValueError):
            field.set_data(np.zeros((4, 4, 4)))  # Wrong shape

    def test_field_repr(self, field):
        """Test field string representation."""
        repr_str = repr(field)
        assert "Field" in repr_str


class TestParameters:
    """Comprehensive tests for Parameters class."""

    @pytest.fixture
    def parameters(self):
        """Create parameters for testing."""
        return Parameters(
            mu=1.0,
            beta=1.5,
            lambda_param=0.1,
            nu=1.0
        )

    def test_parameters_initialization(self, parameters):
        """Test parameters initialization."""
        assert parameters.mu == 1.0
        assert parameters.beta == 1.5
        assert parameters.lambda_param == 0.1
        assert parameters.nu == 1.0

    def test_parameters_validation(self):
        """Test parameters validation."""
        with pytest.raises(ValueError):
            Parameters(mu=-1.0, beta=1.5, lambda_param=0.1, nu=1.0)
        
        with pytest.raises(ValueError):
            Parameters(mu=1.0, beta=0.0, lambda_param=0.1, nu=1.0)
        
        with pytest.raises(ValueError):
            Parameters(mu=1.0, beta=2.0, lambda_param=0.1, nu=1.0)

    def test_parameters_get_spectral_coefficients(self, parameters):
        """Test spectral coefficients calculation."""
        k_magnitude = np.array([0.0, 1.0, 2.0])
        coeffs = parameters.get_spectral_coefficients(k_magnitude)
        
        expected = parameters.mu * (k_magnitude ** (2 * parameters.beta)) + parameters.lambda_param
        assert np.allclose(coeffs, expected)

    def test_parameters_repr(self, parameters):
        """Test parameters string representation."""
        repr_str = repr(parameters)
        assert "Parameters" in repr_str
        assert "mu=1.0" in repr_str


