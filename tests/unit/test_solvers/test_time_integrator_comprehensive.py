"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Comprehensive unit tests for time integrator module.

This module provides comprehensive unit tests for the time integrator module,
covering all classes and methods to achieve high test coverage.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from bhlff.core.domain import Domain
from bhlff.solvers.integrators.time_integrator import TimeIntegrator


class ConcreteTimeIntegrator(TimeIntegrator):
    """Concrete implementation of TimeIntegrator for testing."""
    
    def __init__(self, domain: Domain, config: dict, bvp_core=None):
        """Initialize concrete time integrator."""
        super().__init__(domain, config, bvp_core)
    
    def step(self, field: np.ndarray, dt: float) -> np.ndarray:
        """Perform one time step."""
        # Simple step: add dt to field
        return field + dt
    
    def get_integrator_type(self) -> str:
        """Get integrator type."""
        return "concrete_test_integrator"


class TestTimeIntegrator:
    """Comprehensive tests for TimeIntegrator class."""

    @pytest.fixture
    def domain(self):
        """Create domain for testing."""
        return Domain(
            L=1.0,
            N=8,
            dimensions=3,
            N_phi=4,
            N_t=8,
            T=1.0
        )

    @pytest.fixture
    def config(self):
        """Create config for testing."""
        return {
            "integrator_type": "test",
            "precision": "float64",
            "quench_detection": {
                "amplitude_threshold": 0.8,
                "detuning_threshold": 0.1,
                "gradient_threshold": 0.5
            }
        }

    @pytest.fixture
    def mock_bvp_core(self):
        """Create mock BVP core."""
        bvp_core = Mock()
        bvp_core.detect_quenches.return_value = {
            "quench_locations": [],
            "quench_types": [],
            "energy_dumped": []
        }
        return bvp_core

    @pytest.fixture
    def integrator(self, domain, config):
        """Create time integrator for testing."""
        return ConcreteTimeIntegrator(domain, config)

    @pytest.fixture
    def integrator_with_bvp(self, domain, config, mock_bvp_core):
        """Create time integrator with BVP core for testing."""
        return ConcreteTimeIntegrator(domain, config, mock_bvp_core)

    def test_integrator_initialization(self, integrator, domain, config):
        """Test integrator initialization."""
        assert integrator.domain == domain
        assert integrator.config == config
        assert integrator.bvp_core is None
        assert integrator.quench_detector is None

    def test_integrator_initialization_with_bvp(self, integrator_with_bvp, domain, config, mock_bvp_core):
        """Test integrator initialization with BVP core."""
        assert integrator_with_bvp.domain == domain
        assert integrator_with_bvp.config == config
        assert integrator_with_bvp.bvp_core == mock_bvp_core
        assert integrator_with_bvp.quench_detector is not None

    def test_integrator_step(self, integrator):
        """Test integrator step method."""
        # Create test field
        field = np.random.random(integrator.domain.shape)
        dt = 0.1
        
        # Perform step
        new_field = integrator.step(field, dt)
        
        assert new_field.shape == field.shape
        assert isinstance(new_field, np.ndarray)
        assert np.allclose(new_field, field + dt)

    def test_integrator_get_integrator_type(self, integrator):
        """Test integrator type retrieval."""
        integrator_type = integrator.get_integrator_type()
        assert integrator_type == "concrete_test_integrator"

    def test_integrator_get_domain(self, integrator, domain):
        """Test domain retrieval."""
        retrieved_domain = integrator.get_domain()
        assert retrieved_domain == domain

    def test_integrator_get_config(self, integrator, config):
        """Test config retrieval."""
        retrieved_config = integrator.get_config()
        assert retrieved_config == config.copy()

    def test_integrator_detect_quenches_with_detector(self, integrator_with_bvp):
        """Test quench detection with detector."""
        # Create test envelope
        envelope = np.random.random(integrator_with_bvp.domain.shape)
        
        # Detect quenches
        quenches = integrator_with_bvp.detect_quenches(envelope)
        
        assert isinstance(quenches, dict)
        assert "quench_locations" in quenches
        assert "quench_types" in quenches
        assert "energy_dumped" in quenches

    def test_integrator_detect_quenches_without_detector(self, integrator):
        """Test quench detection without detector."""
        # Create test envelope
        envelope = np.random.random(integrator.domain.shape)
        
        # Detect quenches
        quenches = integrator.detect_quenches(envelope)
        
        assert isinstance(quenches, dict)
        assert quenches["quench_locations"] == []
        assert quenches["quench_types"] == []
        assert quenches["energy_dumped"] == []

    def test_integrator_get_bvp_core(self, integrator_with_bvp, mock_bvp_core):
        """Test BVP core retrieval."""
        retrieved_bvp_core = integrator_with_bvp.get_bvp_core()
        assert retrieved_bvp_core == mock_bvp_core

    def test_integrator_get_bvp_core_none(self, integrator):
        """Test BVP core retrieval when None."""
        retrieved_bvp_core = integrator.get_bvp_core()
        assert retrieved_bvp_core is None

    def test_integrator_set_bvp_core(self, integrator, mock_bvp_core):
        """Test BVP core setting."""
        # Initially no BVP core
        assert integrator.bvp_core is None
        assert integrator.quench_detector is None
        
        # Set BVP core
        integrator.set_bvp_core(mock_bvp_core)
        
        assert integrator.bvp_core == mock_bvp_core
        assert integrator.quench_detector is not None

    def test_integrator_set_bvp_core_none(self, integrator_with_bvp):
        """Test BVP core setting to None."""
        # Initially has BVP core
        assert integrator_with_bvp.bvp_core is not None
        assert integrator_with_bvp.quench_detector is not None
        
        # Set BVP core to None
        integrator_with_bvp.set_bvp_core(None)
        
        assert integrator_with_bvp.bvp_core is None
        assert integrator_with_bvp.quench_detector is None

    def test_integrator_repr(self, integrator):
        """Test integrator string representation."""
        repr_str = repr(integrator)
        assert "ConcreteTimeIntegrator" in repr_str
        assert "domain=" in repr_str
        assert "type=" in repr_str

    def test_integrator_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        domain = Domain(L=1.0, N=8, dimensions=3)
        config = {"integrator_type": "test"}
        
        # Create abstract integrator (should not be possible, but test the methods)
        integrator = TimeIntegrator(domain, config)
        
        with pytest.raises(NotImplementedError):
            integrator.step(np.random.random(domain.shape), 0.1)
        
        with pytest.raises(NotImplementedError):
            integrator.get_integrator_type()

    def test_integrator_step_physics(self, integrator):
        """Test integrator step physics."""
        # Create test field
        field = np.random.random(integrator.domain.shape)
        dt = 0.1
        
        # Perform multiple steps
        current_field = field.copy()
        for _ in range(5):
            current_field = integrator.step(current_field, dt)
        
        # Should be field + 5 * dt
        expected_field = field + 5 * dt
        assert np.allclose(current_field, expected_field)

    def test_integrator_quench_detection_physics(self, integrator_with_bvp):
        """Test quench detection physics."""
        # Create test envelope with high amplitude (should trigger quench)
        envelope = np.full(integrator_with_bvp.domain.shape, 1.0)
        
        # Detect quenches
        quenches = integrator_with_bvp.detect_quenches(envelope)
        
        # Should return valid quench data structure
        assert isinstance(quenches, dict)
        assert all(key in quenches for key in ["quench_locations", "quench_types", "energy_dumped"])

    def test_integrator_config_handling(self, domain):
        """Test config handling."""
        # Test with minimal config
        minimal_config = {"integrator_type": "test"}
        integrator = ConcreteTimeIntegrator(domain, minimal_config)
        
        assert integrator.config == minimal_config
        
        # Test with extended config
        extended_config = {
            "integrator_type": "test",
            "precision": "float64",
            "tolerance": 1e-12,
            "max_iterations": 1000
        }
        integrator = ConcreteTimeIntegrator(domain, extended_config)
        
        assert integrator.config == extended_config

    def test_integrator_domain_properties(self, integrator, domain):
        """Test domain properties access."""
        assert integrator.domain.L == domain.L
        assert integrator.domain.N == domain.N
        assert integrator.domain.dimensions == domain.dimensions
        assert integrator.domain.shape == domain.shape

    def test_integrator_error_handling(self, integrator):
        """Test error handling."""
        # Test with invalid field shape
        invalid_field = np.random.random((4, 4, 4))
        dt = 0.1
        
        # Should not raise error (integrator doesn't validate input)
        result = integrator.step(invalid_field, dt)
        assert isinstance(result, np.ndarray)

    def test_integrator_numerical_stability(self, integrator):
        """Test numerical stability."""
        # Test with very small dt
        field = np.random.random(integrator.domain.shape)
        small_dt = 1e-15
        
        result = integrator.step(field, small_dt)
        assert isinstance(result, np.ndarray)
        assert result.shape == field.shape

    def test_integrator_large_dt(self, integrator):
        """Test with large dt."""
        # Test with large dt
        field = np.random.random(integrator.domain.shape)
        large_dt = 1e6
        
        result = integrator.step(field, large_dt)
        assert isinstance(result, np.ndarray)
        assert result.shape == field.shape

    def test_integrator_quench_detector_initialization(self, domain, config):
        """Test quench detector initialization."""
        # Test with quench detection config
        quench_config = {
            "integrator_type": "test",
            "quench_detection": {
                "amplitude_threshold": 0.9,
                "detuning_threshold": 0.2,
                "gradient_threshold": 0.7
            }
        }
        
        integrator = ConcreteTimeIntegrator(domain, quench_config)
        
        # Should have quench detector
        assert integrator.quench_detector is not None

    def test_integrator_quench_detector_without_config(self, domain):
        """Test quench detector without config."""
        # Test without quench detection config
        config = {"integrator_type": "test"}
        
        integrator = ConcreteTimeIntegrator(domain, config)
        
        # Should not have quench detector
        assert integrator.quench_detector is None

    def test_integrator_bvp_core_integration(self, domain, config, mock_bvp_core):
        """Test BVP core integration."""
        # Create integrator with BVP core
        integrator = ConcreteTimeIntegrator(domain, config, mock_bvp_core)
        
        # Test that BVP core is properly integrated
        assert integrator.bvp_core == mock_bvp_core
        assert integrator.quench_detector is not None
        
        # Test quench detection
        envelope = np.random.random(domain.shape)
        quenches = integrator.detect_quenches(envelope)
        
        # Should call BVP core quench detection
        mock_bvp_core.detect_quenches.assert_called_once_with(envelope)

    def test_integrator_config_copy(self, integrator, config):
        """Test that config is copied."""
        retrieved_config = integrator.get_config()
        
        # Should be a copy, not the same object
        assert retrieved_config is not config
        assert retrieved_config == config
        
        # Modifying retrieved config should not affect original
        retrieved_config["new_key"] = "new_value"
        assert "new_key" not in integrator.config
