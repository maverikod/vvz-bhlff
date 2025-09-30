"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Comprehensive unit tests for BVP constants module.

This module provides comprehensive unit tests for the BVP constants module,
covering all classes and methods to achieve high test coverage.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from bhlff.core.bvp.bvp_constants_base import BVPConstantsBase
from bhlff.core.bvp.constants.bvp_constants_advanced import BVPConstantsAdvanced
from bhlff.core.bvp.constants.frequency_dependent_properties import FrequencyDependentProperties
from bhlff.core.bvp.constants.nonlinear_coefficients import NonlinearCoefficients
from bhlff.core.bvp.constants.renormalized_coefficients import RenormalizedCoefficients


class TestBVPConstantsBase:
    """Comprehensive tests for BVPConstantsBase class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "envelope_equation": {
                "kappa_0": 2.0,
                "kappa_2": 0.2,
                "chi_prime": 2.0,
                "chi_double_prime_0": 0.02,
                "k0_squared": 2.0,
                "carrier_frequency": 2.0e43,
                "amplitude_threshold": 0.9,
                "detuning_threshold": 0.2,
                "gradient_threshold": 0.6
            },
            "material_properties": {
                "em_conductivity": 0.02,
                "weak_conductivity": 0.03,
                "mu": 2.0,
                "beta": 1.2,
                "lambda_param": 0.2,
                "nu": 2.0
            },
            "physical_constants": {
                "speed_of_light": 3.0e8,
                "planck_constant": 6.6e-34,
                "boltzmann_constant": 1.4e-23
            }
        }

    @pytest.fixture
    def constants_base(self, config):
        """Create BVPConstantsBase instance."""
        return BVPConstantsBase(config)

    def test_constants_base_initialization(self, constants_base):
        """Test constants base initialization."""
        assert constants_base.KAPPA_0 == 2.0
        assert constants_base.KAPPA_2 == 0.2
        assert constants_base.CHI_PRIME == 2.0
        assert constants_base.CHI_DOUBLE_PRIME_0 == 0.02
        assert constants_base.K0_SQUARED == 2.0
        assert constants_base.CARRIER_FREQUENCY == 2.0e43

    def test_constants_base_default_initialization(self):
        """Test constants base with default values."""
        constants = BVPConstantsBase()
        assert constants.KAPPA_0 == 1.0
        assert constants.KAPPA_2 == 0.1
        assert constants.CHI_PRIME == 1.0
        assert constants.CHI_DOUBLE_PRIME_0 == 0.01
        assert constants.K0_SQUARED == 1.0
        assert constants.CARRIER_FREQUENCY == 1.85e43

    def test_constants_base_material_properties(self, constants_base):
        """Test material properties setup."""
        assert constants_base.EM_CONDUCTIVITY == 0.02
        assert constants_base.WEAK_CONDUCTIVITY == 0.03
        assert constants_base.MU == 2.0
        assert constants_base.BETA == 1.2
        assert constants_base.LAMBDA_PARAM == 0.2
        assert constants_base.NU == 2.0

    def test_constants_base_physical_constants(self, constants_base):
        """Test physical constants setup."""
        assert constants_base.SPEED_OF_LIGHT == 3.0e8
        assert constants_base.PLANCK_CONSTANT == 6.6e-34
        assert constants_base.BOLTZMANN_CONSTANT == 1.4e-23

    def test_constants_base_quench_parameters(self, constants_base):
        """Test quench parameters setup."""
        assert constants_base.AMPLITUDE_THRESHOLD == 0.9
        assert constants_base.DETUNING_THRESHOLD == 0.2
        assert constants_base.GRADIENT_THRESHOLD == 0.6

    def test_get_envelope_parameter(self, constants_base):
        """Test getting envelope parameters."""
        assert constants_base.get_envelope_parameter("kappa_0") == 2.0
        assert constants_base.get_envelope_parameter("kappa_2") == 0.2
        assert constants_base.get_envelope_parameter("chi_prime") == 2.0
        assert constants_base.get_envelope_parameter("k0_squared") == 2.0

    def test_get_envelope_parameter_invalid(self, constants_base):
        """Test getting invalid envelope parameter."""
        with pytest.raises(KeyError):
            constants_base.get_envelope_parameter("invalid_param")

    def test_get_basic_material_property(self, constants_base):
        """Test getting basic material properties."""
        assert constants_base.get_basic_material_property("mu") == 2.0
        assert constants_base.get_basic_material_property("beta") == 1.2
        assert constants_base.get_basic_material_property("lambda_param") == 0.2
        assert constants_base.get_basic_material_property("nu") == 2.0

    def test_get_basic_material_property_invalid(self, constants_base):
        """Test getting invalid material property."""
        with pytest.raises(KeyError):
            constants_base.get_basic_material_property("invalid_property")

    def test_get_physical_constant(self, constants_base):
        """Test getting physical constants."""
        assert constants_base.get_physical_constant("speed_of_light") == 3.0e8
        assert constants_base.get_physical_constant("planck_constant") == 6.6e-34
        assert constants_base.get_physical_constant("boltzmann_constant") == 1.4e-23

    def test_get_physical_constant_invalid(self, constants_base):
        """Test getting invalid physical constant."""
        with pytest.raises(KeyError):
            constants_base.get_physical_constant("invalid_constant")

    def test_get_physical_parameter(self, constants_base):
        """Test getting physical parameters."""
        assert constants_base.get_physical_parameter("phase_velocity_threshold") == 1e6
        assert constants_base.get_physical_parameter("epsilon_threshold") == 0.1

    def test_get_physical_parameter_invalid(self, constants_base):
        """Test getting invalid physical parameter."""
        with pytest.raises(KeyError):
            constants_base.get_physical_parameter("invalid_parameter")

    def test_get_quench_parameter(self, constants_base):
        """Test getting quench parameters."""
        assert constants_base.get_quench_parameter("amplitude_threshold") == 0.9
        assert constants_base.get_quench_parameter("detuning_threshold") == 0.2
        assert constants_base.get_quench_parameter("gradient_threshold") == 0.6

    def test_get_quench_parameter_invalid(self, constants_base):
        """Test getting invalid quench parameter."""
        with pytest.raises(KeyError):
            constants_base.get_quench_parameter("invalid_quench_param")


class TestBVPConstantsAdvanced:
    """Comprehensive tests for BVPConstantsAdvanced class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "envelope_equation": {
                "kappa_0": 1.0,
                "kappa_2": 0.1,
                "chi_prime": 1.0,
                "chi_double_prime_0": 0.01,
                "k0_squared": 1.0,
                "carrier_frequency": 1.85e43
            },
            "material_properties": {
                "admittance_coeff_1": 0.2,
                "admittance_coeff_2": 0.02,
                "admittance_coeff_3": 0.002,
                "admittance_coeff_4": 0.0002,
                "renorm_coeff_0": 2.0,
                "renorm_coeff_1": 0.2,
                "renorm_coeff_2": 0.02,
                "boundary_pressure_0": 2.0,
                "boundary_pressure_1": 0.2,
                "boundary_stiffness_0": 2.0,
                "boundary_stiffness_1": 0.2
            }
        }

    @pytest.fixture
    def constants_advanced(self, config):
        """Create BVPConstantsAdvanced instance."""
        return BVPConstantsAdvanced(config)

    def test_constants_advanced_initialization(self, constants_advanced):
        """Test constants advanced initialization."""
        assert constants_advanced.ADMITTANCE_COEFF_1 == 0.2
        assert constants_advanced.ADMITTANCE_COEFF_2 == 0.02
        assert constants_advanced.ADMITTANCE_COEFF_3 == 0.002
        assert constants_advanced.ADMITTANCE_COEFF_4 == 0.0002

    def test_constants_advanced_renormalized_coeffs(self, constants_advanced):
        """Test renormalized coefficients."""
        assert constants_advanced.RENORM_COEFF_0 == 2.0
        assert constants_advanced.RENORM_COEFF_1 == 0.2
        assert constants_advanced.RENORM_COEFF_2 == 0.02

    def test_constants_advanced_boundary_coeffs(self, constants_advanced):
        """Test boundary coefficients."""
        assert constants_advanced.BOUNDARY_PRESSURE_0 == 2.0
        assert constants_advanced.BOUNDARY_PRESSURE_1 == 0.2
        assert constants_advanced.BOUNDARY_STIFFNESS_0 == 2.0
        assert constants_advanced.BOUNDARY_STIFFNESS_1 == 0.2

    def test_get_advanced_material_property(self, constants_advanced):
        """Test getting advanced material properties."""
        assert constants_advanced.get_advanced_material_property("admittance_coeff_1") == 0.2
        assert constants_advanced.get_advanced_material_property("renorm_coeff_0") == 2.0
        assert constants_advanced.get_advanced_material_property("boundary_pressure_0") == 2.0

    def test_get_advanced_material_property_invalid(self, constants_advanced):
        """Test getting invalid advanced material property."""
        with pytest.raises(KeyError):
            constants_advanced.get_advanced_material_property("invalid_property")

    def test_constants_advanced_components(self, constants_advanced):
        """Test that components are initialized."""
        assert hasattr(constants_advanced, 'frequency_properties')
        assert hasattr(constants_advanced, 'nonlinear_coeffs')
        assert hasattr(constants_advanced, 'renormalized_coeffs')
        assert isinstance(constants_advanced.frequency_properties, FrequencyDependentProperties)
        assert isinstance(constants_advanced.nonlinear_coeffs, NonlinearCoefficients)
        assert isinstance(constants_advanced.renormalized_coeffs, RenormalizedCoefficients)


class TestFrequencyDependentProperties:
    """Comprehensive tests for FrequencyDependentProperties class."""

    @pytest.fixture
    def mock_constants(self):
        """Create mock constants."""
        constants = Mock()
        constants.get_basic_material_property.return_value = 1.0
        constants.get_physical_constant.return_value = 3.0e8
        return constants

    @pytest.fixture
    def frequency_properties(self, mock_constants):
        """Create FrequencyDependentProperties instance."""
        return FrequencyDependentProperties(mock_constants)

    def test_frequency_properties_initialization(self, frequency_properties, mock_constants):
        """Test frequency properties initialization."""
        assert frequency_properties.constants == mock_constants

    def test_compute_frequency_dependent_conductivity(self, frequency_properties):
        """Test frequency-dependent conductivity computation."""
        frequency = np.array([1e9, 1e10, 1e11])
        conductivity = frequency_properties.compute_frequency_dependent_conductivity(frequency)
        
        assert isinstance(conductivity, np.ndarray)
        assert conductivity.shape == frequency.shape
        assert np.all(conductivity > 0)

    def test_compute_frequency_dependent_capacitance(self, frequency_properties):
        """Test frequency-dependent capacitance computation."""
        frequency = np.array([1e9, 1e10, 1e11])
        capacitance = frequency_properties.compute_frequency_dependent_capacitance(frequency)
        
        assert isinstance(capacitance, np.ndarray)
        assert capacitance.shape == frequency.shape
        assert np.all(capacitance > 0)

    def test_compute_frequency_dependent_inductance(self, frequency_properties):
        """Test frequency-dependent inductance computation."""
        frequency = np.array([1e9, 1e10, 1e11])
        inductance = frequency_properties.compute_frequency_dependent_inductance(frequency)
        
        assert isinstance(inductance, np.ndarray)
        assert inductance.shape == frequency.shape
        assert np.all(inductance > 0)

    def test_compute_frequency_dependent_conductivity_scalar(self, frequency_properties):
        """Test frequency-dependent conductivity with scalar input."""
        frequency = 1e9
        conductivity = frequency_properties.compute_frequency_dependent_conductivity(frequency)
        
        assert isinstance(conductivity, (float, np.ndarray))
        assert conductivity > 0

    def test_compute_frequency_dependent_capacitance_scalar(self, frequency_properties):
        """Test frequency-dependent capacitance with scalar input."""
        frequency = 1e9
        capacitance = frequency_properties.compute_frequency_dependent_capacitance(frequency)
        
        assert isinstance(capacitance, (float, np.ndarray))
        assert capacitance > 0

    def test_compute_frequency_dependent_inductance_scalar(self, frequency_properties):
        """Test frequency-dependent inductance with scalar input."""
        frequency = 1e9
        inductance = frequency_properties.compute_frequency_dependent_inductance(frequency)
        
        assert isinstance(inductance, (float, np.ndarray))
        assert inductance > 0


class TestNonlinearCoefficients:
    """Comprehensive tests for NonlinearCoefficients class."""

    @pytest.fixture
    def mock_constants(self):
        """Create mock constants."""
        constants = Mock()
        constants.get_advanced_material_property.return_value = 0.1
        return constants

    @pytest.fixture
    def nonlinear_coeffs(self, mock_constants):
        """Create NonlinearCoefficients instance."""
        return NonlinearCoefficients(mock_constants)

    def test_nonlinear_coeffs_initialization(self, nonlinear_coeffs, mock_constants):
        """Test nonlinear coefficients initialization."""
        assert nonlinear_coeffs.constants == mock_constants

    def test_compute_nonlinear_admittance_coefficients(self, nonlinear_coeffs):
        """Test nonlinear admittance coefficients computation."""
        frequency = np.array([1e9, 1e10, 1e11])
        amplitude = np.array([0.1, 0.5, 1.0])
        
        coeffs = nonlinear_coeffs.compute_nonlinear_admittance_coefficients(frequency, amplitude)
        
        assert isinstance(coeffs, dict)
        assert "linear" in coeffs
        assert "quadratic" in coeffs
        assert "cubic" in coeffs
        assert "quartic" in coeffs
        
        for key, value in coeffs.items():
            assert isinstance(value, np.ndarray)
            assert value.shape == frequency.shape

    def test_compute_nonlinear_admittance_coefficients_scalar(self, nonlinear_coeffs):
        """Test nonlinear admittance coefficients with scalar inputs."""
        frequency = 1e9
        amplitude = 0.5
        
        coeffs = nonlinear_coeffs.compute_nonlinear_admittance_coefficients(frequency, amplitude)
        
        assert isinstance(coeffs, dict)
        for key, value in coeffs.items():
            assert isinstance(value, (float, np.ndarray))


class TestRenormalizedCoefficients:
    """Comprehensive tests for RenormalizedCoefficients class."""

    @pytest.fixture
    def mock_constants(self):
        """Create mock constants."""
        constants = Mock()
        constants.get_advanced_material_property.return_value = 1.0
        return constants

    @pytest.fixture
    def renormalized_coeffs(self, mock_constants):
        """Create RenormalizedCoefficients instance."""
        return RenormalizedCoefficients(mock_constants)

    def test_renormalized_coeffs_initialization(self, renormalized_coeffs, mock_constants):
        """Test renormalized coefficients initialization."""
        assert renormalized_coeffs.constants == mock_constants

    def test_compute_renormalized_coefficients(self, renormalized_coeffs):
        """Test renormalized coefficients computation."""
        amplitude = np.array([0.1, 0.5, 1.0])
        gradient = np.array([0.01, 0.05, 0.1])
        
        coeffs = renormalized_coeffs.compute_renormalized_coefficients(amplitude, gradient)
        
        assert isinstance(coeffs, dict)
        assert "renormalized_0" in coeffs
        assert "renormalized_1" in coeffs
        assert "renormalized_2" in coeffs
        
        for key, value in coeffs.items():
            assert isinstance(value, np.ndarray)
            assert value.shape == amplitude.shape

    def test_compute_renormalized_coefficients_scalar(self, renormalized_coeffs):
        """Test renormalized coefficients with scalar inputs."""
        amplitude = 0.5
        gradient = 0.05
        
        coeffs = renormalized_coeffs.compute_renormalized_coefficients(amplitude, gradient)
        
        assert isinstance(coeffs, dict)
        for key, value in coeffs.items():
            assert isinstance(value, (float, np.ndarray))
