"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP constants and configuration parameters.

This module defines all physical constants, numerical parameters, and
configuration defaults for the BVP (Base High-Frequency Field) system.

Physical Meaning:
    Contains all physical constants and numerical parameters required
    for the BVP envelope equation, impedance calculations, and quench
    detection algorithms.

Mathematical Foundation:
    Defines constants for:
    - Envelope equation parameters (κ₀, κ₂, χ', χ'')
    - Impedance calculation parameters
    - Quench detection thresholds
    - Numerical solver parameters
    - Physical material properties

Example:
    >>> constants = BVPConstants()
    >>> kappa_0 = constants.get_envelope_parameter('kappa_0')
    >>> sigma_em = constants.get_material_property('em_conductivity')
"""

import numpy as np
from typing import Dict, Any, Tuple


class BVPConstants:
    """
    Physical constants and configuration parameters for BVP system.
    
    Physical Meaning:
        Centralized storage for all physical constants, numerical parameters,
        and configuration defaults used throughout the BVP system.
        
    Mathematical Foundation:
        Organizes constants by physical category:
        - Envelope equation parameters
        - Material properties
        - Numerical solver parameters
        - Quench detection thresholds
        - Impedance calculation parameters
    """
    
    def __init__(self, config: Dict[str, Any] = None) -> None:
        """
        Initialize BVP constants with optional configuration override.
        
        Physical Meaning:
            Sets up all physical constants with values from configuration
            or uses scientifically accurate defaults.
            
        Args:
            config (Dict[str, Any], optional): Configuration to override defaults.
        """
        self.config = config or {}
        self._setup_envelope_constants()
        self._setup_material_constants()
        self._setup_numerical_constants()
        self._setup_quench_constants()
        self._setup_impedance_constants()
        self._setup_physical_constants()
    
    def _setup_envelope_constants(self) -> None:
        """Setup envelope equation constants."""
        envelope_config = self.config.get("envelope_equation", {})
        
        # Base stiffness coefficient κ₀ (dimensionless)
        self.KAPPA_0 = envelope_config.get("kappa_0", 1.0)
        
        # Nonlinear stiffness coefficient κ₂ (dimensionless)
        self.KAPPA_2 = envelope_config.get("kappa_2", 0.1)
        
        # Real part of susceptibility χ' (dimensionless)
        self.CHI_PRIME = envelope_config.get("chi_prime", 1.0)
        
        # Base imaginary susceptibility χ''₀ (dimensionless)
        self.CHI_DOUBLE_PRIME_0 = envelope_config.get("chi_double_prime_0", 0.01)
        
        # Wave number squared k₀² (1/m²)
        self.K0_SQUARED = envelope_config.get("k0_squared", 1.0)
        
        # Carrier frequency ω₀ (rad/s)
        self.CARRIER_FREQUENCY = envelope_config.get("carrier_frequency", 1.85e43)
    
    def _setup_material_constants(self) -> None:
        """Setup material property constants."""
        material_config = self.config.get("material_properties", {})
        
        # Electromagnetic conductivity σ_EM (S/m)
        self.EM_CONDUCTIVITY = material_config.get("em_conductivity", 0.01)
        
        # Weak interaction conductivity σ_weak (S/m)
        self.WEAK_CONDUCTIVITY = material_config.get("weak_conductivity", 0.001)
        
        # Base admittance Y₀ (S)
        self.BASE_ADMITTANCE = material_config.get("base_admittance", 1.0)
        
        # Nonlinear admittance coefficients
        self.ADMITTANCE_COEFF_1 = material_config.get("admittance_coeff_1", 0.1)
        self.ADMITTANCE_COEFF_2 = material_config.get("admittance_coeff_2", 0.01)
        self.ADMITTANCE_COEFF_3 = material_config.get("admittance_coeff_3", 0.001)
        self.ADMITTANCE_COEFF_4 = material_config.get("admittance_coeff_4", 0.0001)
        
        # Renormalized coefficients
        self.RENORM_COEFF_0 = material_config.get("renorm_coeff_0", 1.0)
        self.RENORM_COEFF_1 = material_config.get("renorm_coeff_1", 0.1)
        self.RENORM_COEFF_2 = material_config.get("renorm_coeff_2", 0.01)
        
        # Boundary condition coefficients
        self.BOUNDARY_PRESSURE_0 = material_config.get("boundary_pressure_0", 1.0)
        self.BOUNDARY_PRESSURE_1 = material_config.get("boundary_pressure_1", 0.1)
        self.BOUNDARY_STIFFNESS_0 = material_config.get("boundary_stiffness_0", 1.0)
        self.BOUNDARY_STIFFNESS_1 = material_config.get("boundary_stiffness_1", 0.1)
    
    def _setup_numerical_constants(self) -> None:
        """Setup numerical solver constants."""
        numerical_config = self.config.get("numerical_parameters", {})
        
        # Newton-Raphson solver parameters
        self.MAX_ITERATIONS = numerical_config.get("max_iterations", 50)
        self.TOLERANCE = numerical_config.get("tolerance", 1e-8)
        self.DAMPING_FACTOR = numerical_config.get("damping_factor", 0.8)
        self.MIN_STEP_SIZE = numerical_config.get("min_step_size", 1e-12)
        
        # Finite difference parameters
        self.FINITE_DIFF_STEP = numerical_config.get("finite_diff_step", 1e-8)
        self.REGULARIZATION = numerical_config.get("regularization", 1e-12)
        
        # Line search parameters
        self.LINE_SEARCH_MAX_ITER = numerical_config.get("line_search_max_iter", 20)
        self.LINE_SEARCH_BETA = numerical_config.get("line_search_beta", 0.5)
        self.LINE_SEARCH_GAMMA = numerical_config.get("line_search_gamma", 1e-4)
        self.ARMIJO_C1 = numerical_config.get("armijo_c1", 1e-4)
        self.CURVATURE_C2 = numerical_config.get("curvature_c2", 0.9)
        
        # Gradient descent fallback
        self.GRADIENT_DESCENT_STEP = numerical_config.get("gradient_descent_step", 0.1)
    
    def _setup_quench_constants(self) -> None:
        """Setup quench detection constants."""
        quench_config = self.config.get("quench_detection", {})
        
        # Quench detection thresholds
        self.AMPLITUDE_THRESHOLD = quench_config.get("amplitude_threshold", 0.8)
        self.DETUNING_THRESHOLD = quench_config.get("detuning_threshold", 0.1)
        self.GRADIENT_THRESHOLD = quench_config.get("gradient_threshold", 0.5)
    
    def _setup_impedance_constants(self) -> None:
        """Setup impedance calculation constants."""
        impedance_config = self.config.get("impedance_calculation", {})
        
        # Frequency analysis parameters
        self.FREQUENCY_RANGE = impedance_config.get("frequency_range", (0.1, 10.0))
        self.FREQUENCY_POINTS = impedance_config.get("frequency_points", 1000)
        self.BOUNDARY_CONDITIONS = impedance_config.get("boundary_conditions", "periodic")
        
        # Quality factor parameters
        self.QUALITY_FACTOR_THRESHOLD = impedance_config.get("quality_factor_threshold", 0.1)
        self.MIN_QUALITY_FACTOR = impedance_config.get("min_quality_factor", 1.0)
        self.MAX_QUALITY_FACTOR = impedance_config.get("max_quality_factor", 1000.0)
        
        # Peak detection parameters
        self.PROMINENCE_THRESHOLD_MULTIPLIER = impedance_config.get("prominence_threshold_multiplier", 2.0)
        self.PHASE_THRESHOLD_MULTIPLIER = impedance_config.get("phase_threshold_multiplier", 2.0)
        self.PEAK_WINDOW_SIZE = impedance_config.get("peak_window_size", 20)
        self.SMOOTHING_WINDOW_SIZE = impedance_config.get("smoothing_window_size", 5)
    
    def _setup_physical_constants(self) -> None:
        """Setup fundamental physical constants."""
        physical_config = self.config.get("physical_constants", {})
        
        # Speed of light (m/s)
        self.SPEED_OF_LIGHT = physical_config.get("speed_of_light", 299792458.0)
        
        # Vacuum permeability (H/m)
        self.VACUUM_PERMEABILITY = physical_config.get("vacuum_permeability", 4e-7 * np.pi)
        
        # Vacuum permittivity (F/m)
        self.VACUUM_PERMITTIVITY = physical_config.get("vacuum_permittivity", 8.854187817e-12)
        
        # Planck constant (J⋅s)
        self.PLANCK_CONSTANT = physical_config.get("planck_constant", 6.62607015e-34)
        
        # Boltzmann constant (J/K)
        self.BOLTZMANN_CONSTANT = physical_config.get("boltzmann_constant", 1.380649e-23)
    
    def get_envelope_parameter(self, parameter_name: str) -> float:
        """
        Get envelope equation parameter.
        
        Args:
            parameter_name (str): Name of the parameter.
            
        Returns:
            float: Parameter value.
        """
        parameter_map = {
            "kappa_0": self.KAPPA_0,
            "kappa_2": self.KAPPA_2,
            "chi_prime": self.CHI_PRIME,
            "chi_double_prime_0": self.CHI_DOUBLE_PRIME_0,
            "k0_squared": self.K0_SQUARED,
            "carrier_frequency": self.CARRIER_FREQUENCY,
        }
        return parameter_map.get(parameter_name, 0.0)
    
    def get_material_property(self, property_name: str) -> float:
        """
        Get material property constant.
        
        Args:
            property_name (str): Name of the material property.
            
        Returns:
            float: Property value.
        """
        property_map = {
            "em_conductivity": self.EM_CONDUCTIVITY,
            "weak_conductivity": self.WEAK_CONDUCTIVITY,
            "base_admittance": self.BASE_ADMITTANCE,
            "admittance_coeff_1": self.ADMITTANCE_COEFF_1,
            "admittance_coeff_2": self.ADMITTANCE_COEFF_2,
            "admittance_coeff_3": self.ADMITTANCE_COEFF_3,
            "admittance_coeff_4": self.ADMITTANCE_COEFF_4,
            "renorm_coeff_0": self.RENORM_COEFF_0,
            "renorm_coeff_1": self.RENORM_COEFF_1,
            "renorm_coeff_2": self.RENORM_COEFF_2,
            "boundary_pressure_0": self.BOUNDARY_PRESSURE_0,
            "boundary_pressure_1": self.BOUNDARY_PRESSURE_1,
            "boundary_stiffness_0": self.BOUNDARY_STIFFNESS_0,
            "boundary_stiffness_1": self.BOUNDARY_STIFFNESS_1,
        }
        return property_map.get(property_name, 0.0)
    
    def get_numerical_parameter(self, parameter_name: str) -> float:
        """
        Get numerical solver parameter.
        
        Args:
            parameter_name (str): Name of the numerical parameter.
            
        Returns:
            float: Parameter value.
        """
        parameter_map = {
            "max_iterations": self.MAX_ITERATIONS,
            "tolerance": self.TOLERANCE,
            "damping_factor": self.DAMPING_FACTOR,
            "min_step_size": self.MIN_STEP_SIZE,
            "finite_diff_step": self.FINITE_DIFF_STEP,
            "regularization": self.REGULARIZATION,
            "line_search_max_iter": self.LINE_SEARCH_MAX_ITER,
            "line_search_beta": self.LINE_SEARCH_BETA,
            "line_search_gamma": self.LINE_SEARCH_GAMMA,
            "armijo_c1": self.ARMIJO_C1,
            "curvature_c2": self.CURVATURE_C2,
            "gradient_descent_step": self.GRADIENT_DESCENT_STEP,
        }
        return parameter_map.get(parameter_name, 0.0)
    
    def get_quench_threshold(self, threshold_name: str) -> float:
        """
        Get quench detection threshold.
        
        Args:
            threshold_name (str): Name of the threshold.
            
        Returns:
            float: Threshold value.
        """
        threshold_map = {
            "amplitude_threshold": self.AMPLITUDE_THRESHOLD,
            "detuning_threshold": self.DETUNING_THRESHOLD,
            "gradient_threshold": self.GRADIENT_THRESHOLD,
        }
        return threshold_map.get(threshold_name, 0.0)
    
    def get_impedance_parameter(self, parameter_name: str) -> Any:
        """
        Get impedance calculation parameter.
        
        Args:
            parameter_name (str): Name of the impedance parameter.
            
        Returns:
            Any: Parameter value.
        """
        parameter_map = {
            "frequency_range": self.FREQUENCY_RANGE,
            "frequency_points": self.FREQUENCY_POINTS,
            "boundary_conditions": self.BOUNDARY_CONDITIONS,
            "quality_factor_threshold": self.QUALITY_FACTOR_THRESHOLD,
            "min_quality_factor": self.MIN_QUALITY_FACTOR,
            "max_quality_factor": self.MAX_QUALITY_FACTOR,
            "prominence_threshold_multiplier": self.PROMINENCE_THRESHOLD_MULTIPLIER,
            "phase_threshold_multiplier": self.PHASE_THRESHOLD_MULTIPLIER,
            "peak_window_size": self.PEAK_WINDOW_SIZE,
            "smoothing_window_size": self.SMOOTHING_WINDOW_SIZE,
        }
        return parameter_map.get(parameter_name, None)
    
    def get_physical_constant(self, constant_name: str) -> float:
        """
        Get fundamental physical constant.
        
        Args:
            constant_name (str): Name of the physical constant.
            
        Returns:
            float: Constant value.
        """
        constant_map = {
            "speed_of_light": self.SPEED_OF_LIGHT,
            "vacuum_permeability": self.VACUUM_PERMEABILITY,
            "vacuum_permittivity": self.VACUUM_PERMITTIVITY,
            "planck_constant": self.PLANCK_CONSTANT,
            "boltzmann_constant": self.BOLTZMANN_CONSTANT,
        }
        return constant_map.get(constant_name, 0.0)
    
    def compute_frequency_dependent_conductivity(self, frequency: float) -> float:
        """
        Compute frequency-dependent conductivity using Drude model.
        
        Physical Meaning:
            Computes conductivity using the Drude model for free electrons
            with frequency-dependent relaxation effects.
            
        Mathematical Foundation:
            σ(ω) = σ₀ / (1 + iωτ) where σ₀ is DC conductivity and τ is relaxation time.
            
        Args:
            frequency (float): Frequency in rad/s.
            
        Returns:
            float: Frequency-dependent conductivity.
        """
        # Drude model parameters
        dc_conductivity = self.EM_CONDUCTIVITY
        relaxation_time = 1e-12  # 1 ps relaxation time
        
        # Drude model: σ(ω) = σ₀ / (1 + (ωτ)²)
        omega_tau = frequency * relaxation_time
        return dc_conductivity / (1.0 + omega_tau**2)
    
    def compute_frequency_dependent_capacitance(self, frequency: float) -> float:
        """
        Compute frequency-dependent capacitance using Debye model.
        
        Physical Meaning:
            Computes capacitance using the Debye model for dielectric
            relaxation with frequency-dependent polarization.
            
        Mathematical Foundation:
            C(ω) = C₀ / (1 + (ωτ)²) where C₀ is static capacitance and τ is relaxation time.
            
        Args:
            frequency (float): Frequency in rad/s.
            
        Returns:
            float: Frequency-dependent capacitance.
        """
        # Debye model parameters
        static_capacitance = 1.0
        relaxation_time = 1e-9  # 1 ns relaxation time
        
        # Debye model: C(ω) = C₀ / (1 + (ωτ)²)
        omega_tau = frequency * relaxation_time
        return static_capacitance / (1.0 + omega_tau**2)
    
    def compute_frequency_dependent_inductance(self, frequency: float) -> float:
        """
        Compute frequency-dependent inductance using skin effect model.
        
        Physical Meaning:
            Computes inductance considering skin effect and frequency-dependent
            magnetic field penetration.
            
        Mathematical Foundation:
            L(ω) = L₀ * (1 + α√ω) where L₀ is DC inductance and α is skin effect parameter.
            
        Args:
            frequency (float): Frequency in rad/s.
            
        Returns:
            float: Frequency-dependent inductance.
        """
        # Skin effect parameters
        dc_inductance = 1.0
        skin_effect_alpha = 0.05
        
        # Skin effect model: L(ω) = L₀ * (1 + α√ω)
        return dc_inductance * (1.0 + skin_effect_alpha * np.sqrt(frequency))
    
    def __repr__(self) -> str:
        """String representation of BVP constants."""
        return (
            f"BVPConstants(carrier_freq={self.CARRIER_FREQUENCY}, "
            f"kappa_0={self.KAPPA_0}, kappa_2={self.KAPPA_2})"
        )
