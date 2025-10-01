"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP Core Facade - Unified interface for BVP framework.

This module provides a unified facade for the BVP framework, consolidating
all BVP core functionality into a single, well-organized interface that
follows the 1 class = 1 file principle while maintaining modularity.

Physical Meaning:
    The BVP Core Facade serves as the central backbone of the entire system,
    where all observed particles and fields are manifestations of envelope
    modulations and beatings of the high-frequency carrier field. This facade
    provides a unified interface to all BVP operations while maintaining
    the modular architecture underneath.

Mathematical Foundation:
    BVP implements the envelope equation:
    ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    where κ(|a|) = κ₀ + κ₂|a|² is nonlinear stiffness and
    χ(|a|) = χ' + iχ''(|a|) is effective susceptibility with quenches.

Architecture:
    - BVPCoreFacade: Main facade class
    - BVPCoreOperations: Core operations (envelope solving, quench detection)
    - BVPCore7DInterface: 7D space-time operations
    - BVPConstants: Configuration and constants management

Example:
    >>> bvp_core = BVPCoreFacade(domain, config, domain_7d)
    >>> envelope = bvp_core.solve_envelope(source)
    >>> quenches = bvp_core.detect_quenches(envelope)
    >>> impedance = bvp_core.compute_impedance(envelope)
"""

import numpy as np
from typing import Dict, Any, Optional, Union
import logging

from ...domain import Domain
from ...domain.domain_7d import Domain7D
from ..bvp_constants import BVPConstants
from .bvp_operations import BVPCoreOperations
from .bvp_7d_interface import BVPCore7DInterface


class BVPCoreFacade:
    """
    BVP Core Facade - Unified interface for BVP framework.
    
    Physical Meaning:
        Serves as the central backbone of the entire system, where all observed
        particles and fields are manifestations of envelope modulations and
        beatings of the high-frequency carrier field. Provides a unified
        interface to all BVP operations while maintaining modular architecture.
        
    Mathematical Foundation:
        Implements the 7D envelope equation:
        ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
        where κ(|a|) = κ₀ + κ₂|a|² is nonlinear stiffness and
        χ(|a|) = χ' + iχ''(|a|) is effective susceptibility with quenches.
        
    Architecture:
        - Uses composition to combine BVPCoreOperations and BVPCore7DInterface
        - Maintains backward compatibility with existing interfaces
        - Provides both standard and 7D operations
        - Centralizes configuration and constants management
        
    Attributes:
        domain (Domain): Standard computational domain for BVP operations.
        domain_7d (Optional[Domain7D]): 7D computational domain for full space-time.
        config (Dict[str, Any]): Configuration parameters.
        _bvp_constants (BVPConstants): BVP constants and configuration.
        _operations (BVPCoreOperations): Core BVP operations.
        _7d_interface (Optional[BVPCore7DInterface]): 7D space-time interface.
        logger (logging.Logger): Logger for BVP operations.
    """
    
    def __init__(
        self, 
        domain: Domain, 
        config: Dict[str, Any], 
        domain_7d: Optional[Domain7D] = None
    ) -> None:
        """
        Initialize BVP Core Facade.
        
        Physical Meaning:
            Sets up the central BVP framework with computational domains
            and configuration parameters. Initializes all necessary
            components for BVP operations including envelope solving,
            quench detection, impedance computation, and 7D operations.
            
        Args:
            domain (Domain): Standard computational domain for BVP operations.
            config (Dict[str, Any]): Configuration parameters including:
                - carrier_frequency: BVP carrier frequency ω₀
                - envelope_equation: Envelope equation parameters
                - quench_detection: Quench detection parameters
                - impedance_calculation: Impedance calculation parameters
            domain_7d (Optional[Domain7D]): 7D computational domain for
                full space-time operations. If None, 7D operations are disabled.
        """
        self.domain = domain
        self.config = config
        self.domain_7d = domain_7d
        self.logger = logging.getLogger(__name__)
        
        # Initialize BVP constants
        self._bvp_constants = BVPConstants(config)
        
        # Initialize core operations
        self._operations = BVPCoreOperations(domain, config, domain_7d)
        
        # Initialize 7D interface if 7D domain is available
        self._7d_interface = None
        if domain_7d is not None:
            self._7d_interface = BVPCore7DInterface(domain_7d, config)
            self.logger.info("7D interface initialized")
        else:
            self.logger.info("7D interface disabled - domain_7d not provided")
    
    # ============================================================================
    # Core BVP Operations (Standard Interface)
    # ============================================================================
    
    def solve_envelope(self, source: np.ndarray) -> np.ndarray:
        """
        Solve BVP envelope equation for U(1)³ phase structure.
        
        Physical Meaning:
            Computes the envelope a(x,φ,t) of the Base High-Frequency Field
            in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ that modulates the high-frequency carrier.
            The envelope is a vector of three U(1) phase components Θ_a (a=1..3).
            
        Mathematical Foundation:
            Solves ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t) for the envelope a(x,φ,t)
            where a is a vector of three U(1) phase components in 7D space-time.
            
        Args:
            source (np.ndarray): Source term s(x,φ,t) in 7D space-time.
                Represents external excitations or initial conditions in M₇.
                
        Returns:
            np.ndarray: BVP envelope a(x,φ,t) in 7D space-time.
                Represents the envelope modulation of the high-frequency carrier
                as a vector of three U(1) phase components.
                
        Raises:
            ValueError: If source has incompatible shape with domain.
            RuntimeError: If envelope equation solution fails.
        """
        return self._operations.solve_envelope(source)
    
    def detect_quenches(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Detect quench events when local thresholds are reached.
        
        Physical Meaning:
            Identifies when BVP dissipatively "dumps" energy into
            the medium at local thresholds (amplitude/detuning/gradient).
            Quenches represent threshold events where the BVP field
            undergoes a local regime transition.
            
        Mathematical Foundation:
            Applies three threshold criteria:
            - amplitude: |A| > |A_q|
            - detuning: |ω - ω_0| > Δω_q
            - gradient: |∇A| > |∇A_q|
            where A_q, Δω_q, and ∇A_q are the quench thresholds.
            
        Args:
            envelope (np.ndarray): BVP envelope a(x) to analyze.
            
        Returns:
            Dict[str, Any]: Quench detection results including:
                - quenches_detected (bool): Whether any quenches were found
                - quench_locations (List[Tuple]): Coordinates of quench events
                - quench_types (List[str]): Types of quenches (amplitude/detuning/gradient)
                - quench_strengths (List[float]): Strength of each quench
        """
        return self._operations.detect_quenches(envelope)
    
    def compute_impedance(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Compute impedance/admittance from BVP envelope.
        
        Physical Meaning:
            Calculates Y(ω), R(ω), T(ω), and peaks {ω_n,Q_n}
            from the BVP envelope at boundaries. This provides
            the interface between BVP and other system components
            (tail resonators, transition zone, core).
            
        Mathematical Foundation:
            Computes boundary functions from BVP envelope:
            - Admittance: Y(ω) = I(ω)/V(ω)
            - Reflection coefficient: R(ω)
            - Transmission coefficient: T(ω)
            - Resonance peaks: {ω_n, Q_n}
            
        Args:
            envelope (np.ndarray): BVP envelope a(x) to analyze.
            
        Returns:
            Dict[str, Any]: Impedance analysis results including:
                - admittance: Admittance Y(ω)
                - reflection: Reflection coefficient R(ω)
                - transmission: Transmission coefficient T(ω)
                - resonance_peaks: Resonance peaks {ω_n,Q_n}
        """
        return self._operations.compute_impedance(envelope)
    
    # ============================================================================
    # 7D Space-Time Operations
    # ============================================================================
    
    def solve_envelope_7d(self, source_7d: np.ndarray) -> np.ndarray:
        """
        Solve 7D BVP envelope equation.
        
        Physical Meaning:
            Solves the full 7D envelope equation in space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ
            using the 7D envelope equation solver.
            
        Args:
            source_7d (np.ndarray): 7D source term s(x,φ,t).
            
        Returns:
            np.ndarray: 7D envelope solution a(x,φ,t).
            
        Raises:
            RuntimeError: If 7D domain is not available.
        """
        if self._7d_interface is None:
            raise RuntimeError("7D domain not available for 7D envelope equation")
        
        return self._7d_interface.solve_envelope_7d(source_7d)
    
    def validate_postulates_7d(self, envelope_7d: np.ndarray) -> Dict[str, Any]:
        """
        Validate all 9 BVP postulates for 7D field.
        
        Physical Meaning:
            Validates all 9 BVP postulates to ensure the 7D field
            satisfies the fundamental properties of the BVP framework.
            
        Args:
            envelope_7d (np.ndarray): 7D BVP envelope field.
            
        Returns:
            Dict[str, Any]: Validation results from all postulates.
            
        Raises:
            RuntimeError: If 7D postulates are not available.
        """
        if self._7d_interface is None:
            raise RuntimeError("7D postulates not available")
        
        return self._7d_interface.validate_postulates_7d(envelope_7d)
    
    # ============================================================================
    # Configuration and Constants
    # ============================================================================
    
    def get_carrier_frequency(self) -> float:
        """
        Get BVP carrier frequency.
        
        Physical Meaning:
            Returns the high-frequency carrier frequency ω₀ of the BVP field,
            which is the fundamental frequency that all envelope modulations
            and beatings are based upon.
            
        Returns:
            float: BVP carrier frequency ω₀.
        """
        return self._bvp_constants.get_carrier_frequency()
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get BVP core parameters.
        
        Physical Meaning:
            Returns the current values of all BVP parameters for
            monitoring and analysis purposes.
            
        Returns:
            Dict[str, Any]: Dictionary containing all parameters.
        """
        params = self._bvp_constants.get_all_constants()
        
        if self._7d_interface is not None:
            params["7d_interface"] = self._7d_interface.get_7d_parameters()
        
        return params
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update BVP configuration.
        
        Physical Meaning:
            Updates the BVP configuration parameters and reinitializes
            components that depend on configuration changes.
            
        Args:
            new_config (Dict[str, Any]): New configuration parameters.
        """
        self.config.update(new_config)
        self._bvp_constants = BVPConstants(self.config)
        
        # Reinitialize operations with new config
        self._operations = BVPCoreOperations(self.domain, self.config, self.domain_7d)
        
        # Reinitialize 7D interface if available
        if self._7d_interface is not None:
            self._7d_interface = BVPCore7DInterface(self.domain_7d, self.config)
        
        self.logger.info("BVP configuration updated")
    
    # ============================================================================
    # Domain and Interface Access
    # ============================================================================
    
    def get_7d_domain(self) -> Optional[Domain7D]:
        """
        Get the 7D domain.
        
        Returns:
            Optional[Domain7D]: The 7D space-time domain if available.
        """
        if self._7d_interface is None:
            return None
        return self._7d_interface.get_7d_domain()
    
    def is_7d_available(self) -> bool:
        """
        Check if 7D operations are available.
        
        Returns:
            bool: True if 7D domain and interface are available.
        """
        return self._7d_interface is not None
    
    def get_operations(self) -> BVPCoreOperations:
        """
        Get the core operations instance.
        
        Returns:
            BVPCoreOperations: The core operations instance.
        """
        return self._operations
    
    def get_7d_interface(self) -> Optional[BVPCore7DInterface]:
        """
        Get the 7D interface instance.
        
        Returns:
            Optional[BVPCore7DInterface]: The 7D interface if available.
        """
        return self._7d_interface
    
    # ============================================================================
    # String Representation and Debugging
    # ============================================================================
    
    def __repr__(self) -> str:
        """
        String representation of BVP Core Facade.
        
        Returns:
            str: String representation showing domain and carrier frequency.
        """
        return (
            f"BVPCoreFacade(domain={self.domain}, "
            f"carrier_freq={self.get_carrier_frequency()}, "
            f"7d_available={self.is_7d_available()})"
        )
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status information.
        
        Returns:
            Dict[str, Any]: Status information including:
                - domain_info: Domain configuration
                - 7d_available: Whether 7D operations are available
                - carrier_frequency: BVP carrier frequency
                - config_keys: Available configuration keys
        """
        status = {
            "domain_info": {
                "shape": self.domain.shape,
                "dimensions": self.domain.dimensions,
                "L": getattr(self.domain, 'L', None)
            },
            "7d_available": self.is_7d_available(),
            "carrier_frequency": self.get_carrier_frequency(),
            "config_keys": list(self.config.keys())
        }
        
        if self.is_7d_available():
            status["7d_domain_info"] = {
                "shape": self.domain_7d.shape,
                "dimensions": self.domain_7d.dimensions
            }
        
        return status
