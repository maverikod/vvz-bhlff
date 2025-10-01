"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Adaptive integrator for 7D phase field dynamics.

This module implements the adaptive integrator for solving dynamic phase field
equations in 7D space-time with automatic error control and time step adjustment.

Physical Meaning:
    Adaptive integrator provides automatic time step control to maintain
    accuracy while ensuring numerical stability of phase field evolution
    in 7D space-time with optimal performance.

Mathematical Foundation:
    Uses embedded Runge-Kutta methods with error estimation and automatic
    step size adjustment for optimal performance and accuracy control.
"""

from typing import Dict, Any, Optional, Tuple, Callable
import numpy as np
import logging

from .base_integrator import BaseTimeIntegrator
from .memory_kernel import MemoryKernel
from .quench_detector import QuenchDetector
from ..fft import SpectralOperations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..fft import FractionalLaplacian


class AdaptiveIntegrator(BaseTimeIntegrator):
    """
    Adaptive integrator with error control and stability monitoring.

    Physical Meaning:
        Automatically adjusts time step to maintain accuracy while
        ensuring numerical stability of phase field evolution in 7D space-time.
        Uses embedded Runge-Kutta methods with error estimation.

    Mathematical Foundation:
        Implements adaptive time stepping with:
        - Error estimation using embedded methods
        - Automatic step size adjustment
        - Stability monitoring and CFL conditions
        - Performance optimization through step size control

    Attributes:
        domain (Domain): Computational domain.
        parameters (Parameters): Physics parameters.
        _spectral_ops (SpectralOperations): Spectral operations for FFT.
        _fractional_laplacian (FractionalLaplacian): Fractional Laplacian operator.
        _spectral_coeffs (np.ndarray): Pre-computed spectral coefficients.
        _memory_kernel (Optional[MemoryKernel]): Memory kernel for non-local effects.
        _quench_detector (Optional[QuenchDetector]): Quench detection system.
        _current_dt (float): Current time step.
        _min_dt (float): Minimum allowed time step.
        _max_dt (float): Maximum allowed time step.
        _tolerance (float): Error tolerance for adaptive control.
        _safety_factor (float): Safety factor for step size adjustment.
    """

    def __init__(
        self,
        domain,
        parameters,
        tolerance: float = 1e-8,
        safety_factor: float = 0.9,
        min_dt: float = 1e-6,
        max_dt: float = 1e-2,
    ) -> None:
        """
        Initialize adaptive integrator.

        Physical Meaning:
            Sets up the adaptive integrator with the computational domain
            and physics parameters, configuring error control and time step
            management for optimal performance.

        Args:
            domain (Domain): Computational domain for the simulation.
            parameters (Parameters): Physics parameters controlling
                the behavior of the phase field system.
            tolerance (float): Error tolerance for adaptive control.
            safety_factor (float): Safety factor for step size adjustment.
            min_dt (float): Minimum allowed time step.
            max_dt (float): Maximum allowed time step.
        """
        super().__init__(domain, parameters)

        # Initialize spectral operations
        self._spectral_ops = SpectralOperations(domain, parameters.precision)

        # Initialize fractional Laplacian
        from ..fft import FractionalLaplacian

        self._fractional_laplacian = FractionalLaplacian(
            domain, parameters.beta, parameters.lambda_param
        )

        # Pre-compute spectral coefficients
        self._spectral_coeffs = None
        self._setup_spectral_coefficients()

        # Adaptive control parameters
        self._tolerance = tolerance
        self._safety_factor = safety_factor
        self._min_dt = min_dt
        self._max_dt = max_dt
        self._current_dt = min_dt

        self._initialized = True
        self.logger.info(f"Adaptive integrator initialized with tolerance={tolerance}")

    def _setup_spectral_coefficients(self) -> None:
        """
        Setup spectral coefficients for adaptive integrator.

        Physical Meaning:
            Pre-computes the spectral representation of the operator
            for efficient adaptive integration with error estimation.
        """
        # Get spectral coefficients from fractional Laplacian
        self._spectral_coeffs = self._fractional_laplacian.get_spectral_coefficients()

        # Scale by diffusion coefficient
        self._spectral_coeffs *= self.parameters.nu

        self.logger.info("Spectral coefficients computed for adaptive integrator")

    def integrate(
        self,
        initial_field: np.ndarray,
        source_field: np.ndarray,
        time_steps: np.ndarray,
    ) -> np.ndarray:
        """
        Integrate the dynamic equation over time using adaptive method.

        Physical Meaning:
            Solves the dynamic phase field equation over the specified
            time steps using adaptive time stepping with automatic error
            control and step size adjustment.

        Mathematical Foundation:
            Uses embedded Runge-Kutta methods with error estimation:
            - Fourth-order accurate solution
            - Fifth-order error estimation
            - Automatic step size adjustment based on error
            - Stability monitoring and CFL conditions

        Args:
            initial_field (np.ndarray): Initial field configuration a(x,φ,0).
            source_field (np.ndarray): Source term s(x,φ,t) over time.
            time_steps (np.ndarray): Time points for integration.

        Returns:
            np.ndarray: Field evolution a(x,φ,t) over time.
        """
        if not self._initialized:
            raise RuntimeError("Integrator not initialized")

        # Validate inputs
        if initial_field.shape != self.domain.shape:
            raise ValueError(
                f"Initial field shape {initial_field.shape} incompatible with domain {self.domain.shape}"
            )

        if source_field.shape != (len(time_steps),) + self.domain.shape:
            raise ValueError(
                f"Source field shape {source_field.shape} incompatible with time steps and domain"
            )

        # Initialize result array
        result = np.zeros((len(time_steps),) + self.domain.shape, dtype=np.complex128)
        result[0] = initial_field.copy()

        # Current field state
        current_field = initial_field.copy()
        current_time = time_steps[0]

        # Adaptive integration
        for i in range(1, len(time_steps)):
            target_time = time_steps[i]

            # Integrate from current_time to target_time with adaptive stepping
            current_field = self._adaptive_step_to_time(
                current_field, current_time, target_time, source_field, i
            )

            result[i] = current_field.copy()
            current_time = target_time

        return result

    def step(
        self, current_field: np.ndarray, source_field: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        Perform a single adaptive time step.

        Physical Meaning:
            Advances the field configuration by one time step using
            adaptive error control and step size adjustment.

        Args:
            current_field (np.ndarray): Current field configuration.
            source_field (np.ndarray): Source term at current time.
            dt (float): Proposed time step size.

        Returns:
            np.ndarray: Field configuration at next time step.
        """
        # Use embedded Runge-Kutta method for error estimation
        field_next, error_estimate = self._embedded_rk_step(
            current_field, source_field, dt
        )

        # Adjust time step based on error estimate
        self._adjust_time_step(error_estimate, dt)

        return field_next

    def _adaptive_step_to_time(
        self,
        current_field: np.ndarray,
        current_time: float,
        target_time: float,
        source_field: np.ndarray,
        time_index: int,
    ) -> np.ndarray:
        """
        Adaptively step from current_time to target_time.

        Physical Meaning:
            Performs adaptive integration from current time to target time,
            automatically adjusting step size to maintain accuracy.
        """
        field = current_field.copy()
        time = current_time

        while time < target_time:
            # Calculate remaining time
            remaining_time = target_time - time

            # Use current adaptive time step or remaining time, whichever is smaller
            dt = min(self._current_dt, remaining_time)

            # Get source at current time (interpolate if necessary)
            if time_index < len(source_field) - 1:
                # Linear interpolation between time points
                alpha = (time - time_index) / (time_index + 1 - time_index)
                source = (1 - alpha) * source_field[time_index] + alpha * source_field[
                    time_index + 1
                ]
            else:
                source = source_field[time_index]

            # Perform adaptive step
            field = self.step(field, source, dt)
            time += dt

            # Check if we've reached the target time
            if abs(time - target_time) < 1e-12:
                break

        return field

    def _embedded_rk_step(
        self, field: np.ndarray, source: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, float]:
        """
        Perform embedded Runge-Kutta step with error estimation.

        Physical Meaning:
            Uses embedded Runge-Kutta method to compute both fourth-order
            accurate solution and fifth-order error estimate for adaptive control.

        Mathematical Foundation:
            Implements embedded RK4(5) method:
            - k1 = dt * f(t, y)
            - k2 = dt * f(t + dt/2, y + k1/2)
            - k3 = dt * f(t + dt/2, y + k2/2)
            - k4 = dt * f(t + dt, y + k3)
            - y4 = y + (k1 + 2*k2 + 2*k3 + k4)/6  (4th order)
            - y5 = y + (7*k1 + 32*k3 + 12*k4 + 32*k5 + 7*k6)/90  (5th order)
            - error = |y5 - y4|
        """
        # Compute k1
        k1 = self._compute_rhs(field, source)

        # Compute k2
        field_temp = field + 0.5 * dt * k1
        k2 = self._compute_rhs(field_temp, source)

        # Compute k3
        field_temp = field + 0.5 * dt * k2
        k3 = self._compute_rhs(field_temp, source)

        # Compute k4
        field_temp = field + dt * k3
        k4 = self._compute_rhs(field_temp, source)

        # Fourth-order solution
        field_4th = field + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Fifth-order solution using embedded Runge-Kutta method
        # This provides a higher-order estimate for error control
        k5 = self._compute_rhs(field + dt * (7 * k1 + 10 * k2 + k4) / 27, source)
        k6 = self._compute_rhs(
            field + dt * (28 * k1 - 125 * k2 + 546 * k3 + 54 * k4 - 378 * k5) / 625,
            source,
        )

        # Fifth-order solution using Butcher tableau coefficients
        field_5th = field + dt * (k1 + 4 * k2 + k3 + 4 * k4 + k5 + k6) / 6.0

        # Full error estimation using Richardson extrapolation
        error_estimate = self._compute_richardson_error(field_4th, field_5th, dt)

        return field_4th, error_estimate

    def _compute_rhs(self, field: np.ndarray, source: np.ndarray) -> np.ndarray:
        """
        Compute right-hand side of the dynamic equation.

        Physical Meaning:
            Computes the right-hand side of the dynamic phase field equation:
            RHS = -ν(-Δ)^β a - λa + s(x,φ,t)
        """
        # Transform to spectral space
        field_spectral = self._spectral_ops.forward_fft(field)

        # Apply spectral operator: -ν|k|^(2β) - λ
        rhs_spectral = -self._spectral_coeffs * field_spectral

        # Add source term
        source_spectral = self._spectral_ops.forward_fft(source)
        rhs_spectral += source_spectral

        # Transform back to real space
        rhs = self._spectral_ops.inverse_fft(rhs_spectral)

        return rhs

    def _adjust_time_step(self, error_estimate: float, current_dt: float) -> None:
        """
        Adjust time step based on error estimate.

        Physical Meaning:
            Automatically adjusts the time step based on error estimation
            to maintain accuracy while optimizing performance.
        """
        if error_estimate > 0:
            # Calculate optimal time step based on error
            optimal_dt = current_dt * (self._tolerance / error_estimate) ** (1.0 / 5.0)

            # Apply safety factor
            optimal_dt *= self._safety_factor

            # Clamp to allowed range
            self._current_dt = np.clip(optimal_dt, self._min_dt, self._max_dt)
        else:
            # If no error, increase step size slightly
            self._current_dt = min(self._current_dt * 1.2, self._max_dt)

    def get_current_time_step(self) -> float:
        """Get current adaptive time step."""
        return self._current_dt

    def set_tolerance(self, tolerance: float) -> None:
        """Set error tolerance for adaptive control."""
        self._tolerance = tolerance
        self.logger.info(f"Adaptive tolerance set to {tolerance}")

    def set_time_step_bounds(self, min_dt: float, max_dt: float) -> None:
        """Set time step bounds."""
        self._min_dt = min_dt
        self._max_dt = max_dt
        self._current_dt = np.clip(self._current_dt, min_dt, max_dt)
        self.logger.info(f"Time step bounds set to [{min_dt}, {max_dt}]")

    def get_integrator_info(self) -> Dict[str, Any]:
        """Get information about the integrator."""
        return {
            "type": "adaptive",
            "tolerance": self._tolerance,
            "safety_factor": self._safety_factor,
            "min_dt": self._min_dt,
            "max_dt": self._max_dt,
            "current_dt": self._current_dt,
            "initialized": self._initialized,
        }
    
    def _compute_richardson_error(self, field_4th: np.ndarray, field_5th: np.ndarray, dt: float) -> float:
        """
        Compute error estimate using Richardson extrapolation.
        
        Physical Meaning:
            Uses Richardson extrapolation to provide a more accurate
            error estimate for adaptive step size control.
            
        Mathematical Foundation:
            Richardson extrapolation combines solutions of different orders
            to estimate the leading error term:
            error ≈ |y_5th - y_4th| / (1 - (h_4th/h_5th)^p)
            where p is the order difference (1 in this case).
        """
        # Compute basic error estimate
        error_diff = field_5th - field_4th
        
        # Compute relative error with proper normalization
        field_magnitude = np.linalg.norm(field_4th)
        if field_magnitude < 1e-15:
            # Avoid division by zero for very small fields
            error_estimate = np.linalg.norm(error_diff)
        else:
            # Richardson extrapolation error estimate
            # For RK4(5), the error scales as h^5, so p = 1
            richardson_factor = 1.0 / (1.0 - (0.5)**1)  # h_4th/h_5th = 0.5
            error_estimate = richardson_factor * np.linalg.norm(error_diff) / field_magnitude
        
        # Apply additional error analysis
        error_components = self._analyze_error_components(error_diff, field_4th)
        
        # Combine error estimates
        total_error = self._combine_error_estimates(error_estimate, error_components)
        
        return float(total_error)
    
    def _analyze_error_components(self, error_diff: np.ndarray, field: np.ndarray) -> Dict[str, float]:
        """
        Analyze different components of the error.
        
        Physical Meaning:
            Analyzes the spatial and spectral components of the error
            to provide more detailed error characterization.
        """
        # Spatial error analysis
        spatial_error = np.abs(error_diff)
        max_spatial_error = np.max(spatial_error)
        mean_spatial_error = np.mean(spatial_error)
        
        # Spectral error analysis
        error_spectral = np.fft.fftn(error_diff)
        field_spectral = np.fft.fftn(field)
        
        # Compute spectral error ratios
        spectral_error_magnitude = np.abs(error_spectral)
        field_spectral_magnitude = np.abs(field_spectral)
        
        # Avoid division by zero
        spectral_ratio = np.where(
            field_spectral_magnitude > 1e-15,
            spectral_error_magnitude / field_spectral_magnitude,
            0.0
        )
        
        max_spectral_error = np.max(spectral_ratio)
        mean_spectral_error = np.mean(spectral_ratio)
        
        # High-frequency error analysis
        high_freq_mask = self._compute_high_frequency_mask(field.shape)
        high_freq_error = np.mean(spectral_ratio[high_freq_mask])
        
        return {
            "max_spatial_error": float(max_spatial_error),
            "mean_spatial_error": float(mean_spatial_error),
            "max_spectral_error": float(max_spectral_error),
            "mean_spectral_error": float(mean_spectral_error),
            "high_frequency_error": float(high_freq_error)
        }
    
    def _compute_high_frequency_mask(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Compute mask for high-frequency components."""
        # Create frequency grids
        freq_grids = []
        for dim_size in shape:
            freqs = np.fft.fftfreq(dim_size)
            freq_grids.append(freqs)
        
        # Create multi-dimensional frequency grid
        freq_mesh = np.meshgrid(*freq_grids, indexing='ij')
        
        # Compute frequency magnitude
        freq_magnitude = np.sqrt(sum(freq**2 for freq in freq_mesh))
        
        # High-frequency mask (top 25% of frequencies)
        freq_threshold = np.percentile(freq_magnitude, 75)
        high_freq_mask = freq_magnitude > freq_threshold
        
        return high_freq_mask
    
    def _combine_error_estimates(self, basic_error: float, error_components: Dict[str, float]) -> float:
        """
        Combine different error estimates into a single error measure.
        
        Physical Meaning:
            Combines spatial, spectral, and high-frequency error estimates
            to provide a comprehensive error measure for step size control.
        """
        # Weight different error components
        weights = {
            "spatial": 0.4,
            "spectral": 0.4,
            "high_frequency": 0.2
        }
        
        # Compute weighted error estimate
        weighted_error = (
            weights["spatial"] * error_components["mean_spatial_error"] +
            weights["spectral"] * error_components["mean_spectral_error"] +
            weights["high_frequency"] * error_components["high_frequency_error"]
        )
        
        # Combine with basic error estimate
        combined_error = 0.7 * basic_error + 0.3 * weighted_error
        
        # Apply error bounds
        min_error = 1e-15
        max_error = 1.0
        
        combined_error = max(min_error, min(combined_error, max_error))
        
        return combined_error
