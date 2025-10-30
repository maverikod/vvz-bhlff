"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

U(1)³ phase vector structure for BVP.

This module implements the main PhaseVector class that coordinates
the three U(1) phase components and electroweak coupling for the
Base High-Frequency Field.

Physical Meaning:
    Implements the three-component phase vector Θ_a (a=1..3)
    that represents the fundamental phase structure of the BVP field.
    Each component corresponds to a different U(1) symmetry group,
    and together they form the U(1)³ structure required by the theory.

Mathematical Foundation:
    The phase vector Θ = (Θ₁, Θ₂, Θ₃) represents three independent
    U(1) phase degrees of freedom. The BVP field is constructed as:
    a(x) = |A(x)| * exp(i * Θ(x))
    where Θ(x) = Σ_a Θ_a(x) * e_a and e_a are the basis vectors.

Example:
    >>> phase_vector = PhaseVector(domain, config)
    >>> theta_components = phase_vector.get_phase_components()
    >>> electroweak_currents = phase_vector.compute_electroweak_currents(envelope)
"""

import numpy as np
from typing import Dict, Any, List
import logging

from bhlff.core.domain import Domain
from bhlff.core.bvp.bvp_constants import BVPConstants
from .phase_components import PhaseComponents
from .electroweak_coupling import ElectroweakCoupling

# CUDA optimization
try:
    import cupy as cp

    CUDA_AVAILABLE = True
    logging.info("CUDA support enabled with CuPy")
except ImportError:
    CUDA_AVAILABLE = False
    logging.warning("CUDA not available, falling back to CPU")

# Memory monitoring
try:
    from bhlff.utils.memory_monitor import MemoryMonitor, memory_monitor_context

    MEMORY_MONITORING_AVAILABLE = True
except ImportError:
    MEMORY_MONITORING_AVAILABLE = False
    logging.warning("Memory monitoring not available")


class PhaseVector:
    """
    U(1)³ phase vector structure for BVP.

    Physical Meaning:
        Implements the three-component phase vector Θ_a (a=1..3)
        that represents the fundamental phase structure of the BVP field.
        Each component corresponds to a different U(1) symmetry group.

    Mathematical Foundation:
        The phase vector Θ = (Θ₁, Θ₂, Θ₃) represents three independent
        U(1) phase degrees of freedom with weak hierarchical coupling
        to SU(2)/core through invariant mixed terms.

    Attributes:
        domain (Domain): Computational domain.
        config (Dict[str, Any]): Phase vector configuration.
        constants (BVPConstants): BVP constants instance.
        _phase_components (PhaseComponents): Phase components manager.
        _electroweak_coupling (ElectroweakCoupling): Electroweak coupling.
        coupling_matrix (np.ndarray): SU(2) coupling matrix.
    """

    def __init__(
        self, domain: Domain, config: Dict[str, Any], constants: BVPConstants = None
    ) -> None:
        """
        Initialize U(1)³ phase vector structure.

        Physical Meaning:
            Sets up the three-component phase vector Θ_a (a=1..3)
            with proper U(1)³ structure and weak SU(2) coupling.

        Args:
            domain (Domain): Computational domain.
            config (Dict[str, Any]): Phase vector configuration including:
                - phase_amplitudes: Amplitudes for each phase component
                - phase_frequencies: Frequencies for each phase component
                - su2_coupling_strength: Strength of SU(2) coupling
                - electroweak_coefficients: Electroweak coupling parameters
            constants (BVPConstants, optional): BVP constants instance.
        """
        self.domain = domain
        self.config = config
        self.constants = constants or BVPConstants(config)

        # CUDA optimization setup
        self.cuda_available = CUDA_AVAILABLE
        self.use_cuda = config.get("use_cuda", True) and self.cuda_available
        self.logger = logging.getLogger(__name__)

        # Memory monitoring setup
        self.memory_monitoring_available = MEMORY_MONITORING_AVAILABLE
        self.enable_memory_monitoring = config.get("enable_memory_monitoring", True)
        self.memory_monitor = None

        if self.memory_monitoring_available and self.enable_memory_monitoring:
            self.memory_monitor = MemoryMonitor(
                log_interval=0.5
            )  # Monitor every 0.5 seconds
            self.logger.info("PhaseVector: Memory monitoring enabled")

        if self.use_cuda:
            self.logger.info("PhaseVector: CUDA optimization enabled")
        else:
            self.logger.info("PhaseVector: Using CPU computation")

        # Initialize components
        self._phase_components = PhaseComponents(domain, config)
        self._electroweak_coupling = ElectroweakCoupling(config)

        # Setup SU(2) coupling
        self._setup_su2_coupling()

    def _setup_su2_coupling(self) -> None:
        """
        Setup weak hierarchical coupling to SU(2)/core.

        Physical Meaning:
            Establishes the weak hierarchical coupling between
            the U(1)³ phase structure and SU(2)/core through
            invariant mixed terms.
        """
        su2_config = self.config.get("su2_coupling", {})
        coupling_strength = su2_config.get("coupling_strength", 0.1)

        # Create SU(2) coupling matrix (weak coupling)
        # This represents the invariant mixed terms between U(1)³ and SU(2)
        self.coupling_matrix = np.array(
            [
                [1.0, coupling_strength, 0.0],
                [coupling_strength, 1.0, coupling_strength],
                [0.0, coupling_strength, 1.0],
            ],
            dtype=complex,
        )

        # Add weak coupling terms
        self.su2_coupling_terms = {
            "theta_1_theta_2": coupling_strength * 0.1,
            "theta_2_theta_3": coupling_strength * 0.1,
            "theta_1_theta_3": coupling_strength * 0.05,  # Weaker coupling
        }

    def get_phase_components(self) -> List[np.ndarray]:
        """
        Get the three U(1) phase components Θ_a (a=1..3).

        Physical Meaning:
            Returns the three independent U(1) phase components
            that form the U(1)³ structure.

        Returns:
            List[np.ndarray]: List of three phase components Θ_a.
        """
        components = self._phase_components.get_components()

        # Convert to CPU if using CUDA
        if self.use_cuda:
            return [self._to_cpu(comp) for comp in components]
        return components

    def get_total_phase(self) -> np.ndarray:
        """
        Get the total phase from U(1)³ structure.

        Physical Meaning:
            Computes the total phase by combining the three
            U(1) components with proper SU(2) coupling.

        Mathematical Foundation:
            Θ_total = Σ_a Θ_a + Σ_{a,b} g_{ab} Θ_a Θ_b
            where g_{ab} are the SU(2) coupling coefficients.

        Returns:
            np.ndarray: Total phase field.
        """
        return self._phase_components.get_total_phase(self.coupling_matrix)

    def compute_electroweak_currents(
        self, envelope: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute electroweak currents as functionals of the envelope.

        Physical Meaning:
            Computes electromagnetic and weak currents that are
            generated as functionals of the BVP envelope through
            the U(1)³ phase structure.

        Mathematical Foundation:
            J_EM = g_EM * |A|² * ∇Θ_EM
            J_weak = g_weak * |A|⁴ * ∇Θ_weak
            where Θ_EM and Θ_weak are combinations of Θ_a components.

        Args:
            envelope (np.ndarray): BVP envelope |A|.

        Returns:
            Dict[str, np.ndarray]: Electroweak currents including:
                - em_current: Electromagnetic current
                - weak_current: Weak interaction current
                - mixed_current: Mixed electroweak current
        """
        # Check memory usage before computation
        self._check_memory_usage("electroweak_currents_start")

        try:
            phase_components = self._phase_components.get_components()
            result = self._electroweak_coupling.compute_electroweak_currents(
                envelope, phase_components, self.domain
            )

            # Check memory usage after computation
            self._check_memory_usage("electroweak_currents_end")

            return result
        except Exception as e:
            self.logger.error(f"Error in electroweak currents computation: {e}")
            # Force memory cleanup on error
            self.force_memory_cleanup()
            raise

    def compute_phase_coherence(self) -> np.ndarray:
        """
        Compute phase coherence measure.

        Physical Meaning:
            Computes a measure of phase coherence across the
            U(1)³ structure, indicating the degree of
            synchronization between the three phase components.

        Mathematical Foundation:
            Coherence = |Σ_a exp(iΘ_a)| / 3
            where the magnitude indicates coherence strength.

        Returns:
            np.ndarray: Phase coherence measure.
        """
        return self._phase_components.compute_phase_coherence()

    def get_su2_coupling_strength(self) -> float:
        """
        Get the SU(2) coupling strength.

        Physical Meaning:
            Returns the strength of the weak hierarchical
            coupling to SU(2)/core.

        Returns:
            float: SU(2) coupling strength.
        """
        return np.abs(self.coupling_matrix[0, 1])  # Off-diagonal element

    def set_su2_coupling_strength(self, strength: float) -> None:
        """
        Set the SU(2) coupling strength.

        Physical Meaning:
            Updates the strength of the weak hierarchical
            coupling to SU(2)/core.

        Args:
            strength (float): New SU(2) coupling strength.
        """
        # Update coupling matrix
        self.coupling_matrix[0, 1] = strength
        self.coupling_matrix[1, 0] = strength
        self.coupling_matrix[1, 2] = strength
        self.coupling_matrix[2, 1] = strength

        # Update coupling terms
        self.su2_coupling_terms["theta_1_theta_2"] = strength * 0.1
        self.su2_coupling_terms["theta_2_theta_3"] = strength * 0.1
        self.su2_coupling_terms["theta_1_theta_3"] = strength * 0.05

    def update_phase_components(self, envelope: np.ndarray) -> None:
        """
        Update phase components from solved envelope.

        Physical Meaning:
            Updates the three U(1) phase components Θ_a (a=1..3)
            from the solved BVP envelope field.

        Mathematical Foundation:
            Extracts phase components from the envelope solution
            and updates the U(1)³ phase structure.

        Args:
            envelope (np.ndarray): Solved BVP envelope in 7D space-time.
        """
        self._phase_components.update_components(envelope)

    def get_electroweak_coefficients(self) -> Dict[str, float]:
        """
        Get electroweak coupling coefficients.

        Physical Meaning:
            Returns the current electroweak coupling coefficients
            used for current calculations.

        Returns:
            Dict[str, float]: Electroweak coupling coefficients.
        """
        return self._electroweak_coupling.get_electroweak_coefficients()

    def set_electroweak_coefficients(self, coefficients: Dict[str, float]) -> None:
        """
        Set electroweak coupling coefficients.

        Physical Meaning:
            Updates the electroweak coupling coefficients
            used for current calculations.

        Args:
            coefficients (Dict[str, float]): New coupling coefficients.
        """
        self._electroweak_coupling.set_electroweak_coefficients(coefficients)

    def decompose_phase_structure(self, envelope: np.ndarray = None) -> tuple:
        """
        Decompose phase structure into amplitude and phases.

        Physical Meaning:
            Decomposes the BVP field into amplitude and three phase components
            according to the U(1)³ structure: a = |a|e^(iφ₁)e^(iφ₂)e^(iφ₃)

        Mathematical Foundation:
            Extracts amplitude |a| and phases φ₁, φ₂, φ₃ from the field
            such that a = |a| * exp(i * (φ₁ + φ₂ + φ₃))

        Args:
            envelope (np.ndarray, optional): BVP envelope field. If None, uses current phase components.

        Returns:
            tuple: (amplitude, phases) where:
                - amplitude: Field amplitude |a|
                - phases: List of three phase components [φ₁, φ₂, φ₃]
        """
        if envelope is not None:
            # Extract amplitude from envelope
            amplitude = np.abs(envelope)

            # Extract phases from envelope
            total_phase = np.angle(envelope)

            # Distribute total phase among three U(1) components
            phases = []
            for a in range(3):
                # Each component gets a portion of the total phase
                phase_portion = total_phase / 3.0 + 2 * np.pi * a / 3.0
                phases.append(phase_portion)
        else:
            # Use current phase components
            phase_components = self._phase_components.get_components()
            amplitude = np.abs(phase_components[0])  # Use first component for amplitude

            # Extract phases from components
            phases = []
            for theta_a in phase_components:
                phases.append(np.angle(theta_a))

        return amplitude, phases

    def compute_topological_charge(self, envelope: np.ndarray = None) -> float:
        """
        Compute topological charge of the phase structure.

        Physical Meaning:
            Computes the topological charge (winding number) of the U(1)³
            phase structure, which is quantized according to the theory.

        Mathematical Foundation:
            Topological charge = (1/2π) ∮ ∇φ · dl
            where φ is the total phase and the integral is over a closed loop.

        Args:
            envelope (np.ndarray, optional): BVP envelope field. If None, uses current phase components.

        Returns:
            float: Topological charge (should be quantized).
        """
        # Check memory usage before computation
        self._check_memory_usage("topological_charge_start")

        try:
            if envelope is not None:
                # Extract total phase from envelope
                envelope_gpu = self._to_gpu(envelope)
                total_phase = self._cuda_angle(envelope_gpu)
            else:
                # Use current phase components
                total_phase = self._phase_components.get_total_phase()
                total_phase = self._to_gpu(total_phase)

            # Compute topological charge using gradient
            if self.domain.dimensions == 1:
                # 1D case
                phase_gradient = self._cuda_gradient(total_phase)
                topological_charge = self._cuda_sum(phase_gradient) / (2 * np.pi)
            elif self.domain.dimensions == 2:
                # 2D case - use line integral around boundary
                phase_gradient_x = self._cuda_gradient(total_phase, axis=0)
                phase_gradient_y = self._cuda_gradient(total_phase, axis=1)

                # Compute line integral around boundary
                boundary_integral = 0.0
                # Top boundary
                boundary_integral += self._cuda_sum(phase_gradient_x[0, :])
                # Right boundary
                boundary_integral += self._cuda_sum(phase_gradient_y[:, -1])
                # Bottom boundary
                boundary_integral += self._cuda_sum(-phase_gradient_x[-1, :])
                # Left boundary
                boundary_integral += self._cuda_sum(-phase_gradient_y[:, 0])

                topological_charge = boundary_integral / (2 * np.pi)
            else:
                # 3D case - use surface integral
                phase_gradient_x = self._cuda_gradient(total_phase, axis=0)
                phase_gradient_y = self._cuda_gradient(total_phase, axis=1)
                phase_gradient_z = self._cuda_gradient(total_phase, axis=2)

                # Compute surface integral (simplified)
                surface_integral = self._cuda_sum(
                    phase_gradient_x + phase_gradient_y + phase_gradient_z
                )
                topological_charge = surface_integral / (2 * np.pi)

            # Check memory usage after computation
            self._check_memory_usage("topological_charge_end")

            # Convert to CPU and return scalar
            return float(self._to_cpu(topological_charge))

        except Exception as e:
            self.logger.error(f"Error in topological charge computation: {e}")
            # Force memory cleanup on error
            self.force_memory_cleanup()
            raise

    def compute_phase_coherence(self, envelope: np.ndarray = None) -> float:
        """
        Compute phase coherence measure.

        Physical Meaning:
            Computes a measure of phase coherence across the
            U(1)³ structure, indicating the degree of
            synchronization between the three phase components.

        Mathematical Foundation:
            Coherence = |Σ_a exp(iΘ_a)| / 3
            where the magnitude indicates coherence strength.

        Args:
            envelope (np.ndarray, optional): BVP envelope field. If None, uses current phase components.

        Returns:
            float: Phase coherence measure (0-1).
        """
        if envelope is not None:
            # Extract phases from envelope
            envelope_gpu = self._to_gpu(envelope)
            total_phase = self._cuda_angle(envelope_gpu)

            # Compute coherence from total phase
            coherence_sum = self._cuda_exp(1j * total_phase)
            coherence = self._cuda_abs(self._cuda_mean(coherence_sum))
        else:
            # Use current phase components
            coherence = self._phase_components.compute_phase_coherence()
            coherence_gpu = self._to_gpu(coherence)
            coherence = self._cuda_mean(coherence_gpu)  # Average over spatial points

        # Convert to CPU and return scalar
        return float(self._to_cpu(coherence))

    def _to_gpu(self, array: np.ndarray) -> "cp.ndarray":
        """
        Convert numpy array to GPU array.

        Physical Meaning:
            Transfers array to GPU memory for CUDA computation.

        Args:
            array (np.ndarray): Input array.

        Returns:
            cp.ndarray: GPU array.
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return cp.asarray(array)
        return array

    def _to_cpu(self, array) -> np.ndarray:
        """
        Convert GPU array to numpy array.

        Physical Meaning:
            Transfers array from GPU memory to CPU memory.

        Args:
            array: Input array (GPU or CPU).

        Returns:
            np.ndarray: CPU array.
        """
        if self.use_cuda and CUDA_AVAILABLE and hasattr(array, "get"):
            return array.get()
        return array

    def _cuda_gradient(self, array, axis: int = 0) -> "cp.ndarray":
        """
        Compute gradient using CUDA.

        Physical Meaning:
            Computes gradient using CUDA for optimal performance.

        Args:
            array: Input array.
            axis (int): Axis along which to compute gradient.

        Returns:
            cp.ndarray: Gradient array.
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return cp.gradient(array, axis=axis)
        return np.gradient(array, axis=axis)

    def _cuda_abs(self, array) -> "cp.ndarray":
        """
        Compute absolute value using CUDA.

        Physical Meaning:
            Computes absolute value using CUDA for optimal performance.

        Args:
            array: Input array.

        Returns:
            cp.ndarray: Absolute value array.
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return cp.abs(array)
        return np.abs(array)

    def _cuda_angle(self, array) -> "cp.ndarray":
        """
        Compute angle using CUDA.

        Physical Meaning:
            Computes angle using CUDA for optimal performance.

        Args:
            array: Input array.

        Returns:
            cp.ndarray: Angle array.
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return cp.angle(array)
        return np.angle(array)

    def _cuda_exp(self, array) -> "cp.ndarray":
        """
        Compute exponential using CUDA.

        Physical Meaning:
            Computes exponential using CUDA for optimal performance.

        Args:
            array: Input array.

        Returns:
            cp.ndarray: Exponential array.
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return cp.exp(array)
        return np.exp(array)

    def _cuda_sum(self, array, axis=None) -> "cp.ndarray":
        """
        Compute sum using CUDA.

        Physical Meaning:
            Computes sum using CUDA for optimal performance.

        Args:
            array: Input array.
            axis: Axis along which to sum.

        Returns:
            cp.ndarray: Sum array.
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return cp.sum(array, axis=axis)
        return np.sum(array, axis=axis)

    def _cuda_mean(self, array, axis=None) -> "cp.ndarray":
        """
        Compute mean using CUDA.

        Physical Meaning:
            Computes mean using CUDA for optimal performance.

        Args:
            array: Input array.
            axis: Axis along which to compute mean.

        Returns:
            cp.ndarray: Mean array.
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return cp.mean(array, axis=axis)
        return np.mean(array, axis=axis)

    def _cuda_sqrt(self, array) -> "cp.ndarray":
        """
        Compute square root using CUDA.

        Physical Meaning:
            Computes square root using CUDA for optimal performance.

        Args:
            array: Input array.

        Returns:
            cp.ndarray: Square root array.
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return cp.sqrt(array)
        return np.sqrt(array)

    def _cuda_sin(self, array) -> "cp.ndarray":
        """
        Compute sine using CUDA.

        Physical Meaning:
            Computes sine using CUDA for optimal performance.

        Args:
            array: Input array.

        Returns:
            cp.ndarray: Sine array.
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return cp.sin(array)
        return np.sin(array)

    def _cuda_cos(self, array) -> "cp.ndarray":
        """
        Compute cosine using CUDA.

        Physical Meaning:
            Computes cosine using CUDA for optimal performance.

        Args:
            array: Input array.

        Returns:
            cp.ndarray: Cosine array.
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return cp.cos(array)
        return np.cos(array)

    def _check_memory_usage(self, operation_name: str = "operation") -> None:
        """
        Check memory usage and log warnings if needed.

        Physical Meaning:
            Monitors memory usage during computations
            and provides warnings if thresholds are exceeded.

        Args:
            operation_name (str): Name of the operation being performed.
        """
        if not self.memory_monitor:
            return

        try:
            stats = self.memory_monitor.get_memory_stats()

            # Log memory usage
            cpu_used = stats["cpu"]["used_mb"]
            gpu_used = stats["gpu"]["used_mb"] if stats["gpu"] else 0

            self.logger.debug(
                f"{operation_name}: CPU memory {cpu_used:.1f}MB, GPU memory {gpu_used:.1f}MB"
            )

            # Check for warnings
            if "warnings" in stats:
                for warning in stats["warnings"]:
                    self.logger.warning(f"{operation_name}: {warning}")

        except Exception as e:
            self.logger.warning(
                f"Failed to check memory usage for {operation_name}: {e}"
            )

    def start_memory_monitoring(self) -> None:
        """
        Start memory monitoring.

        Physical Meaning:
            Starts continuous monitoring of memory usage
            during phase vector computations.
        """
        if self.memory_monitor:
            self.memory_monitor.start_monitoring()
            self.logger.info("Memory monitoring started")

    def stop_memory_monitoring(self) -> None:
        """
        Stop memory monitoring.

        Physical Meaning:
            Stops continuous monitoring of memory usage.
        """
        if self.memory_monitor:
            self.memory_monitor.stop_monitoring()
            self.logger.info("Memory monitoring stopped")

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current memory statistics.

        Physical Meaning:
            Returns current memory usage statistics
            for both CPU and GPU memory.

        Returns:
            Dict[str, Any]: Memory statistics.
        """
        if self.memory_monitor:
            return self.memory_monitor.get_memory_stats()
        return {}

    def force_memory_cleanup(self) -> None:
        """
        Force memory cleanup.

        Physical Meaning:
            Forces garbage collection and memory cleanup
            to optimize memory usage.
        """
        if self.memory_monitor:
            self.memory_monitor.force_garbage_collection()
            self.logger.info("Memory cleanup completed")

    def __repr__(self) -> str:
        """String representation of phase vector."""
        coupling_strength = self.get_su2_coupling_strength()
        em_coupling = self.get_electroweak_coefficients()["em_coupling"]
        cuda_status = "CUDA" if self.use_cuda else "CPU"
        return (
            f"PhaseVector(domain={self.domain}, "
            f"su2_coupling={coupling_strength:.3f}, "
            f"em_coupling={em_coupling:.3f}, "
            f"compute={cuda_status})"
        )
