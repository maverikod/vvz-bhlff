"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Quench Detector implementation according to step 00 specification.

This module implements the detector for quench events in BVP,
monitoring local thresholds and detecting when BVP dissipatively
"dumps" energy into the medium.

Theoretical Background:
    Quenches represent threshold events in the BVP field where
    local thresholds (amplitude/detuning/gradient) are reached,
    causing the BVP to dissipatively "dump" energy into the medium.
    This results in a local regime transition with increased losses
    and Q-factor changes.

Example:
    >>> detector = QuenchDetector(domain_7d, config)
    >>> quenches = detector.detect_quenches(envelope)
    >>> print(f"Quenches detected: {quenches['quenches_detected']}")
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging

try:
    import cupy as cp

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

    # Create a dummy class for type hints when CUDA is not available
    class DummyCuPy:
        class ndarray:
            pass

    cp = DummyCuPy()

from ..domain.domain_7d import Domain7D
from .quench_thresholds import QuenchThresholdComputer
from .quench_morphology import QuenchMorphology
from .quench_characteristics import QuenchCharacteristics


class QuenchDetector:
    """
    Detector for quench events in BVP.

    Physical Meaning:
        Monitors local thresholds (amplitude/detuning/gradient)
        and detects when BVP dissipatively "dumps" energy into
        the medium. Quenches represent threshold events where
        the BVP field undergoes a local regime transition.

    Mathematical Foundation:
        Applies three threshold criteria for quench detection:
        1. Amplitude threshold: |A| > |A_q|
        2. Detuning threshold: |ω - ω_0| > Δω_q
        3. Gradient threshold: |∇A| > |∇A_q|
        where A_q, Δω_q, and ∇A_q are the quench thresholds.
    """

    def __init__(self, domain_7d: Domain7D, config: Dict[str, Any]):
        """
        Initialize quench detector.

        Physical Meaning:
            Sets up the quench detector with threshold parameters
            for detecting amplitude, detuning, and gradient quenches
            in the 7D BVP field.

        Args:
            domain_7d (Domain7D): 7D computational domain.
            config (Dict[str, Any]): Configuration parameters including:
                - amplitude_threshold (float): Amplitude quench threshold |A_q|
                - detuning_threshold (float): Detuning quench threshold Δω_q
                - gradient_threshold (float): Gradient quench threshold |∇A_q|
                - carrier_frequency (float): BVP carrier frequency ω₀
        """
        self.domain_7d = domain_7d
        self.config = config

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Check CUDA availability and setup
        self.cuda_available = CUDA_AVAILABLE and config.get("use_cuda", True)
        if self.cuda_available:
            self.logger.info(
                "CUDA available - using GPU acceleration for quench detection"
            )
        else:
            self.logger.info(
                "CUDA not available - using CPU processing for quench detection"
            )

        # Initialize threshold computer, morphology processor, and characteristics computer
        self.threshold_computer = QuenchThresholdComputer(domain_7d)
        self.morphology = QuenchMorphology()
        self.characteristics = QuenchCharacteristics(domain_7d)

        # Block processing configuration
        self.block_size = int(config.get("block_size", 0))  # 0 disables blocked mode
        self.overlap = int(config.get("overlap", 2))
        self.batch_size = int(config.get("batch_size", 1))
        self.verbose = bool(config.get("verbose", True))
        self.progress_interval = int(config.get("progress_interval", 10))

        # Compute physical thresholds from theoretical principles
        thresholds = self.threshold_computer.compute_all_thresholds()
        self.amplitude_threshold = thresholds["amplitude_threshold"]
        self.detuning_threshold = thresholds["detuning_threshold"]
        self.gradient_threshold = thresholds["gradient_threshold"]
        self.carrier_frequency = thresholds["carrier_frequency"]

        # Override with config values if provided (for testing/debugging)
        if "amplitude_threshold" in config:
            self.amplitude_threshold = config["amplitude_threshold"]
        if "detuning_threshold" in config:
            self.detuning_threshold = config["detuning_threshold"]
        if "gradient_threshold" in config:
            self.gradient_threshold = config["gradient_threshold"]
        if "carrier_frequency" in config:
            self.carrier_frequency = config["carrier_frequency"]

        # Setup threshold validation
        self._validate_thresholds()

        # Auto-configure block size based on GPU memory if not specified
        if self.block_size == 0 and self.cuda_available:
            self.block_size = self._compute_optimal_block_size_from_gpu_memory()

        # Verbose logging level
        if self.verbose:
            self.logger.setLevel(logging.INFO)

    def _compute_optimal_block_size_from_gpu_memory(self) -> int:
        """
        Compute optimal block size based on available GPU memory.

        Physical Meaning:
            Calculates the maximum block size that can fit in 80% of available
            GPU memory, ensuring efficient memory usage while avoiding OOM.

        Returns:
            int: Optimal block size for 7D processing.
        """
        if not self.cuda_available:
            return 8  # Default small size for CPU

        try:
            # Get GPU memory info
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            available_mem = int(free_mem * 0.8)  # Use 80% of free memory

            # Estimate memory per element (complex128 = 16 bytes)
            bytes_per_element = 16

            # For 7D array, we need space for:
            # - Input field
            # - Amplitude computation
            # - Gradient computation (7 gradients)
            # - Morphology operations
            # - Connected components
            # Total overhead factor ~10x
            overhead_factor = 10

            # Calculate maximum elements per block
            max_elements = available_mem // (bytes_per_element * overhead_factor)

            # For 7D, calculate block size per dimension
            # Assuming roughly equal dimensions
            elements_per_dim = int(max_elements ** (1 / 7))

            # Ensure reasonable bounds
            block_size = max(4, min(elements_per_dim, 64))  # Between 4 and 64

            self.logger.info(
                f"GPU memory: {free_mem/1e9:.2f}GB free, {total_mem/1e9:.2f}GB total"
            )
            self.logger.info(
                f"Optimal block size: {block_size} (using 80% of free memory)"
            )

            return block_size

        except Exception as e:
            self.logger.warning(
                f"Failed to compute optimal block size: {e}, using default 8"
            )
            return 8

    def detect_quenches(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Detect quench events based on three thresholds.

        Physical Meaning:
            Applies three threshold criteria to detect quench events:
            - amplitude: |A| > |A_q| - detects high-amplitude quenches
            - detuning: |ω - ω_0| > Δω_q - detects frequency detuning quenches
            - gradient: |∇A| > |∇A_q| - detects high-gradient quenches

        Mathematical Foundation:
            For each point in 7D space-time, checks:
            1. |A(x,φ,t)| > |A_q|
            2. |ω_local - ω_0| > Δω_q
            3. |∇A(x,φ,t)| > |∇A_q|
            where ω_local is the local frequency derived from phase evolution.

        Args:
            envelope (np.ndarray): 7D envelope field with shape
                (N_x, N_y, N_z, N_φ₁, N_φ₂, N_φ₃, N_t)

        Returns:
            Dict[str, Any]: Quench detection results including:
                - quenches_detected (bool): Whether any quenches were found
                - quench_locations (List[Tuple]): 7D coordinates of quench events
                - quench_types (List[str]): Types of quenches detected
                - quench_strengths (List[float]): Strength of each quench
                - amplitude_quenches (List[Tuple]): Amplitude quench locations
                - detuning_quenches (List[Tuple]): Detuning quench locations
                - gradient_quenches (List[Tuple]): Gradient quench locations
        """
        # Prefer blocked processing when configured to avoid OOM
        if self.block_size and self.block_size > 0:
            self.logger.info(
                f"Blocked detection enabled: block_size={self.block_size}, overlap={self.overlap}, batch_size={self.batch_size}, cuda={self.cuda_available}"
            )
            return self._detect_quenches_blocked(envelope)

        # Fallback: whole-domain processing
        if self.cuda_available:
            return self._detect_quenches_cuda(envelope)
        return self._detect_quenches_cpu(envelope)

    def _iter_block_slices(self, shape: Tuple[int, ...]) -> List[Tuple[slice, ...]]:
        """Generate overlapping block slices for a 7D array."""
        sizes = [self.block_size] * len(shape)
        step = [max(1, s - self.overlap) for s in sizes]
        indices = []
        for dim, dim_len in enumerate(shape):
            pos = 0
            dim_slices = []
            while pos < dim_len:
                end = min(pos + sizes[dim], dim_len)
                start = max(0, end - sizes[dim]) if end == dim_len else pos
                dim_slices.append((start, end))
                if end == dim_len:
                    break
                pos += step[dim]
            indices.append(dim_slices)

        # Cartesian product of per-dimension slices
        from itertools import product

        blocks: List[Tuple[slice, ...]] = []
        for combo in product(*indices):
            slices = tuple(slice(s, e) for (s, e) in combo)
            blocks.append(slices)
        return blocks

    def _detect_quenches_blocked(self, envelope: np.ndarray) -> Dict[str, Any]:
        """Detect quenches using vectorized overlapped block processing (CPU/GPU)."""
        all_quenches: List[Dict[str, Any]] = []
        blocks = self._iter_block_slices(envelope.shape)
        total_blocks = len(blocks)
        self.logger.info(f"Total blocks: {total_blocks} for shape {envelope.shape}")

        import time

        start_time = time.time()

        def _gpu_mem_info() -> str:
            if self.cuda_available:
                try:
                    free_b, total_b = cp.cuda.runtime.memGetInfo()
                    used_b = total_b - free_b
                    return f"GPU mem used {used_b/1e9:.2f}GB / {total_b/1e9:.2f}GB"
                except Exception:
                    return "GPU mem n/a"
            return "CPU mode"

        # Process blocks in batches
        processed_blocks = 0
        for i in range(0, total_blocks, max(1, self.batch_size)):
            batch = blocks[i : i + self.batch_size]
            for blk_idx, blk in enumerate(batch):
                block_view = envelope[blk]
                blk_global_index = i + blk_idx + 1
                self.logger.info(
                    f"Start block {blk_global_index}/{total_blocks} slices={blk}"
                )
                blk_t0 = time.time()
                try:
                    if self.cuda_available:
                        # GPU path per block (single transfer)
                        block_gpu = cp.asarray(block_view)
                        amp_q = self._detect_amplitude_quenches_cuda(block_gpu)
                        self.logger.info(
                            f"  block {blk_global_index}: amplitude done (n={len(amp_q)}) | {_gpu_mem_info()}"
                        )
                        det_q = self._detect_detuning_quenches_cuda(block_gpu)
                        self.logger.info(
                            f"  block {blk_global_index}: detuning done (n={len(det_q)}) | {_gpu_mem_info()}"
                        )
                        grad_q = self._detect_gradient_quenches_cuda(block_gpu)
                        self.logger.info(
                            f"  block {blk_global_index}: gradient done (n={len(grad_q)}) | {_gpu_mem_info()}"
                        )
                        del block_gpu
                        cp.get_default_memory_pool().free_all_blocks()
                    else:
                        amp_q = self._detect_amplitude_quenches(block_view)
                        self.logger.info(
                            f"  block {blk_global_index}: amplitude done (n={len(amp_q)})"
                        )
                        det_q = self._detect_detuning_quenches(block_view)
                        self.logger.info(
                            f"  block {blk_global_index}: detuning done (n={len(det_q)})"
                        )
                        grad_q = self._detect_gradient_quenches(block_view)
                        self.logger.info(
                            f"  block {blk_global_index}: gradient done (n={len(grad_q)})"
                        )
                except Exception as e:
                    # Fallback to CPU for this block
                    self.logger.warning(
                        f"Block {i+blk_idx}: CUDA path failed ({e}); falling back to CPU"
                    )
                    amp_q = self._detect_amplitude_quenches(block_view)
                    det_q = self._detect_detuning_quenches(block_view)
                    grad_q = self._detect_gradient_quenches(block_view)

                # Offset block-local centers to global coordinates
                def _offset_events(
                    events: List[Dict[str, Any]],
                ) -> List[Dict[str, Any]]:
                    start_indices = [sl.start or 0 for sl in blk]
                    adjusted: List[Dict[str, Any]] = []
                    for ev in events:
                        loc = ev.get("location")
                        if loc is not None and len(loc) == len(start_indices):
                            loc = tuple(
                                float(loc[d]) + float(start_indices[d])
                                for d in range(len(start_indices))
                            )
                        ev2 = dict(ev)
                        ev2["location"] = loc
                        adjusted.append(ev2)
                    return adjusted

                all_quenches.extend(_offset_events(amp_q))
                all_quenches.extend(_offset_events(det_q))
                all_quenches.extend(_offset_events(grad_q))

                processed_blocks += 1
                blk_dt = time.time() - blk_t0
                self.logger.info(
                    f"End block {blk_global_index}: took {blk_dt:.2f}s, total events={len(amp_q)+len(det_q)+len(grad_q)}"
                )
                if (
                    processed_blocks % max(1, self.progress_interval) == 0
                    or processed_blocks == total_blocks
                ):
                    elapsed = time.time() - start_time
                    rate = processed_blocks / max(1e-6, elapsed)
                    eta = (total_blocks - processed_blocks) / max(1e-6, rate)
                    self.logger.info(
                        f"Progress: {processed_blocks}/{total_blocks} blocks | {rate:.2f} blk/s | ETA {eta:.1f}s | {_gpu_mem_info()}"
                    )

        quench_locations = [q.get("location") for q in all_quenches]
        quench_types = [q.get("type") for q in all_quenches]
        quench_strengths = [q.get("strength", 0.0) for q in all_quenches]

        return {
            "quenches_detected": len(all_quenches) > 0,
            "quench_locations": quench_locations,
            "quench_types": quench_types,
            "quench_strengths": quench_strengths,
            "amplitude_quenches": [
                q["location"] for q in all_quenches if q.get("type") == "amplitude"
            ],
            "detuning_quenches": [
                q["location"] for q in all_quenches if q.get("type") == "detuning"
            ],
            "gradient_quenches": [
                q["location"] for q in all_quenches if q.get("type") == "gradient"
            ],
            "total_quenches": len(all_quenches),
            "detection_method": (
                "blocked_cuda_7d" if self.cuda_available else "blocked_cpu_7d"
            ),
        }

    def _detect_quenches_cuda(self, envelope: np.ndarray) -> Dict[str, Any]:
        """Detect quenches using CUDA acceleration."""
        try:
            # Transfer envelope to GPU
            envelope_gpu = cp.asarray(envelope)

            # Detect different types of quenches on GPU
            amplitude_quenches = self._detect_amplitude_quenches_cuda(envelope_gpu)
            detuning_quenches = self._detect_detuning_quenches_cuda(envelope_gpu)
            gradient_quenches = self._detect_gradient_quenches_cuda(envelope_gpu)

            # Cleanup GPU memory
            del envelope_gpu
            cp.get_default_memory_pool().free_all_blocks()

            # Force garbage collection
            import gc

            gc.collect()

            # Combine results
            all_quenches = amplitude_quenches + detuning_quenches + gradient_quenches
            quench_locations = [q["location"] for q in all_quenches]
            quench_types = [q["type"] for q in all_quenches]
            quench_strengths = [q["strength"] for q in all_quenches]

            return {
                "quenches_detected": len(all_quenches) > 0,
                "quench_locations": quench_locations,
                "quench_types": quench_types,
                "quench_strengths": quench_strengths,
                "amplitude_quenches": [q["location"] for q in amplitude_quenches],
                "detuning_quenches": [q["location"] for q in detuning_quenches],
                "gradient_quenches": [q["location"] for q in gradient_quenches],
                "total_quenches": len(all_quenches),
                "detection_method": "cuda_7d_bvp",
            }

        except Exception as e:
            self.logger.warning(
                f"CUDA quench detection failed: {e}, falling back to CPU"
            )
            import traceback

            self.logger.warning(f"CUDA error traceback: {traceback.format_exc()}")
            return self._detect_quenches_cpu(envelope)

    def _detect_quenches_cpu(self, envelope: np.ndarray) -> Dict[str, Any]:
        """Detect quenches using CPU processing."""
        # Detect different types of quenches
        amplitude_quenches = self._detect_amplitude_quenches(envelope)
        detuning_quenches = self._detect_detuning_quenches(envelope)
        gradient_quenches = self._detect_gradient_quenches(envelope)

        # Combine all quenches
        all_quenches = amplitude_quenches + detuning_quenches + gradient_quenches
        quench_locations = [q["location"] for q in all_quenches]
        quench_types = [q["type"] for q in all_quenches]
        quench_strengths = [q["strength"] for q in all_quenches]

        return {
            "quenches_detected": len(all_quenches) > 0,
            "quench_locations": quench_locations,
            "quench_types": quench_types,
            "quench_strengths": quench_strengths,
            "amplitude_quenches": [q["location"] for q in amplitude_quenches],
            "detuning_quenches": [q["location"] for q in detuning_quenches],
            "gradient_quenches": [q["location"] for q in gradient_quenches],
            "total_quenches": len(all_quenches),
        }

    def _detect_amplitude_quenches(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect amplitude quenches: |A| > |A_q| with advanced processing.

        Physical Meaning:
            Detects locations where the envelope amplitude exceeds
            the amplitude threshold, indicating potential quench events
            due to high field strength. Uses morphological operations
            to filter noise and find connected components.

        Mathematical Foundation:
            Applies morphological operations to filter noise:
            - Binary opening: removes small noise components
            - Binary closing: fills small gaps in quench regions
            - Connected component analysis: groups nearby quench events

        Args:
            envelope (np.ndarray): 7D envelope field.

        Returns:
            List[Dict[str, Any]]: List of amplitude quench events with
                enhanced characteristics including size and center of mass.
        """
        quenches = []

        # Compute amplitude
        amplitude = np.abs(envelope)

        # Find locations exceeding threshold
        quench_mask = amplitude > self.amplitude_threshold

        if np.any(quench_mask):
            # Apply morphological operations to filter noise
            quench_mask = self.morphology.apply_morphological_operations(quench_mask)

            # Find connected components
            quench_components = self.morphology.find_connected_components(quench_mask)

            # Process each component
            for component_id, component_mask in quench_components.items():
                if np.sum(component_mask) < self.config.get("min_quench_size", 5):
                    continue  # Skip small components

                # Compute component characteristics
                center = self.characteristics.compute_center_of_mass(component_mask)
                strength = self.characteristics.compute_quench_strength(
                    component_mask, amplitude
                )
                size = np.sum(component_mask)

                quenches.append(
                    {
                        "location": center,
                        "type": "amplitude",
                        "strength": float(strength),
                        "threshold": self.amplitude_threshold,
                        "size": int(size),
                        "component_id": component_id,
                    }
                )

        return quenches

    def _detect_detuning_quenches(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect detuning quenches: |ω - ω_0| > Δω_q with advanced processing.

        Physical Meaning:
            Detects locations where the local frequency deviates
            significantly from the carrier frequency, indicating
            detuning quench events. Uses advanced frequency analysis
            and morphological operations for robust detection.

        Mathematical Foundation:
            Computes local frequency using phase evolution:
            ω_local = |dφ/dt| / dt
            Detuning = |ω_local - ω_0|
            Applies same morphological operations as amplitude quenches.

        Args:
            envelope (np.ndarray): 7D envelope field.

        Returns:
            List[Dict[str, Any]]: List of detuning quench events with
                enhanced characteristics.
        """
        quenches = []

        # Compute local frequency from phase evolution
        if envelope.shape[-1] > 1:  # Need at least 2 time slices
            local_frequency = self.characteristics.compute_local_frequency(envelope)

            # Detuning from carrier frequency
            detuning = np.abs(local_frequency - self.carrier_frequency)

            # Find locations exceeding detuning threshold
            quench_mask = detuning > self.detuning_threshold

            if np.any(quench_mask):
                # Apply morphological operations to filter noise
                quench_mask = self.morphology.apply_morphological_operations(
                    quench_mask
                )

                # Find connected components
                quench_components = self.morphology.find_connected_components(
                    quench_mask
                )

                # Process each component
                for component_id, component_mask in quench_components.items():
                    if np.sum(component_mask) < self.config.get("min_quench_size", 5):
                        continue  # Skip small components

                    # Compute component characteristics
                    center = self.characteristics.compute_center_of_mass(component_mask)
                    strength = self.characteristics.compute_detuning_strength(
                        component_mask, detuning
                    )
                    size = np.sum(component_mask)

                    quenches.append(
                        {
                            "location": center,
                            "type": "detuning",
                            "strength": float(strength),
                            "threshold": self.detuning_threshold,
                            "size": int(size),
                            "component_id": component_id,
                        }
                    )

        return quenches

    def _detect_gradient_quenches(self, envelope: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect gradient quenches: |∇A| > |∇A_q| with advanced processing.

        Physical Meaning:
            Detects locations where the envelope gradient exceeds
            the gradient threshold, indicating potential quench events
            due to high spatial/phase gradients. Uses 7D gradient computation
            and morphological operations for robust detection.

        Mathematical Foundation:
            Computes 7D gradient: ∇A = (∂A/∂x, ∂A/∂y, ∂A/∂z, ∂A/∂φ₁, ∂A/∂φ₂, ∂A/∂φ₃, ∂A/∂t)
            Gradient magnitude: |∇A| = √(Σ|∂A/∂xᵢ|²)
            Applies same morphological operations as other quench types.

        Args:
            envelope (np.ndarray): 7D envelope field.

        Returns:
            List[Dict[str, Any]]: List of gradient quench events with
                enhanced characteristics.
        """
        quenches = []

        # Compute 7D gradient
        gradient_magnitude = self.characteristics.compute_7d_gradient_magnitude(
            envelope
        )

        # Find locations exceeding gradient threshold
        quench_mask = gradient_magnitude > self.gradient_threshold

        if np.any(quench_mask):
            # Apply morphological operations to filter noise
            quench_mask = self.morphology.apply_morphological_operations(quench_mask)

            # Find connected components
            quench_components = self.morphology.find_connected_components(quench_mask)

            # Process each component
            for component_id, component_mask in quench_components.items():
                if np.sum(component_mask) < self.config.get("min_quench_size", 5):
                    continue  # Skip small components

                # Compute component characteristics
                center = self.characteristics.compute_center_of_mass(component_mask)
                strength = self.characteristics.compute_gradient_strength(
                    component_mask, gradient_magnitude
                )
                size = np.sum(component_mask)

                quenches.append(
                    {
                        "location": center,
                        "type": "gradient",
                        "strength": float(strength),
                        "threshold": self.gradient_threshold,
                        "size": int(size),
                        "component_id": component_id,
                    }
                )

        return quenches

    def _detect_amplitude_quenches_cuda(self, envelope_gpu) -> List[Dict[str, Any]]:
        """Detect amplitude quenches using CUDA acceleration."""
        quenches = []

        # Compute amplitude on GPU
        amplitude = cp.abs(envelope_gpu)

        # Find locations exceeding threshold
        quench_mask = amplitude > self.amplitude_threshold

        if cp.any(quench_mask):
            # Apply morphological operations to filter noise
            quench_mask = self.morphology.apply_morphological_operations_cuda(
                quench_mask
            )

            # Find connected components
            quench_components = self.morphology.find_connected_components_cuda(
                quench_mask
            )

            # Process each component
            for component_id, component_mask in quench_components.items():
                # Convert CuPy array to numpy for size check
                component_mask_cpu = (
                    cp.asnumpy(component_mask)
                    if hasattr(component_mask, "get")
                    else component_mask
                )
                size = np.sum(component_mask_cpu)

                # Use adaptive minimum size based on array size
                min_size = max(1, min(5, envelope_gpu.size // 1000))
                if size < min_size:
                    continue  # Skip small components

                # Compute component characteristics
                center = self.characteristics.compute_center_of_mass_cuda(
                    component_mask
                )
                strength = self.characteristics.compute_quench_strength_cuda(
                    component_mask, amplitude
                )

                quenches.append(
                    {
                        "location": center,
                        "type": "amplitude",
                        "strength": float(strength),
                        "threshold": self.amplitude_threshold,
                        "size": int(size),
                        "component_id": component_id,
                    }
                )

        # Cleanup GPU memory
        del amplitude, quench_mask
        if "quench_components" in locals():
            del quench_components

        return quenches

    def _detect_detuning_quenches_cuda(self, envelope_gpu) -> List[Dict[str, Any]]:
        """Detect detuning quenches using CUDA acceleration."""
        quenches = []

        # Compute local frequency from phase evolution
        if envelope_gpu.shape[-1] > 1:  # Need at least 2 time slices
            local_frequency = self.characteristics.compute_local_frequency_cuda(
                envelope_gpu
            )

            # Detuning from carrier frequency
            detuning = cp.abs(local_frequency - self.carrier_frequency)

            # Find locations exceeding detuning threshold
            quench_mask = detuning > self.detuning_threshold

            if cp.any(quench_mask):
                # Apply morphological operations to filter noise
                quench_mask = self.morphology.apply_morphological_operations_cuda(
                    quench_mask
                )

                # Find connected components
                quench_components = self.morphology.find_connected_components_cuda(
                    quench_mask
                )

                # Process each component
                for component_id, component_mask in quench_components.items():
                    # Convert CuPy array to numpy for size check
                    component_mask_cpu = (
                        cp.asnumpy(component_mask)
                        if hasattr(component_mask, "get")
                        else component_mask
                    )
                    size = np.sum(component_mask_cpu)

                    if size < self.config.get("min_quench_size", 5):
                        continue  # Skip small components

                    # Compute component characteristics
                    center = self.characteristics.compute_center_of_mass_cuda(
                        component_mask
                    )
                    strength = self.characteristics.compute_detuning_strength_cuda(
                        component_mask, detuning
                    )

                    quenches.append(
                        {
                            "location": center,
                            "type": "detuning",
                            "strength": float(strength),
                            "threshold": self.detuning_threshold,
                            "size": int(size),
                            "component_id": component_id,
                        }
                    )

        # Cleanup GPU memory
        if "local_frequency" in locals():
            del local_frequency
        if "detuning" in locals():
            del detuning
        if "quench_mask" in locals():
            del quench_mask
        if "quench_components" in locals():
            del quench_components

        return quenches

    def _detect_gradient_quenches_cuda(self, envelope_gpu) -> List[Dict[str, Any]]:
        """Detect gradient quenches using CUDA acceleration."""
        quenches = []

        # Compute 7D gradient on GPU
        gradient_magnitude = self.characteristics.compute_7d_gradient_magnitude_cuda(
            envelope_gpu
        )

        # Find locations exceeding gradient threshold
        quench_mask = gradient_magnitude > self.gradient_threshold

        if cp.any(quench_mask):
            # Apply morphological operations to filter noise
            quench_mask = self.morphology.apply_morphological_operations_cuda(
                quench_mask
            )

            # Find connected components
            quench_components = self.morphology.find_connected_components_cuda(
                quench_mask
            )

            # Process each component
            for component_id, component_mask in quench_components.items():
                # Convert CuPy array to numpy for size check
                component_mask_cpu = (
                    cp.asnumpy(component_mask)
                    if hasattr(component_mask, "get")
                    else component_mask
                )
                size = np.sum(component_mask_cpu)

                # Use adaptive minimum size based on array size
                min_size = max(1, min(5, envelope_gpu.size // 1000))
                if size < min_size:
                    continue  # Skip small components

                # Compute component characteristics
                center = self.characteristics.compute_center_of_mass_cuda(
                    component_mask
                )
                strength = self.characteristics.compute_gradient_strength_cuda(
                    component_mask, gradient_magnitude
                )

                quenches.append(
                    {
                        "location": center,
                        "type": "gradient",
                        "strength": float(strength),
                        "threshold": self.gradient_threshold,
                        "size": int(size),
                        "component_id": component_id,
                    }
                )

        # Cleanup GPU memory
        if "gradient_magnitude" in locals():
            del gradient_magnitude
        if "quench_mask" in locals():
            del quench_mask
        if "quench_components" in locals():
            del quench_components

        return quenches

    def _validate_thresholds(self) -> None:
        """
        Validate threshold parameters.

        Physical Meaning:
            Ensures that threshold parameters are physically reasonable
            and consistent with the BVP theory.

        Raises:
            ValueError: If thresholds are invalid.
        """
        if self.amplitude_threshold <= 0:
            raise ValueError("Amplitude threshold must be positive")

        if self.detuning_threshold <= 0:
            raise ValueError("Detuning threshold must be positive")

        if self.gradient_threshold <= 0:
            raise ValueError("Gradient threshold must be positive")

        if self.carrier_frequency <= 0:
            raise ValueError("Carrier frequency must be positive")
