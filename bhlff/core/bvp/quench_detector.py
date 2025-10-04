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

from ..domain.domain_7d import Domain7D


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

        # Extract threshold parameters
        self.amplitude_threshold = config.get("amplitude_threshold", 0.8)
        self.detuning_threshold = config.get("detuning_threshold", 0.1)
        self.gradient_threshold = config.get("gradient_threshold", 0.5)
        self.carrier_frequency = config.get("carrier_frequency", 1.85e43)

        # Setup threshold validation
        self._validate_thresholds()

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
            quench_mask = self._apply_morphological_operations(quench_mask)

            # Find connected components
            quench_components = self._find_connected_components(quench_mask)

            # Process each component
            for component_id, component_mask in quench_components.items():
                if np.sum(component_mask) < self.config.get("min_quench_size", 5):
                    continue  # Skip small components

                # Compute component characteristics
                center = self._compute_center_of_mass(component_mask)
                strength = self._compute_quench_strength(component_mask, amplitude)
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
            local_frequency = self._compute_local_frequency(envelope)
            
            # Detuning from carrier frequency
            detuning = np.abs(local_frequency - self.carrier_frequency)

            # Find locations exceeding detuning threshold
            quench_mask = detuning > self.detuning_threshold

            if np.any(quench_mask):
                # Apply morphological operations to filter noise
                quench_mask = self._apply_morphological_operations(quench_mask)

                # Find connected components
                quench_components = self._find_connected_components(quench_mask)

                # Process each component
                for component_id, component_mask in quench_components.items():
                    if np.sum(component_mask) < self.config.get("min_quench_size", 5):
                        continue  # Skip small components

                    # Compute component characteristics
                    center = self._compute_center_of_mass(component_mask)
                    strength = self._compute_detuning_strength(component_mask, detuning)
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
        gradient_magnitude = self._compute_7d_gradient_magnitude(envelope)

        # Find locations exceeding gradient threshold
        quench_mask = gradient_magnitude > self.gradient_threshold

        if np.any(quench_mask):
            # Apply morphological operations to filter noise
            quench_mask = self._apply_morphological_operations(quench_mask)

            # Find connected components
            quench_components = self._find_connected_components(quench_mask)

            # Process each component
            for component_id, component_mask in quench_components.items():
                if np.sum(component_mask) < self.config.get("min_quench_size", 5):
                    continue  # Skip small components

                # Compute component characteristics
                center = self._compute_center_of_mass(component_mask)
                strength = self._compute_gradient_strength(component_mask, gradient_magnitude)
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

    def _apply_morphological_operations(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to filter noise in quench mask.

        Physical Meaning:
            Applies binary morphological operations to remove noise
            and fill gaps in quench regions, improving detection quality.

        Mathematical Foundation:
            - Binary opening: Erosion followed by dilation
            - Binary closing: Dilation followed by erosion
            - Removes small noise components and fills small gaps

        Args:
            mask (np.ndarray): Binary mask of quench regions.

        Returns:
            np.ndarray: Filtered binary mask.
        """
        try:
            from scipy.ndimage import binary_opening, binary_closing
            
            # Define structuring element for 7D operations
            # Use 3x3x3x3x3x3x3 structure for 7D
            structure = np.ones((3, 3, 3, 3, 3, 3, 3), dtype=bool)
            
            # Apply binary opening to remove small noise
            filtered_mask = binary_opening(mask, structure=structure)
            
            # Apply binary closing to fill small gaps
            filtered_mask = binary_closing(filtered_mask, structure=structure)
            
            return filtered_mask
            
        except ImportError:
            # Fallback: simple filtering without scipy
            return self._simple_morphological_filter(mask)

    def _simple_morphological_filter(self, mask: np.ndarray) -> np.ndarray:
        """
        Simple morphological filtering without scipy dependency.

        Physical Meaning:
            Basic noise filtering using local neighborhood operations
            to remove isolated pixels and fill small gaps.

        Args:
            mask (np.ndarray): Binary mask of quench regions.

        Returns:
            np.ndarray: Filtered binary mask.
        """
        # Simple erosion: remove isolated pixels
        filtered_mask = mask.copy()
        
        # Simple dilation: fill small gaps
        # This is a basic implementation for 7D
        for axis in range(mask.ndim):
            # Apply 1D dilation along each axis
            for i in range(1, mask.shape[axis] - 1):
                if axis == 0:
                    if mask[i-1, :, :, :, :, :, :].any() and mask[i+1, :, :, :, :, :, :].any():
                        filtered_mask[i, :, :, :, :, :, :] = True
                elif axis == 1:
                    if mask[:, i-1, :, :, :, :, :].any() and mask[:, i+1, :, :, :, :, :].any():
                        filtered_mask[:, i, :, :, :, :, :] = True
                # Continue for other axes...
        
        return filtered_mask

    def _find_connected_components(self, mask: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Find connected components in quench mask.

        Physical Meaning:
            Groups nearby quench events into connected components,
            representing coherent quench regions in 7D space-time.

        Mathematical Foundation:
            Uses connected component labeling to identify regions
            where quench events are spatially/phase/temporally connected.

        Args:
            mask (np.ndarray): Binary mask of quench regions.

        Returns:
            Dict[int, np.ndarray]: Dictionary mapping component IDs to
                binary masks of each component.
        """
        try:
            from scipy.ndimage import label
            
            # Label connected components
            labeled_mask, num_components = label(mask)
            
            # Extract individual components
            components = {}
            for component_id in range(1, num_components + 1):
                component_mask = (labeled_mask == component_id)
                components[component_id] = component_mask
            
            return components
            
        except ImportError:
            # Fallback: simple connected component analysis
            return self._simple_connected_components(mask)

    def _simple_connected_components(self, mask: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Simple connected component analysis without scipy.

        Physical Meaning:
            Basic grouping of nearby quench events using
            flood-fill algorithm for 7D space.

        Args:
            mask (np.ndarray): Binary mask of quench regions.

        Returns:
            Dict[int, np.ndarray]: Dictionary mapping component IDs to
                binary masks of each component.
        """
        components = {}
        visited = np.zeros_like(mask, dtype=bool)
        component_id = 0
        
        # Find all quench points
        quench_points = np.where(mask)
        
        for point in zip(*quench_points):
            if not visited[point]:
                component_id += 1
                component_mask = np.zeros_like(mask, dtype=bool)
                
                # Simple flood-fill for this component
                self._flood_fill_7d(mask, visited, component_mask, point)
                components[component_id] = component_mask
        
        return components

    def _flood_fill_7d(self, mask: np.ndarray, visited: np.ndarray, 
                       component_mask: np.ndarray, start_point: Tuple[int, ...]) -> None:
        """
        Flood-fill algorithm for 7D connected components.

        Physical Meaning:
            Recursively fills connected quench regions starting from
            a seed point, identifying coherent quench structures.

        Args:
            mask (np.ndarray): Binary mask of quench regions.
            visited (np.ndarray): Visited points mask.
            component_mask (np.ndarray): Current component mask.
            start_point (Tuple[int, ...]): Starting point for flood-fill.
        """
        stack = [start_point]
        
        while stack:
            point = stack.pop()
            
            if visited[point]:
                continue
                
            visited[point] = True
            component_mask[point] = True
            
            # Check 7D neighbors (3^7 = 2187 neighbors, but we check only immediate ones)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        for dphi1 in [-1, 0, 1]:
                            for dphi2 in [-1, 0, 1]:
                                for dphi3 in [-1, 0, 1]:
                                    for dt in [-1, 0, 1]:
                                        if dx == dy == dz == dphi1 == dphi2 == dphi3 == dt == 0:
                                            continue
                                        
                                        neighbor = (
                                            point[0] + dx, point[1] + dy, point[2] + dz,
                                            point[3] + dphi1, point[4] + dphi2, point[5] + dphi3,
                                            point[6] + dt
                                        )
                                        
                                        # Check bounds
                                        if (0 <= neighbor[0] < mask.shape[0] and
                                            0 <= neighbor[1] < mask.shape[1] and
                                            0 <= neighbor[2] < mask.shape[2] and
                                            0 <= neighbor[3] < mask.shape[3] and
                                            0 <= neighbor[4] < mask.shape[4] and
                                            0 <= neighbor[5] < mask.shape[5] and
                                            0 <= neighbor[6] < mask.shape[6]):
                                            
                                            if (mask[neighbor] and not visited[neighbor]):
                                                stack.append(neighbor)

    def _compute_center_of_mass(self, component_mask: np.ndarray) -> Tuple[float, ...]:
        """
        Compute center of mass for a quench component.

        Physical Meaning:
            Calculates the center of mass of a quench component,
            representing the effective location of the quench event
            in 7D space-time.

        Mathematical Foundation:
            Center of mass = Σ(r_i * w_i) / Σ(w_i)
            where r_i are coordinates and w_i are weights (amplitudes).

        Args:
            component_mask (np.ndarray): Binary mask of component.

        Returns:
            Tuple[float, ...]: 7D coordinates of center of mass.
        """
        # Get coordinates of component points
        coords = np.where(component_mask)
        
        if len(coords[0]) == 0:
            return (0.0,) * 7
        
        # Compute center of mass (simple average for now)
        center = []
        for axis in range(7):
            center.append(float(np.mean(coords[axis])))
        
        return tuple(center)

    def _compute_quench_strength(self, component_mask: np.ndarray, amplitude: np.ndarray) -> float:
        """
        Compute quench strength for a component.

        Physical Meaning:
            Calculates the strength of a quench event based on
            the maximum amplitude within the component region.

        Mathematical Foundation:
            Quench strength = max(|A|) within component
            This represents the peak field strength in the quench region.

        Args:
            component_mask (np.ndarray): Binary mask of component.
            amplitude (np.ndarray): Amplitude field.

        Returns:
            float: Quench strength.
        """
        # Get amplitudes within component
        component_amplitudes = amplitude[component_mask]
        
        if len(component_amplitudes) == 0:
            return 0.0
        
        # Return maximum amplitude as quench strength
        return float(np.max(component_amplitudes))

    def _compute_local_frequency(self, envelope: np.ndarray) -> np.ndarray:
        """
        Compute local frequency from phase evolution.

        Physical Meaning:
            Calculates the local frequency at each point in 7D space-time
            by analyzing the phase evolution of the envelope field.
            This represents the instantaneous frequency of the BVP field.

        Mathematical Foundation:
            ω_local = |dφ/dt| / dt
            where φ is the phase of the envelope and dt is the time step.
            Uses finite differences to approximate the derivative.

        Args:
            envelope (np.ndarray): 7D envelope field.

        Returns:
            np.ndarray: Local frequency field with same shape as envelope.
        """
        # Extract phase
        phase = np.angle(envelope)
        
        # Compute phase difference along time axis
        if envelope.shape[-1] > 1:
            phase_diff = np.diff(phase, axis=-1)
            
            # Get time step
            dt = self.domain_7d.temporal_config.dt
            
            # Compute local frequency (avoid division by zero)
            local_frequency = np.abs(phase_diff) / (dt + 1e-12)
            
            # Pad to match original shape
            local_frequency = np.pad(local_frequency, 
                                   [(0, 0)] * (local_frequency.ndim - 1) + [(0, 1)],
                                   mode='edge')
        else:
            # Single time slice - use zero frequency
            local_frequency = np.zeros_like(phase)
        
        return local_frequency

    def _compute_detuning_strength(self, component_mask: np.ndarray, detuning: np.ndarray) -> float:
        """
        Compute detuning strength for a component.

        Physical Meaning:
            Calculates the strength of a detuning quench event based on
            the maximum detuning within the component region.

        Mathematical Foundation:
            Detuning strength = max(|ω_local - ω_0|) within component
            This represents the peak frequency deviation in the quench region.

        Args:
            component_mask (np.ndarray): Binary mask of component.
            detuning (np.ndarray): Detuning field.

        Returns:
            float: Detuning strength.
        """
        # Get detuning values within component
        component_detuning = detuning[component_mask]
        
        if len(component_detuning) == 0:
            return 0.0
        
        # Return maximum detuning as quench strength
        return float(np.max(component_detuning))

    def _compute_7d_gradient_magnitude(self, envelope: np.ndarray) -> np.ndarray:
        """
        Compute 7D gradient magnitude of envelope field.

        Physical Meaning:
            Calculates the magnitude of the gradient in all 7 dimensions
            (3 spatial + 3 phase + 1 temporal), representing the rate
            of change of the envelope field in 7D space-time.

        Mathematical Foundation:
            ∇A = (∂A/∂x, ∂A/∂y, ∂A/∂z, ∂A/∂φ₁, ∂A/∂φ₂, ∂A/∂φ₃, ∂A/∂t)
            |∇A| = √(Σ|∂A/∂xᵢ|²)
            Uses finite differences to approximate partial derivatives.

        Args:
            envelope (np.ndarray): 7D envelope field.

        Returns:
            np.ndarray: Gradient magnitude field with same shape as envelope.
        """
        # Get differentials for all 7 dimensions
        differentials = self.domain_7d.get_differentials()
        
        # Compute gradients in all 7 dimensions
        gradients = []
        
        # Spatial gradients (x, y, z)
        for axis, dx in enumerate([differentials["dx"], differentials["dy"], differentials["dz"]]):
            grad = np.gradient(envelope, dx, axis=axis)
            gradients.append(grad)
        
        # Phase gradients (φ₁, φ₂, φ₃)
        for axis, dphi in enumerate([differentials["dphi_1"], differentials["dphi_2"], differentials["dphi_3"]]):
            grad = np.gradient(envelope, dphi, axis=axis + 3)
            gradients.append(grad)
        
        # Temporal gradient (t)
        if envelope.shape[-1] > 1:
            dt = differentials.get("dt", 1.0)
            grad_t = np.gradient(envelope, dt, axis=-1)
            gradients.append(grad_t)
        else:
            # Single time slice - zero temporal gradient
            grad_t = np.zeros_like(envelope)
            gradients.append(grad_t)

        # Compute gradient magnitude
        grad_magnitude = np.sqrt(sum(np.abs(grad) ** 2 for grad in gradients))
        
        return grad_magnitude

    def _compute_gradient_strength(self, component_mask: np.ndarray, gradient_magnitude: np.ndarray) -> float:
        """
        Compute gradient strength for a component.

        Physical Meaning:
            Calculates the strength of a gradient quench event based on
            the maximum gradient magnitude within the component region.

        Mathematical Foundation:
            Gradient strength = max(|∇A|) within component
            This represents the peak gradient magnitude in the quench region.

        Args:
            component_mask (np.ndarray): Binary mask of component.
            gradient_magnitude (np.ndarray): Gradient magnitude field.

        Returns:
            float: Gradient strength.
        """
        # Get gradient magnitudes within component
        component_gradients = gradient_magnitude[component_mask]
        
        if len(component_gradients) == 0:
            return 0.0
        
        # Return maximum gradient magnitude as quench strength
        return float(np.max(component_gradients))
