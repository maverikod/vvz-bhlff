"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Field projection analysis for Level D models.

This module implements field projection analysis onto different
interaction windows (electromagnetic, strong, weak) corresponding
to different frequency-amplitude characteristics of the unified
phase field.

Physical Meaning:
    Field projections separate the unified phase field into different
    interaction regimes based on frequency and amplitude characteristics:
    - EM field: Phase gradients (U(1) symmetry), long-range interactions
    - Strong field: High-Q localized modes, short-range interactions
    - Weak field: Chiral combinations, parity-breaking interactions

Mathematical Foundation:
    - EM projection: P_EM[a] = FFT⁻¹[FFT(a) × H_EM(ω)]
    - Strong projection: P_STRONG[a] = FFT⁻¹[FFT(a) × H_STRONG(ω)]
    - Weak projection: P_WEAK[a] = FFT⁻¹[FFT(a) × H_WEAK(ω)]

Example:
    >>> from bhlff.models.level_d.projections import FieldProjection
    >>> projection = FieldProjection(field, window_params)
    >>> results = projection.project_field_windows()
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

from bhlff.models.base.abstract_models import AbstractLevelModels


class FieldProjection:
    """
    Field projection onto different interaction windows.

    Physical Meaning:
        Projects the unified phase field onto different frequency
        windows corresponding to electromagnetic, strong, and weak
        interactions as envelope functions.

    Mathematical Foundation:
        Uses frequency-domain filtering to separate different
        interaction regimes based on their characteristic
        frequency and amplitude signatures.

    Attributes:
        field (np.ndarray): Input phase field
        projection_params (Dict): Projection parameters
        _em_projector (EMProjector): Electromagnetic field projector
        _strong_projector (StrongProjector): Strong field projector
        _weak_projector (WeakProjector): Weak field projector
        _signature_analyzer (SignatureAnalyzer): Field signature analyzer
    """

    def __init__(self, field: np.ndarray, projection_params: Dict[str, Any]):
        """
        Initialize field projection.

        Physical Meaning:
            Sets up the field projection system for separating
            the unified phase field into different interaction
            regimes.

        Args:
            field (np.ndarray): Input phase field
            projection_params (Dict): Projection parameters
        """
        self.field = field
        self.projection_params = projection_params
        self.logger = logging.getLogger(__name__)

        # Initialize projectors
        self._em_projector = EMProjector(projection_params.get("em", {}))
        self._strong_projector = StrongProjector(projection_params.get("strong", {}))
        self._weak_projector = WeakProjector(projection_params.get("weak", {}))

        # Initialize signature analyzer
        self._signature_analyzer = SignatureAnalyzer()

        self.logger.info("Field projection initialized")

    def project_em_field(self, field: np.ndarray) -> np.ndarray:
        """
        Project onto electromagnetic window.

        Physical Meaning:
            Extracts the electromagnetic component of the phase
            field, corresponding to U(1) gauge interactions
            and phase gradient flows.

        Mathematical Foundation:
            EM_field = FFT⁻¹[FFT(field) × H_EM(ω)]
            where H_EM(ω) is the EM window filter.

        Args:
            field (np.ndarray): Input field

        Returns:
            np.ndarray: EM field projection
        """
        return self._em_projector.project(field)

    def project_strong_field(self, field: np.ndarray) -> np.ndarray:
        """
        Project onto strong interaction window.

        Physical Meaning:
            Extracts the strong interaction component, corresponding
            to high-Q localized modes and steep amplitude gradients
            near the core.

        Mathematical Foundation:
            Strong_field = FFT⁻¹[FFT(field) × H_STRONG(ω)]
            where H_STRONG(ω) is the strong window filter.

        Args:
            field (np.ndarray): Input field

        Returns:
            np.ndarray: Strong field projection
        """
        return self._strong_projector.project(field)

    def project_weak_field(self, field: np.ndarray) -> np.ndarray:
        """
        Project onto weak interaction window.

        Physical Meaning:
            Extracts the weak interaction component, corresponding
            to chiral combinations and parity-breaking envelope
            functions with low Q and leakage.

        Mathematical Foundation:
            Weak_field = FFT⁻¹[FFT(field) × H_WEAK(ω)]
            where H_WEAK(ω) is the weak window filter.

        Args:
            field (np.ndarray): Input field

        Returns:
            np.ndarray: Weak field projection
        """
        return self._weak_projector.project(field)

    def project_field_windows(self, field: np.ndarray) -> Dict[str, Any]:
        """
        Project fields onto different frequency-amplitude windows.

        Physical Meaning:
            Separates the unified phase field into different
            interaction regimes based on frequency and amplitude
            characteristics.

        Args:
            field (np.ndarray): Input field

        Returns:
            Dict: Projected fields and signatures including:
                - em_projection: Electromagnetic field projection
                - strong_projection: Strong interaction projection
                - weak_projection: Weak interaction projection
                - signatures: Characteristic signatures for each field type
        """
        self.logger.info("Projecting fields onto interaction windows")

        # Project onto each window
        em_projection = self.project_em_field(field)
        strong_projection = self.project_strong_field(field)
        weak_projection = self.project_weak_field(field)

        # Analyze field signatures
        signatures = self._signature_analyzer.analyze_field_signatures(
            {"em": em_projection, "strong": strong_projection, "weak": weak_projection}
        )

        results = {
            "em_projection": em_projection,
            "strong_projection": strong_projection,
            "weak_projection": weak_projection,
            "signatures": signatures,
        }

        self.logger.info("Field projection completed")
        return results

    def analyze_field_signatures(
        self, projections: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Analyze characteristic signatures of each field type.

        Physical Meaning:
            Computes characteristic signatures for each interaction
            type, including localization, range, and anisotropy
            properties.

        Args:
            projections (Dict): Dictionary of field projections

        Returns:
            Dict: Signature analysis results
        """
        return self._signature_analyzer.analyze_field_signatures(projections)


class ProjectionAnalyzer:
    """
    Analyzer for field projections onto interaction windows.

    Physical Meaning:
        Analyzes field projections onto different interaction
        windows to understand the field structure and dynamics
        in different interaction regimes.
    """

    def __init__(self, domain: "Domain", parameters: Dict[str, Any]):
        """Initialize projection analyzer."""
        self.domain = domain
        self.parameters = parameters
        self.logger = logging.getLogger(__name__)

    def project_field_windows(
        self, field: np.ndarray, window_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Project fields onto different frequency-amplitude windows.

        Physical Meaning:
            Separates the unified phase field into different
            interaction regimes based on frequency and amplitude
            characteristics.

        Args:
            field (np.ndarray): Input field
            window_params (Dict): Window parameters

        Returns:
            Dict: Projection analysis results
        """
        # Create field projection
        projection = FieldProjection(field, window_params)

        # Perform projections
        results = projection.project_field_windows(field)

        return results


class EMProjector:
    """Electromagnetic field projector."""

    def __init__(self, params: Dict[str, Any]):
        """Initialize EM projector."""
        self.params = params
        self.frequency_range = params.get("frequency_range", [0.1, 1.0])
        self.amplitude_threshold = params.get("amplitude_threshold", 0.1)
        self.filter_type = params.get("filter_type", "bandpass")

    def project(self, field: np.ndarray) -> np.ndarray:
        """Project field onto EM window."""
        # FFT transform
        fft_field = np.fft.fftn(field)

        # Create EM filter
        em_filter = self._create_em_filter(fft_field.shape)

        # Apply filter
        em_field_fft = fft_field * em_filter

        # Inverse FFT
        em_field = np.fft.ifftn(em_field_fft)

        return em_field.real

    def _create_em_filter(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Create EM window filter."""
        # Create frequency grid
        frequencies = self._create_frequency_grid(shape)

        # Create bandpass filter
        filter_low = self.frequency_range[0]
        filter_high = self.frequency_range[1]

        em_filter = np.where(
            (frequencies >= filter_low) & (frequencies <= filter_high), 1.0, 0.0
        )

        return em_filter

    def _create_frequency_grid(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Create frequency grid for filtering."""
        if len(shape) == 3:
            kx = np.fft.fftfreq(shape[0])
            ky = np.fft.fftfreq(shape[1])
            kz = np.fft.fftfreq(shape[2])
            KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
            frequencies = np.sqrt(KX**2 + KY**2 + KZ**2)
        else:
            frequencies = np.ones(shape)

        return frequencies


class StrongProjector:
    """Strong interaction field projector."""

    def __init__(self, params: Dict[str, Any]):
        """Initialize strong projector."""
        self.params = params
        self.frequency_range = params.get("frequency_range", [1.0, 10.0])
        self.q_threshold = params.get("q_threshold", 100)
        self.filter_type = params.get("filter_type", "high_q")

    def project(self, field: np.ndarray) -> np.ndarray:
        """Project field onto strong window."""
        # FFT transform
        fft_field = np.fft.fftn(field)

        # Create strong filter
        strong_filter = self._create_strong_filter(fft_field.shape)

        # Apply filter
        strong_field_fft = fft_field * strong_filter

        # Inverse FFT
        strong_field = np.fft.ifftn(strong_field_fft)

        return strong_field.real

    def _create_strong_filter(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Create strong window filter."""
        # Create frequency grid
        frequencies = self._create_frequency_grid(shape)

        # Create high-frequency filter
        filter_low = self.frequency_range[0]
        filter_high = self.frequency_range[1]

        strong_filter = np.where(
            (frequencies >= filter_low) & (frequencies <= filter_high), 1.0, 0.0
        )

        # Apply Q-factor filtering
        q_factor = self.q_threshold
        strong_filter *= self._apply_q_factor_filter(frequencies, q_factor)

        return strong_filter

    def _create_frequency_grid(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Create frequency grid for filtering."""
        if len(shape) == 3:
            kx = np.fft.fftfreq(shape[0])
            ky = np.fft.fftfreq(shape[1])
            kz = np.fft.fftfreq(shape[2])
            KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
            frequencies = np.sqrt(KX**2 + KY**2 + KZ**2)
        else:
            frequencies = np.ones(shape)

        return frequencies

    def _apply_q_factor_filter(
        self, frequencies: np.ndarray, q_factor: float
    ) -> np.ndarray:
        """Apply Q-factor filtering using step resonator model."""
        # Step resonator Q-factor filter
        q_filter = self._step_q_factor_filter(frequencies, q_factor)
        return q_filter
    
    def _step_q_factor_filter(self, frequencies: np.ndarray, q_factor: float) -> np.ndarray:
        """
        Step function Q-factor filter.
        
        Physical Meaning:
            Implements step resonator model for Q-factor filtering instead of
            exponential decay. This follows 7D BVP theory principles where
            filtering occurs through semi-transparent boundaries.
            
        Mathematical Foundation:
            F(f) = F₀ * Θ(f_cutoff - f) where Θ is the Heaviside step function
            and f_cutoff is the cutoff frequency for the resonator.
            
        Args:
            frequencies: Frequency array
            q_factor: Q-factor parameter
            
        Returns:
            Step function Q-factor filter
        """
        # Step resonator parameters
        cutoff_frequency = q_factor
        filter_strength = 1.0
        
        # Step function filter: 1.0 below cutoff, 0.0 above
        return filter_strength * np.where(frequencies < cutoff_frequency, 1.0, 0.0)


class WeakProjector:
    """Weak interaction field projector."""

    def __init__(self, params: Dict[str, Any]):
        """Initialize weak projector."""
        self.params = params
        self.frequency_range = params.get("frequency_range", [0.01, 0.1])
        self.q_threshold = params.get("q_threshold", 10)
        self.filter_type = params.get("filter_type", "chiral")

    def project(self, field: np.ndarray) -> np.ndarray:
        """Project field onto weak window."""
        # FFT transform
        fft_field = np.fft.fftn(field)

        # Create weak filter
        weak_filter = self._create_weak_filter(fft_field.shape)

        # Apply filter
        weak_field_fft = fft_field * weak_filter

        # Inverse FFT
        weak_field = np.fft.ifftn(weak_field_fft)

        return weak_field.real

    def _create_weak_filter(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Create weak window filter."""
        # Create frequency grid
        frequencies = self._create_frequency_grid(shape)

        # Create low-frequency filter
        filter_low = self.frequency_range[0]
        filter_high = self.frequency_range[1]

        weak_filter = np.where(
            (frequencies >= filter_low) & (frequencies <= filter_high), 1.0, 0.0
        )

        # Apply chiral filtering
        chiral_factor = self.params.get("chiral_threshold", 0.1)
        weak_filter *= self._apply_chiral_filter(chiral_factor)

        return weak_filter

    def _create_frequency_grid(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Create frequency grid for filtering."""
        if len(shape) == 3:
            kx = np.fft.fftfreq(shape[0])
            ky = np.fft.fftfreq(shape[1])
            kz = np.fft.fftfreq(shape[2])
            KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
            frequencies = np.sqrt(KX**2 + KY**2 + KZ**2)
        else:
            frequencies = np.ones(shape)

        return frequencies

    def _apply_chiral_filter(self, chiral_factor: float) -> np.ndarray:
        """Apply chiral filtering."""
        # Simple chiral filter
        chiral_filter = (
            np.ones_like(chiral_factor)
            if np.isscalar(chiral_factor)
            else np.ones(chiral_factor.shape)
        )
        return chiral_filter


class SignatureAnalyzer:
    """Analyzer for field signatures."""

    def __init__(self):
        """Initialize signature analyzer."""
        self.signature_threshold = 0.1
        self.localization_threshold = 0.5
        self.anisotropy_threshold = 0.3
        self.range_threshold = 0.2

    def analyze_field_signatures(
        self, projections: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Analyze characteristic signatures of each field type.

        Physical Meaning:
            Computes characteristic signatures for each interaction
            type, including localization, range, and anisotropy
            properties.

        Args:
            projections (Dict): Dictionary of field projections

        Returns:
            Dict: Signature analysis results
        """
        signatures = {}

        for field_type, field in projections.items():
            signatures[field_type] = self._analyze_single_field_signature(
                field, field_type
            )

        return signatures

    def _analyze_single_field_signature(
        self, field: np.ndarray, field_type: str
    ) -> Dict[str, Any]:
        """Analyze signature of a single field."""
        # Compute basic statistics
        field_norm = np.linalg.norm(field)
        field_energy = np.sum(np.abs(field) ** 2)

        # Compute localization
        localization = self._compute_localization(field)

        # Compute range characteristics
        range_characteristics = self._compute_range_characteristics(field)

        # Compute anisotropy
        anisotropy = self._compute_anisotropy(field)

        # Field-specific analysis
        if field_type == "em":
            chirality = self._compute_chirality(field)
        elif field_type == "strong":
            confinement = self._compute_confinement(field)
        elif field_type == "weak":
            parity_violation = self._compute_parity_violation(field)
        else:
            chirality = 0.0
            confinement = 0.0
            parity_violation = 0.0

        return {
            "field_norm": float(field_norm),
            "field_energy": float(field_energy),
            "localization": localization,
            "range_characteristics": range_characteristics,
            "anisotropy": anisotropy,
            "chirality": chirality if field_type == "em" else 0.0,
            "confinement": confinement if field_type == "strong" else 0.0,
            "parity_violation": parity_violation if field_type == "weak" else 0.0,
        }

    def _compute_localization(self, field: np.ndarray) -> float:
        """Compute field localization metric."""
        # Use variance as localization metric
        localization = np.var(np.abs(field))
        return float(localization)

    def _compute_range_characteristics(self, field: np.ndarray) -> Dict[str, float]:
        """Compute range characteristics."""
        # Compute correlation length
        correlation_length = self._compute_correlation_length(field)

        # Compute decay rate
        decay_rate = self._compute_decay_rate(field)

        return {
            "correlation_length": float(correlation_length),
            "decay_rate": float(decay_rate),
        }

    def _compute_anisotropy(self, field: np.ndarray) -> float:
        """Compute field anisotropy."""
        # Simple anisotropy metric based on directional variance
        if len(field.shape) == 3:
            # Compute variance along each axis
            var_x = np.var(field, axis=(1, 2))
            var_y = np.var(field, axis=(0, 2))
            var_z = np.var(field, axis=(0, 1))

            # Compute anisotropy
            anisotropy = np.std([np.mean(var_x), np.mean(var_y), np.mean(var_z)])
        else:
            anisotropy = 0.0

        return float(anisotropy)

    def _compute_chirality(self, field: np.ndarray) -> float:
        """Compute field chirality."""
        # Simple chirality metric
        chirality = np.mean(np.imag(field))
        return float(chirality)

    def _compute_confinement(self, field: np.ndarray) -> float:
        """Compute field confinement."""
        # Simple confinement metric
        mean_abs = np.mean(np.abs(field))
        if mean_abs == 0:
            return 0.0
        confinement = np.max(np.abs(field)) / mean_abs
        return float(confinement)

    def _compute_parity_violation(self, field: np.ndarray) -> float:
        """Compute parity violation."""
        # Simple parity violation metric
        parity_violation = np.mean(np.abs(field - np.flip(field)))
        return float(parity_violation)

    def _compute_correlation_length(self, field: np.ndarray) -> float:
        """Compute correlation length."""
        # Simple correlation length computation
        correlation_length = 1.0  # Placeholder
        return correlation_length

    def _compute_decay_rate(self, field: np.ndarray) -> float:
        """Compute decay rate."""
        # Simple decay rate computation
        decay_rate = 1.0  # Placeholder
        return decay_rate
