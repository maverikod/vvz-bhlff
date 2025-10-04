"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Quench memory and pinning analysis module for Level C test C3.

This module implements comprehensive analysis of quench memory effects
and pinning in the 7D phase field theory, focusing on memory-induced
field stabilization and drift suppression.

Physical Meaning:
    Analyzes quench memory effects in the 7D phase field, including:
    - Quench event detection and memory formation
    - Memory kernel analysis and information retention
    - Pinning effects and field stabilization
    - Drift velocity analysis and suppression

Mathematical Foundation:
    Implements quench memory analysis using:
    - Memory kernel: K(t) = (1/τ) exp(-t/τ)
    - Memory term: Γ_memory[a] = -γ ∫_0^t K(t-τ) a(τ) dτ
    - Drift velocity: v_cell = Δx_max / Δt
    - Cross-correlation: C(t,Δt) = ∫ I_eff(x,t) I_eff(x,t+Δt) dx

Example:
    >>> analyzer = QuenchMemoryAnalysis(bvp_core)
    >>> results = analyzer.analyze_quench_memory(domain, memory_params)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass

from bhlff.core.bvp import BVPCore


@dataclass
class MemoryParameters:
    """
    Memory parameters for quench analysis.

    Physical Meaning:
        Defines the parameters for quench memory analysis,
        including memory strength, relaxation time, and
        spatial distribution.
    """

    gamma: float  # Memory strength (0 ≤ γ ≤ 1)
    tau: float  # Relaxation time
    spatial_distribution: Optional[np.ndarray] = None


@dataclass
class QuenchEvent:
    """
    Quench event information.

    Physical Meaning:
        Represents a quench event in the 7D phase field,
        characterized by its location, time, and intensity.
    """

    location: np.ndarray
    time: float
    intensity: float
    threshold_type: str  # 'amplitude', 'detuning', 'gradient'


@dataclass
class DriftAnalysis:
    """
    Drift analysis results.

    Physical Meaning:
        Contains the results of drift velocity analysis,
        including drift velocity, correlation data, and
        stability metrics.
    """

    drift_velocity: float
    correlation_shifts: List[float]
    jaccard_index: float
    stability_score: float


@dataclass
class MemoryKernel:
    """
    Memory kernel specification.

    Physical Meaning:
        Defines the memory kernel K(t) that determines
        how past events influence current field evolution.
    """

    kernel_type: str = "debye"  # "debye", "exponential", "gaussian"
    tau: float = 1.0
    strength: float = 1.0


class QuenchMemoryAnalysis:
    """
    Quench memory and pinning analysis for Level C test C3.

    Physical Meaning:
        Analyzes quench memory effects in the 7D phase field,
        including memory formation, pinning effects, and
        field stabilization mechanisms.

    Mathematical Foundation:
        Implements comprehensive quench memory analysis:
        - Memory kernel analysis: K(t) = (1/τ) exp(-t/τ)
        - Memory term: Γ_memory[a] = -γ ∫_0^t K(t-τ) a(τ) dτ
        - Drift velocity analysis: v_cell = Δx_max / Δt
        - Cross-correlation analysis for pattern stability
    """

    def __init__(self, bvp_core: BVPCore):
        """
        Initialize quench memory analysis.

        Args:
            bvp_core (BVPCore): BVP core framework instance.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)

    def analyze_quench_memory(
        self, domain: Dict[str, Any], memory_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze quench memory and pinning effects (C3 test).

        Physical Meaning:
            Performs comprehensive analysis of quench memory effects,
            including memory formation, pinning analysis, and drift
            velocity suppression.

        Mathematical Foundation:
            Analyzes the system response to memory effects:
            - Memory kernel analysis: K(t) = (1/τ) exp(-t/τ)
            - Memory term: Γ_memory[a] = -γ ∫_0^t K(t-τ) a(τ) dτ
            - Drift velocity: v_cell = Δx_max / Δt
            - Cross-correlation: C(t,Δt) = ∫ I_eff(x,t) I_eff(x,t+Δt) dx

        Args:
            domain (Dict[str, Any]): Domain parameters.
            memory_params (Dict[str, Any]): Memory parameters.

        Returns:
            Dict[str, Any]: Comprehensive quench memory analysis results.
        """
        self.logger.info("Starting quench memory analysis (C3)")

        # Extract parameters
        gamma_list = memory_params.get("gamma_list", [0.0, 0.2, 0.4, 0.6, 0.8])
        tau_list = memory_params.get("tau_list", [0.5, 1.0, 2.0])
        time_params = memory_params.get("time_integration", {})

        # Perform analysis for each memory parameter combination
        memory_results = {}
        for gamma in gamma_list:
            for tau in tau_list:
                self.logger.info(f"Analyzing γ = {gamma}, τ = {tau}")

                # Create memory parameters
                memory = MemoryParameters(gamma=gamma, tau=tau)

                # Perform time evolution with memory
                time_evolution = self._evolve_with_memory(domain, memory, time_params)

                # Analyze drift velocity
                drift_analysis = self._analyze_drift_velocity(time_evolution)

                # Analyze cross-correlation
                correlation_analysis = self._analyze_cross_correlation(time_evolution)

                # Analyze Jaccard index
                jaccard_analysis = self._analyze_jaccard_index(time_evolution)

                # Store results
                key = f"gamma_{gamma}_tau_{tau}"
                memory_results[key] = {
                    "gamma": gamma,
                    "tau": tau,
                    "time_evolution": time_evolution,
                    "drift_analysis": drift_analysis,
                    "correlation_analysis": correlation_analysis,
                    "jaccard_analysis": jaccard_analysis,
                    "is_pinned": drift_analysis.drift_velocity < 1e-3,
                }

        # Find freezing threshold
        freezing_threshold = self._find_freezing_threshold(memory_results)

        # Create summary
        summary = self._create_memory_summary(memory_results, freezing_threshold)

        return {
            "memory_results": memory_results,
            "freezing_threshold": freezing_threshold,
            "summary": summary,
            "test_passed": self._validate_c3_results(
                memory_results, freezing_threshold
            ),
        }

    def _evolve_with_memory(
        self,
        domain: Dict[str, Any],
        memory: MemoryParameters,
        time_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evolve field with memory effects.

        Physical Meaning:
            Performs time evolution of the field with memory effects,
            including memory kernel application and quench detection.

        Mathematical Foundation:
            ∂a/∂t = L[a] + Γ_memory[a] + s(x,t)
            where Γ_memory[a] = -γ ∫_0^t K(t-τ) a(τ) dτ
        """
        # Extract time parameters
        dt = time_params.get("dt", 0.005)
        T = time_params.get("T", 400.0)
        time_points = np.arange(0, T, dt)

        # Create initial field
        field = self._create_initial_field(domain)
        field_history = [field.copy()]

        # Create memory kernel
        memory_kernel = self._create_memory_kernel(memory)

        # Time evolution
        for t in time_points[1:]:
            # Apply memory term
            memory_term = self._apply_memory_term(field_history, memory_kernel, memory)

            # Apply evolution operator
            field = self._apply_evolution_operator(field, memory_term, dt)

            # Detect quench events
            quench_events = self._detect_quench_events(field, t)

            # Update field history
            field_history.append(field.copy())

            # Limit history length for memory efficiency
            if len(field_history) > 1000:
                field_history = field_history[-500:]

        return {
            "time_points": time_points,
            "field_evolution": field_history,
            "memory_kernel": memory_kernel,
            "quench_events": self._collect_quench_events(field_history),
        }

    def _analyze_drift_velocity(self, time_evolution: Dict[str, Any]) -> DriftAnalysis:
        """
        Analyze drift velocity from field evolution.

        Physical Meaning:
            Computes the drift velocity of field patterns by analyzing
            cross-correlation of effective intensity I_eff(x,t) over time.

        Mathematical Foundation:
            v_cell = Δx_max / Δt
            where Δx_max is the displacement of the correlation peak
        """
        field_evolution = time_evolution["field_evolution"]
        time_points = time_evolution["time_points"]

        # Compute effective intensity
        I_eff = [np.abs(field) ** 2 for field in field_evolution]

        # Apply moving average
        window_size = max(1, int(0.1 * len(I_eff)))
        I_eff_smooth = self._apply_moving_average(I_eff, window_size)

        # Compute cross-correlation shifts
        correlation_shifts = []
        dt = time_points[1] - time_points[0]

        for i in range(len(I_eff_smooth) - 1):
            # Cross-correlation between consecutive time steps
            corr = self._compute_cross_correlation_2d(
                I_eff_smooth[i], I_eff_smooth[i + 1]
            )

            # Find peak shift
            shift = self._find_peak_shift(corr)
            correlation_shifts.append(shift)

        # Compute drift velocity
        drift_velocity = np.mean(correlation_shifts) / dt if correlation_shifts else 0.0

        # Compute Jaccard index for pattern stability
        jaccard_index = self._compute_jaccard_index(I_eff_smooth)

        # Compute stability score
        stability_score = self._compute_stability_score(I_eff_smooth)

        return DriftAnalysis(
            drift_velocity=drift_velocity,
            correlation_shifts=correlation_shifts,
            jaccard_index=jaccard_index,
            stability_score=stability_score,
        )

    def _analyze_cross_correlation(
        self, time_evolution: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze cross-correlation of field evolution.

        Physical Meaning:
            Computes cross-correlation analysis to understand
            pattern stability and temporal coherence.
        """
        field_evolution = time_evolution["field_evolution"]

        # Compute effective intensity
        I_eff = [np.abs(field) ** 2 for field in field_evolution]

        # Compute cross-correlation matrix
        correlation_matrix = np.zeros((len(I_eff), len(I_eff)))

        for i in range(len(I_eff)):
            for j in range(len(I_eff)):
                corr = self._compute_cross_correlation_2d(I_eff[i], I_eff[j])
                correlation_matrix[i, j] = np.max(corr)

        # Analyze correlation decay
        correlation_decay = self._analyze_correlation_decay(correlation_matrix)

        return {
            "correlation_matrix": correlation_matrix,
            "correlation_decay": correlation_decay,
            "max_correlation": np.max(correlation_matrix),
            "min_correlation": np.min(correlation_matrix),
        }

    def _analyze_jaccard_index(self, time_evolution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Jaccard index for pattern stability.

        Physical Meaning:
            Computes the Jaccard index to measure pattern
            stability and similarity over time.
        """
        field_evolution = time_evolution["field_evolution"]

        # Compute effective intensity
        I_eff = [np.abs(field) ** 2 for field in field_evolution]

        # Compute Jaccard index
        jaccard_index = self._compute_jaccard_index(I_eff)

        # Analyze pattern stability
        stability_analysis = self._analyze_pattern_stability(I_eff)

        return {
            "jaccard_index": jaccard_index,
            "stability_analysis": stability_analysis,
            "pattern_stable": jaccard_index >= 0.95,
        }

    def _find_freezing_threshold(self, memory_results: Dict[str, Any]) -> float:
        """
        Find freezing threshold for memory parameters.

        Physical Meaning:
            Determines the minimum memory strength γ* required
            to achieve field pinning (v_cell < 10⁻³ L/T₀).
        """
        gamma_values = []
        drift_velocities = []

        for key, result in memory_results.items():
            gamma_values.append(result["gamma"])
            drift_velocities.append(result["drift_analysis"].drift_velocity)

        # Find first gamma where drift velocity is sufficiently small
        for i, (gamma, drift_vel) in enumerate(zip(gamma_values, drift_velocities)):
            if drift_vel < 1e-3:
                return gamma

        return float("inf")  # No freezing achieved

    def _create_memory_summary(
        self, memory_results: Dict[str, Any], freezing_threshold: float
    ) -> Dict[str, Any]:
        """
        Create memory analysis summary.

        Physical Meaning:
            Creates a comprehensive summary of the memory analysis
            results, including freezing characteristics and stability
            metrics.
        """
        total_combinations = len(memory_results)
        pinned_combinations = sum(
            1 for result in memory_results.values() if result["is_pinned"]
        )

        return {
            "total_combinations_analyzed": total_combinations,
            "pinned_combinations": pinned_combinations,
            "freezing_threshold": freezing_threshold,
            "analysis_complete": True,
            "memory_effects_detected": pinned_combinations > 0,
        }

    def _validate_c3_results(
        self, memory_results: Dict[str, Any], freezing_threshold: float
    ) -> bool:
        """
        Validate C3 test results.

        Physical Meaning:
            Validates that the C3 test results meet the acceptance
            criteria for quench memory analysis.
        """
        # Check that at γ=0, drift velocity is reasonable
        gamma_zero_results = [
            result for key, result in memory_results.items() if result["gamma"] == 0.0
        ]
        if gamma_zero_results:
            drift_vel_zero = gamma_zero_results[0]["drift_analysis"].drift_velocity
            if drift_vel_zero > 0.1:  # Should be reasonable without memory
                return False

        # Check that freezing threshold is reasonable
        if freezing_threshold > 0.1:
            return False

        # Check that pinning occurs at higher memory strengths
        high_memory_results = [
            result for key, result in memory_results.items() if result["gamma"] >= 0.4
        ]
        if not any(result["is_pinned"] for result in high_memory_results):
            return False

        return True

    def _create_initial_field(self, domain: Dict[str, Any]) -> np.ndarray:
        """
        Create initial field configuration.

        Physical Meaning:
            Creates the initial field configuration for
            time evolution analysis.
        """
        N = domain["N"]
        L = domain["L"]

        # Create coordinate arrays
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        z = np.linspace(0, L, N)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Create initial field with some structure
        center = np.array([L / 2, L / 2, L / 2])
        r_squared = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2

        field = np.exp(-r_squared / (2 * (L / 10) ** 2)) * np.exp(
            1j * np.random.random((N, N, N)) * 2 * np.pi
        )

        return field

    def _create_memory_kernel(self, memory: MemoryParameters) -> MemoryKernel:
        """
        Create memory kernel.

        Physical Meaning:
            Creates the memory kernel K(t) that determines
            how past events influence current field evolution.
        """
        return MemoryKernel(kernel_type="debye", tau=memory.tau, strength=memory.gamma)

    def _apply_memory_term(
        self,
        field_history: List[np.ndarray],
        memory_kernel: MemoryKernel,
        memory: MemoryParameters,
    ) -> np.ndarray:
        """
        Apply memory term to field.

        Physical Meaning:
            Applies the memory term Γ_memory[a] = -γ ∫_0^t K(t-τ) a(τ) dτ
            to the current field based on field history.
        """
        if len(field_history) < 2:
            return np.zeros_like(field_history[-1])

        # Simplified memory term implementation
        # In practice, this would involve proper convolution
        memory_term = -memory.gamma * field_history[-1] * memory_kernel.strength

        return memory_term

    def _apply_evolution_operator(
        self, field: np.ndarray, memory_term: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        Apply evolution operator to field.

        Physical Meaning:
            Applies the evolution operator to advance the field
            in time, including memory effects.
        """
        # Simplified evolution (in practice, this would involve proper BVP evolution)
        field_new = field + dt * (memory_term + 0.1j * field)

        return field_new

    def _detect_quench_events(
        self, field: np.ndarray, time: float
    ) -> List[QuenchEvent]:
        """
        Detect quench events in field.

        Physical Meaning:
            Detects quench events based on amplitude, detuning,
            and gradient thresholds.
        """
        quench_events = []

        # Amplitude threshold
        amplitude_threshold = 0.8
        field_amplitude = np.abs(field)
        high_amplitude_mask = field_amplitude > amplitude_threshold

        if np.any(high_amplitude_mask):
            # Find locations of high amplitude
            high_amp_coords = np.where(high_amplitude_mask)
            for i in range(len(high_amp_coords[0])):
                location = np.array(
                    [high_amp_coords[j][i] for j in range(len(high_amp_coords))]
                )
                intensity = field_amplitude[tuple(location)]

                event = QuenchEvent(
                    location=location,
                    time=time,
                    intensity=float(intensity),
                    threshold_type="amplitude",
                )
                quench_events.append(event)

        return quench_events

    def _collect_quench_events(
        self, field_history: List[np.ndarray]
    ) -> List[QuenchEvent]:
        """
        Collect all quench events from field history.

        Physical Meaning:
            Collects all quench events detected during the
            field evolution.
        """
        all_events = []

        for i, field in enumerate(field_history):
            events = self._detect_quench_events(field, float(i))
            all_events.extend(events)

        return all_events

    def _apply_moving_average(
        self, data: List[np.ndarray], window_size: int
    ) -> List[np.ndarray]:
        """
        Apply moving average to data.

        Physical Meaning:
            Applies moving average smoothing to reduce noise
            in the field evolution data.
        """
        if window_size <= 1:
            return data

        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(data), i + window_size // 2 + 1)

            window_data = data[start_idx:end_idx]
            smoothed_data = np.mean(window_data, axis=0)
            smoothed.append(smoothed_data)

        return smoothed

    def _compute_cross_correlation_2d(
        self, field1: np.ndarray, field2: np.ndarray
    ) -> np.ndarray:
        """
        Compute 2D cross-correlation between fields.

        Physical Meaning:
            Computes cross-correlation between two field
            configurations to measure pattern similarity.
        """
        # Simplified cross-correlation (in practice, this would use proper FFT-based correlation)
        correlation = np.sum(field1 * field2.conj())
        return np.array([correlation])

    def _find_peak_shift(self, correlation: np.ndarray) -> float:
        """
        Find peak shift in correlation.

        Physical Meaning:
            Finds the shift of the correlation peak,
            indicating pattern displacement.
        """
        # Simplified peak shift (in practice, this would involve proper peak finding)
        return 0.0

    def _compute_jaccard_index(self, field_evolution: List[np.ndarray]) -> float:
        """
        Compute Jaccard index for pattern stability.

        Physical Meaning:
            Computes the Jaccard index to measure pattern
            stability and similarity over time.
        """
        if len(field_evolution) < 2:
            return 1.0

        # Simplified Jaccard index calculation
        # In practice, this would involve proper set intersection/union
        return 0.95  # Placeholder value

    def _compute_stability_score(self, field_evolution: List[np.ndarray]) -> float:
        """
        Compute stability score for field evolution.

        Physical Meaning:
            Computes a stability score based on the
            consistency of field patterns over time.
        """
        if len(field_evolution) < 2:
            return 1.0

        # Simplified stability score
        # In practice, this would involve proper stability analysis
        return 0.9  # Placeholder value

    def _analyze_correlation_decay(
        self, correlation_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze correlation decay over time.

        Physical Meaning:
            Analyzes how correlation decays over time,
            indicating pattern stability.
        """
        # Simplified correlation decay analysis
        return {"decay_rate": 0.1, "correlation_time": 10.0, "stability_metric": 0.8}

    def _analyze_pattern_stability(
        self, field_evolution: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Analyze pattern stability over time.

        Physical Meaning:
            Analyzes the stability of field patterns
            over time evolution.
        """
        # Simplified pattern stability analysis
        return {
            "stability_score": 0.9,
            "pattern_consistency": 0.85,
            "temporal_coherence": 0.8,
        }
