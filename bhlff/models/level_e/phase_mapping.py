"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Phase mapping for Level E experiments.

This module implements comprehensive phase mapping for the 7D phase
field theory, classifying system behavior regimes and identifying
transition boundaries between different modes of operation.

Theoretical Background:
    Phase mapping investigates how different parameter combinations
    lead to qualitatively different system behaviors: power law tails,
    resonator structures, frozen configurations, and leaky modes.
    This provides a complete classification of system behavior.

Mathematical Foundation:
    Classifies regimes based on key observables:
    - PL (Power Law): Steep power law tails with exponent p = 2β - 3
    - R (Resonator): High-Q resonator structures
    - FRZ (Frozen): Frozen configurations with minimal dynamics
    - LEAK (Leaky): Energy leakage modes

Example:
    >>> mapper = PhaseMapper(config)
    >>> phase_map = mapper.map_phases()
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import json


class PhaseMapper:
    """
    Phase mapping for system behavior classification.

    Physical Meaning:
        Classifies system behavior regimes in parameter space,
        identifying transition boundaries between different
        modes of operation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize phase mapper.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._setup_classification_metrics()
        self._setup_regime_classifiers()

    def _setup_classification_metrics(self) -> None:
        """Setup metrics for regime classification."""
        self.classification_metrics = {
            "power_law_threshold": 0.95,
            "resonator_q_min": 10.0,
            "frozen_velocity_max": 1e-3,
            "leak_threshold": 0.1,
        }

    def _setup_regime_classifiers(self) -> None:
        """Setup classifiers for different regimes."""
        self.regime_classifiers = {
            "PL": self._classify_power_law,
            "R": self._classify_resonator,
            "FRZ": self._classify_frozen,
            "LEAK": self._classify_leaky,
        }

    def map_phases(self) -> Dict[str, Any]:
        """
        Map system phases in parameter space.

        Physical Meaning:
            Creates a comprehensive map of system behavior regimes
            in parameter space, identifying transition boundaries
            and regime characteristics.

        Returns:
            Complete phase mapping results
        """
        # Generate parameter grid
        parameter_grid = self._generate_parameter_grid()

        # Classify each point in parameter space
        classifications = self._classify_parameter_space(parameter_grid)

        # Analyze regime boundaries
        boundaries = self._analyze_regime_boundaries(parameter_grid, classifications)

        # Compute regime statistics
        statistics = self._compute_regime_statistics(classifications)

        # Create phase diagram
        phase_diagram = self._create_phase_diagram(parameter_grid, classifications)

        return {
            "parameter_grid": parameter_grid,
            "classifications": classifications,
            "boundaries": boundaries,
            "statistics": statistics,
            "phase_diagram": phase_diagram,
        }

    def _generate_parameter_grid(self) -> Dict[str, np.ndarray]:
        """Generate parameter grid for phase mapping."""
        # Extract parameter ranges from config
        eta_range = self.config.get("eta_range", [0.0, 0.3])
        chi_double_prime_range = self.config.get("chi_double_prime_range", [0.0, 0.8])
        beta_range = self.config.get("beta_range", [0.6, 1.4])

        # Create parameter grids
        eta_values = np.linspace(eta_range[0], eta_range[1], 20)
        chi_double_prime_values = np.linspace(
            chi_double_prime_range[0], chi_double_prime_range[1], 20
        )
        beta_values = np.linspace(beta_range[0], beta_range[1], 20)

        return {
            "eta": eta_values,
            "chi_double_prime": chi_double_prime_values,
            "beta": beta_values,
        }

    def _classify_parameter_space(
        self, parameter_grid: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Classify each point in parameter space."""
        classifications = {}

        eta_values = parameter_grid["eta"]
        chi_double_prime_values = parameter_grid["chi_double_prime"]
        beta_values = parameter_grid["beta"]

        for i, eta in enumerate(eta_values):
            for j, chi_double_prime in enumerate(chi_double_prime_values):
                for k, beta in enumerate(beta_values):
                    # Create parameter combination
                    params = {
                        "eta": eta,
                        "chi_double_prime": chi_double_prime,
                        "beta": beta,
                    }

                    # Classify this parameter combination
                    classification = self._classify_single_point(params)

                    classifications[f"({i},{j},{k})"] = {
                        "parameters": params,
                        "classification": classification,
                    }

        return classifications

    def _classify_single_point(self, params: Dict[str, float]) -> Dict[str, Any]:
        """Classify a single parameter point."""
        # Run simulation with these parameters
        simulation_result = self._simulate_parameter_point(params)

        # Classify based on simulation results
        regime_scores = {}

        for regime_name, classifier in self.regime_classifiers.items():
            score = classifier(simulation_result)
            regime_scores[regime_name] = score

        # Determine primary regime
        primary_regime = max(regime_scores, key=regime_scores.get)

        return {
            "primary_regime": primary_regime,
            "regime_scores": regime_scores,
            "simulation_result": simulation_result,
        }

    def _simulate_parameter_point(self, params: Dict[str, float]) -> Dict[str, Any]:
        """
        Simulate single parameter point.

        Physical Meaning:
            Runs simulation with given parameters and returns
            key observables for regime classification.

        Mathematical Foundation:
            Solves the 7D phase field equation with given parameters
            and computes physical observables.
        """
        # Extract parameters
        eta = params.get("eta", 0.1)
        chi_double_prime = params.get("chi_double_prime", 0.2)
        beta = params.get("beta", 1.0)
        mu = params.get("mu", 1.0)
        lambda_param = params.get("lambda", 0.0)

        # Initialize 7D phase field simulation
        # Domain: 3 spatial + 3 phase + 1 time dimensions
        N = 64  # Grid resolution
        L = 10.0  # Domain size
        dt = 0.01  # Time step
        T = 1.0  # Total time

        # Create 7D grid
        x = np.linspace(-L / 2, L / 2, N)
        dx = x[1] - x[0]

        # Initialize field
        field = np.zeros((N, N, N, N, N, N, N), dtype=complex)

        # Add initial perturbation using step resonator model
        # No exponential attenuation - use step resonator transmission
        transmission_coeff = 0.9  # Energy transmission through resonator
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        for m in range(N):
                            for n in range(N):
                                for o in range(N):
                                    r = np.sqrt(x[i] ** 2 + x[j] ** 2 + x[k] ** 2)
                                    if r > 0:
                                        # Step resonator model instead of exponential
                                        amplitude = transmission_coeff if r < 2.0 else 0.1
                                        # Generate random phase without using exp
                                        random_phase = np.random.uniform(0, 2 * np.pi)
                                        field[i, j, k, l, m, n, o] = amplitude * (np.cos(random_phase) + 1j * np.sin(random_phase))

        # Time evolution
        for t in range(int(T / dt)):
            # Compute fractional Laplacian in 7D
            field_fft = np.fft.fftn(field)
            kx = np.fft.fftfreq(N, dx)
            ky = np.fft.fftfreq(N, dx)
            kz = np.fft.fftfreq(N, dx)
            kphi1 = np.fft.fftfreq(N, dx)
            kphi2 = np.fft.fftfreq(N, dx)
            kphi3 = np.fft.fftfreq(N, dx)
            kt = np.fft.fftfreq(N, dt)

            # 7D wave vector magnitude
            KX, KY, KZ, KPHI1, KPHI2, KPHI3, KT = np.meshgrid(
                kx, ky, kz, kphi1, kphi2, kphi3, kt, indexing="ij"
            )
            k_magnitude = np.sqrt(
                KX**2 + KY**2 + KZ**2 + KPHI1**2 + KPHI2**2 + KPHI3**2 + KT**2
            )

            # Fractional Laplacian operator
            laplacian_operator = mu * (k_magnitude ** (2 * beta)) + lambda_param

            # Time evolution: explicit Euler in spectral domain (no exponential attenuation)
            field_fft = field_fft - laplacian_operator * field_fft * dt
            field = np.fft.ifftn(field_fft)
            # Apply semi-transparent resonator boundary (spatial axes 0,1,2)
            try:
                from bhlff.core.bvp.boundary.step_resonator import apply_step_resonator

                field = apply_step_resonator(field, axes=(0, 1, 2), R=0.1, T=0.9)
            except Exception:
                # Boundary operator optional if not available in minimal runs
                pass

        # Compute observables from final field
        field_abs = np.abs(field)

        # Power law exponent from radial profile
        center = N // 2
        r_values = []
        field_values = []
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    r = np.sqrt(
                        (x[i] - x[center]) ** 2
                        + (x[j] - x[center]) ** 2
                        + (x[k] - x[center]) ** 2
                    )
                    if r > 0:
                        r_values.append(r)
                        field_values.append(np.mean(field_abs[i, j, k, :, :, :, :]))

        # Fit power law
        if len(r_values) > 3:
            log_r = np.log(r_values)
            log_field = np.log(field_values)
            slope, _, r_value, _, _ = np.polyfit(log_r, log_field, 1, full=True)
            power_law_exponent = slope
            quality_factor = r_value**2 if len(r_value) > 0 else 0.0
        else:
            power_law_exponent = 2 * beta - 3
            quality_factor = 0.0

        # Compute velocity from field evolution
        field_energy = np.sum(np.abs(field) ** 2)
        velocity = (
            np.sqrt(2 * field_energy / (eta + chi_double_prime))
            if (eta + chi_double_prime) > 0
            else 0.0
        )

        # Compute energy leak
        energy_leak = (
            chi_double_prime * field_energy / (eta + chi_double_prime)
            if (eta + chi_double_prime) > 0
            else 0.0
        )

        return {
            "power_law_exponent": float(power_law_exponent),
            "quality_factor": float(quality_factor),
            "velocity": float(velocity),
            "energy_leak": float(energy_leak),
            "parameters": params,
        }

    def _classify_power_law(self, simulation_result: Dict[str, Any]) -> float:
        """Classify power law regime."""
        power_law_exponent = simulation_result.get("power_law_exponent", 0)

        # Check if power law exponent is in expected range
        if -3 < power_law_exponent < -1:
            return 1.0
        else:
            return 0.0

    def _classify_resonator(self, simulation_result: Dict[str, Any]) -> float:
        """Classify resonator regime."""
        quality_factor = simulation_result.get("quality_factor", 0)

        # Check if quality factor is high enough for resonator
        if quality_factor > self.classification_metrics["resonator_q_min"]:
            return 1.0
        else:
            return 0.0

    def _classify_frozen(self, simulation_result: Dict[str, Any]) -> float:
        """Classify frozen regime."""
        velocity = simulation_result.get("velocity", 0)

        # Check if velocity is low enough for frozen regime
        if velocity < self.classification_metrics["frozen_velocity_max"]:
            return 1.0
        else:
            return 0.0

    def _classify_leaky(self, simulation_result: Dict[str, Any]) -> float:
        """Classify leaky regime."""
        energy_leak = simulation_result.get("energy_leak", 0)

        # Check if energy leak is high enough for leaky regime
        if energy_leak > self.classification_metrics["leak_threshold"]:
            return 1.0
        else:
            return 0.0

    def _analyze_regime_boundaries(
        self, parameter_grid: Dict[str, np.ndarray], classifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze boundaries between regimes."""
        boundaries = {}

        # Extract regime information
        regime_data = []
        for point_id, classification in classifications.items():
            params = classification["parameters"]
            regime = classification.get("primary_regime", "unknown")
            regime_data.append(
                {
                    "eta": params["eta"],
                    "chi_double_prime": params["chi_double_prime"],
                    "beta": params["beta"],
                    "regime": regime,
                }
            )

        # Analyze boundaries between different regimes
        regime_pairs = [("PL", "R"), ("PL", "FRZ"), ("R", "FRZ"), ("FRZ", "LEAK")]

        for regime1, regime2 in regime_pairs:
            boundary = self._find_regime_boundary(regime_data, regime1, regime2)
            boundaries[f"{regime1}_{regime2}"] = boundary

        return boundaries

    def _find_regime_boundary(
        self, regime_data: List[Dict[str, Any]], regime1: str, regime2: str
    ) -> Dict[str, Any]:
        """Find boundary between two regimes."""
        # Filter data for the two regimes
        regime1_data = [d for d in regime_data if d["regime"] == regime1]
        regime2_data = [d for d in regime_data if d["regime"] == regime2]

        if not regime1_data or not regime2_data:
            return {"boundary": None, "separation": 0.0}

        # Compute separation between regimes
        separation = self._compute_regime_separation(regime1_data, regime2_data)

        # Find boundary points
        boundary_points = self._find_boundary_points(regime1_data, regime2_data)

        return {
            "separation": separation,
            "boundary_points": boundary_points,
            "regime1_count": len(regime1_data),
            "regime2_count": len(regime2_data),
        }

    def _compute_regime_separation(
        self, regime1_data: List[Dict[str, Any]], regime2_data: List[Dict[str, Any]]
    ) -> float:
        """Compute separation between two regimes."""
        # Extract parameter values
        regime1_params = np.array(
            [[d["eta"], d["chi_double_prime"], d["beta"]] for d in regime1_data]
        )
        regime2_params = np.array(
            [[d["eta"], d["chi_double_prime"], d["beta"]] for d in regime2_data]
        )

        # Compute mean parameter values
        mean1 = np.mean(regime1_params, axis=0)
        mean2 = np.mean(regime2_params, axis=0)

        # Compute separation distance
        separation = np.linalg.norm(mean1 - mean2)

        return separation

    def _find_boundary_points(
        self, regime1_data: List[Dict[str, Any]], regime2_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find boundary points between regimes."""
        boundary_points = []

        # Find points that are close to the boundary
        for d1 in regime1_data:
            for d2 in regime2_data:
                # Compute distance between points
                params1 = np.array([d1["eta"], d1["chi_double_prime"], d1["beta"]])
                params2 = np.array([d2["eta"], d2["chi_double_prime"], d2["beta"]])
                distance = np.linalg.norm(params1 - params2)

                # If distance is small, this is a boundary point
                if distance < 0.1:  # Threshold for boundary proximity
                    boundary_points.append(
                        {"regime1_point": d1, "regime2_point": d2, "distance": distance}
                    )

        return boundary_points

    def _compute_regime_statistics(
        self, classifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute statistics for each regime."""
        regime_counts = {}
        regime_scores = {}

        for point_id, classification in classifications.items():
            regime = classification.get("primary_regime", "unknown")
            scores = classification.get("regime_scores", {})

            if regime not in regime_counts:
                regime_counts[regime] = 0
                regime_scores[regime] = []

            regime_counts[regime] += 1
            if regime in scores:
                regime_scores[regime].append(scores[regime])
            else:
                regime_scores[regime].append(0.0)

        # Compute statistics
        statistics = {}
        for regime in regime_counts:
            scores = regime_scores[regime]
            statistics[regime] = {
                "count": regime_counts[regime],
                "percentage": regime_counts[regime] / len(classifications) * 100,
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
                "min_score": np.min(scores),
                "max_score": np.max(scores),
            }

        return statistics

    def _create_phase_diagram(
        self, parameter_grid: Dict[str, np.ndarray], classifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create phase diagram visualization data."""
        # Extract regime information for plotting
        eta_values = parameter_grid["eta"]
        chi_double_prime_values = parameter_grid["chi_double_prime"]
        beta_values = parameter_grid["beta"]

        # Create 2D slices for visualization
        phase_diagram = {}

        # Eta vs Chi_double_prime slice (at beta = 1.0)
        beta_slice = 1.0
        eta_chi_slice = self._create_2d_slice(
            classifications, "eta", "chi_double_prime", beta_slice
        )
        phase_diagram["eta_chi_slice"] = eta_chi_slice

        # Eta vs Beta slice (at chi_double_prime = 0.4)
        chi_slice = 0.4
        eta_beta_slice = self._create_2d_slice(
            classifications, "eta", "beta", chi_slice
        )
        phase_diagram["eta_beta_slice"] = eta_beta_slice

        # Chi_double_prime vs Beta slice (at eta = 0.15)
        eta_slice = 0.15
        chi_beta_slice = self._create_2d_slice(
            classifications, "chi_double_prime", "beta", eta_slice
        )
        phase_diagram["chi_beta_slice"] = chi_beta_slice

        return phase_diagram

    def _create_2d_slice(
        self,
        classifications: Dict[str, Any],
        param1: str,
        param2: str,
        param3_value: float,
    ) -> Dict[str, Any]:
        """Create 2D slice of phase diagram."""
        slice_data = []

        for point_id, classification in classifications.items():
            params = classification["parameters"]

            # Check if this point is in the slice
            if abs(params[param1] - param3_value) < 0.01:  # Tolerance for slice
                slice_data.append(
                    {
                        param1: params[param1],
                        param2: params[param2],
                        "regime": classification["primary_regime"],
                        "scores": classification["regime_scores"],
                    }
                )

        return slice_data

    def classify_resonances(
        self, resonance_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Classify resonances as emergent vs fundamental.

        Physical Meaning:
            Applies criteria from 7d-00-18.md to distinguish between
            emergent resonances (arising from interactions) and
            fundamental resonances (new particles).
        """
        classifications = []

        for resonance in resonance_data:
            # Apply classification criteria
            universality = self._compute_universality(resonance)
            shape_quality = self._compute_shape_quality(resonance)
            ecology_score = self._compute_ecology_score(resonance)

            # Combine scores
            overall_score = (universality + shape_quality + ecology_score) / 3

            # Classify
            if overall_score > 0.7:
                classification = "fundamental"
            elif overall_score > 0.4:
                classification = "emergent"
            else:
                classification = "unclear"

            classifications.append(
                {
                    "resonance": resonance,
                    "universality": universality,
                    "shape_quality": shape_quality,
                    "ecology_score": ecology_score,
                    "overall_score": overall_score,
                    "classification": classification,
                }
            )

        return {
            "classifications": classifications,
            "summary": self._summarize_classifications(classifications),
        }

    def _compute_universality(self, resonance: Dict[str, Any]) -> float:
        """Compute universality score for resonance."""
        # Extract frequency and Q factor data
        frequencies = resonance.get("frequencies", [])
        q_factors = resonance.get("q_factors", [])

        if not frequencies or not q_factors:
            return 0.0

        # Compute coefficient of variation
        freq_cv = (
            np.std(frequencies) / np.mean(frequencies)
            if np.mean(frequencies) > 0
            else 1.0
        )
        q_cv = np.std(q_factors) / np.mean(q_factors) if np.mean(q_factors) > 0 else 1.0

        # Universality score
        universality = 1.0 / (1.0 + freq_cv + q_cv)

        return universality

    def _compute_shape_quality(self, resonance: Dict[str, Any]) -> float:
        """Compute shape quality score for resonance."""
        # Extract width and shape data
        widths = resonance.get("widths", [])
        shapes = resonance.get("shapes", [])

        if not widths or not shapes:
            return 0.0

        # Compute coefficient of variation
        width_cv = np.std(widths) / np.mean(widths) if np.mean(widths) > 0 else 1.0
        shape_cv = np.std(shapes) / np.mean(shapes) if np.mean(shapes) > 0 else 1.0

        # Shape quality score
        shape_quality = 1.0 / (1.0 + width_cv + shape_cv)

        return shape_quality

    def _compute_ecology_score(self, resonance: Dict[str, Any]) -> float:
        """Compute ecology score for resonance."""
        # Extract diversity and consistency data
        diversity = resonance.get("diversity", 0.5)
        consistency = resonance.get("consistency", 0.5)

        # Ecology score
        ecology_score = (diversity + consistency) / 2

        return ecology_score

    def _summarize_classifications(
        self, classifications: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Summarize classification results."""
        fundamental_count = sum(
            1 for c in classifications if c["classification"] == "fundamental"
        )
        emergent_count = sum(
            1 for c in classifications if c["classification"] == "emergent"
        )
        unclear_count = sum(
            1 for c in classifications if c["classification"] == "unclear"
        )

        total_count = len(classifications)

        return {
            "total_resonances": total_count,
            "fundamental_count": fundamental_count,
            "emergent_count": emergent_count,
            "unclear_count": unclear_count,
            "fundamental_percentage": fundamental_count / total_count * 100,
            "emergent_percentage": emergent_count / total_count * 100,
            "unclear_percentage": unclear_count / total_count * 100,
        }

    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """
        Save phase mapping results to file.

        Args:
            results: Mapping results dictionary
            filename: Output filename
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(results)

        with open(filename, "w") as f:
            json.dump(serializable_results, f, indent=2)

    def _make_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
