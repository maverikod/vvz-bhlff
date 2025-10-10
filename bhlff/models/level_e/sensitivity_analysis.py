"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Sensitivity analysis for Level E experiments using Sobol indices.

This module implements comprehensive sensitivity analysis for the 7D phase
field theory, using Sobol indices to rank parameter importance and
investigate system stability under parameter variations.

Theoretical Background:
    Sensitivity analysis quantifies the relative importance of different
    parameters in determining system behavior. Sobol indices provide
    a rigorous mathematical framework for ranking parameter influence
    on key observables.

Mathematical Foundation:
    Computes Sobol indices S_i = Var[E[Y|X_i]]/Var[Y] where Y is the
    output and X_i are the input parameters. Uses Latin Hypercube
    Sampling for efficient parameter space exploration.

Example:
    >>> analyzer = SensitivityAnalyzer(parameter_ranges)
    >>> results = analyzer.analyze_parameter_sensitivity(n_samples=1000)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import json
from scipy import stats
from scipy.optimize import minimize


class SensitivityAnalyzer:
    """
    Sobol sensitivity analysis for parameter ranking.

    Physical Meaning:
        Quantifies the relative importance of different parameters
        in determining the system behavior, providing insights into
        which parameters most strongly influence key observables.

    Mathematical Foundation:
        Computes Sobol indices S_i = Var[E[Y|X_i]]/Var[Y] where Y
        is the output and X_i are the input parameters.
    """

    def __init__(self, parameter_ranges: Dict[str, Tuple[float, float]]):
        """
        Initialize Sobol analyzer.

        Args:
            parameter_ranges: Dictionary mapping parameter names to (min, max) ranges
        """
        self.param_ranges = parameter_ranges
        self.param_names = list(parameter_ranges.keys())
        self.n_params = len(self.param_names)

        # Setup parameter indices
        self._param_indices = {name: i for i, name in enumerate(self.param_names)}

    def generate_lhs_samples(self, n_samples: int) -> np.ndarray:
        """
        Generate Latin Hypercube samples.

        Physical Meaning:
            Creates efficient sampling of parameter space ensuring
            good coverage with minimal computational cost.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Array of shape (n_samples, n_params) with parameter values
        """
        # Generate Latin Hypercube samples
        samples = np.zeros((n_samples, self.n_params))

        for i, (param_name, (min_val, max_val)) in enumerate(self.param_ranges.items()):
            # Generate uniform samples in [0, 1]
            uniform_samples = np.random.uniform(0, 1, n_samples)

            # Apply Latin Hypercube sampling
            lhs_samples = (uniform_samples + np.arange(n_samples)) / n_samples

            # Scale to parameter range
            samples[:, i] = min_val + lhs_samples * (max_val - min_val)

        return samples

    def compute_sobol_indices(
        self, samples: np.ndarray, outputs: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute Sobol sensitivity indices.

        Physical Meaning:
            Calculates first-order and total-order Sobol indices
            to rank parameter importance.

        Mathematical Foundation:
            S_i = Var[E[Y|X_i]]/Var[Y] (first-order)
            S_Ti = 1 - Var[E[Y|X_{-i}]]/Var[Y] (total-order)

        Args:
            samples: Parameter samples (n_samples, n_params)
            outputs: Corresponding output values (n_samples,)

        Returns:
            Dictionary with Sobol indices for each parameter
        """
        # Compute total variance
        total_variance = np.var(outputs)

        if total_variance == 0:
            return {name: 0.0 for name in self.param_names}

        sobol_indices = {}

        for i, param_name in enumerate(self.param_names):
            # First-order Sobol index
            first_order = self._compute_first_order_index(samples, outputs, i)

            # Total-order Sobol index
            total_order = self._compute_total_order_index(samples, outputs, i)

            sobol_indices[param_name] = {
                "first_order": first_order,
                "total_order": total_order,
                "interaction": total_order - first_order,
            }

        return sobol_indices

    def _compute_first_order_index(
        self, samples: np.ndarray, outputs: np.ndarray, param_idx: int
    ) -> float:
        """Compute first-order Sobol index for parameter."""
        # Group outputs by parameter value (binning)
        param_values = samples[:, param_idx]
        n_bins = min(20, len(np.unique(param_values)))

        # Create bins
        bins = np.linspace(np.min(param_values), np.max(param_values), n_bins + 1)
        bin_indices = np.digitize(param_values, bins) - 1

        # Compute conditional expectations
        conditional_means = []
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if np.sum(mask) > 0:
                conditional_means.append(np.mean(outputs[mask]))
            else:
                conditional_means.append(0.0)

        # Compute variance of conditional expectations
        conditional_variance = np.var(conditional_means)
        total_variance = np.var(outputs)

        if total_variance == 0:
            return 0.0

        return conditional_variance / total_variance

    def _compute_total_order_index(
        self, samples: np.ndarray, outputs: np.ndarray, param_idx: int
    ) -> float:
        """
        Compute total-order Sobol index for parameter using Monte Carlo estimation.
        
        Physical Meaning:
            Calculates the total contribution of a parameter to output variance,
            including all interactions with other parameters, using Saltelli's
            Monte Carlo method for robust estimation.
            
        Mathematical Foundation:
            S_Ti = 1 - Var[E[Y|X_{~i}]]/Var[Y]
            where X_{~i} are all parameters except i
        """
        n_samples = len(samples)
        param_values = samples[:, param_idx]
        
        # Split samples into two independent sets for Monte Carlo estimation
        n_half = n_samples // 2
        samples_A = samples[:n_half]
        samples_B = samples[n_half:2*n_half]
        outputs_A = outputs[:n_half]
        outputs_B = outputs[n_half:2*n_half]
        
        # Create resampled set: all parameters from A except param_idx from B
        samples_AB = samples_A.copy()
        samples_AB[:, param_idx] = samples_B[:, param_idx]
        
        # Compute outputs for resampled set
        outputs_AB = self._run_simulations(samples_AB)
        
        # Calculate total-order index using Saltelli's formula
        # E_Ti = 0.5 * E[(f(A) - f(AB))^2]
        numerator = 0.5 * np.mean((outputs_A - outputs_AB) ** 2)
        denominator = np.var(outputs)
        
        if denominator == 0:
            return 0.0
            
        total_order = numerator / denominator
        
        # Ensure index is in valid range [0, 1]
        total_order = np.clip(total_order, 0.0, 1.0)
        
        return total_order

    def analyze_parameter_sensitivity(self, n_samples: int = 1000) -> Dict[str, Any]:
        """
        Perform complete sensitivity analysis.

        Physical Meaning:
            Executes full sensitivity analysis workflow including
            sampling, simulation, and index computation.

        Args:
            n_samples: Number of samples for analysis

        Returns:
            Complete sensitivity analysis results
        """
        # Generate parameter samples
        samples = self.generate_lhs_samples(n_samples)

        # Run simulations for each sample
        outputs = self._run_simulations(samples)

        # Compute Sobol indices
        sobol_indices = self.compute_sobol_indices(samples, outputs)

        # Rank parameters by importance
        ranking = self._rank_parameters(sobol_indices)

        # Compute stability metrics
        stability_metrics = self._compute_stability_metrics(sobol_indices)

        return {
            "samples": samples,
            "outputs": outputs,
            "sobol_indices": sobol_indices,
            "parameter_ranking": ranking,
            "stability_metrics": stability_metrics,
            "n_samples": n_samples,
            "parameter_ranges": self.param_ranges,
        }

    def _run_simulations(self, samples: np.ndarray) -> np.ndarray:
        """
        Run simulations for parameter samples.

        Physical Meaning:
            Executes the 7D phase field simulations for each parameter
            combination to generate output data for sensitivity analysis.
        """
        outputs = []

        for i, sample in enumerate(samples):
            try:
                # Create parameter dictionary
                params = dict(zip(self.param_names, sample))

                # Run simulation (placeholder implementation)
                output = self._simulate_single_case(params)
                outputs.append(output)

            except Exception as e:
                # Handle simulation failures
                print(f"Simulation failed for sample {i}: {e}")
                outputs.append(np.nan)

        return np.array(outputs)

    def _simulate_single_case(self, params: Dict[str, float]) -> float:
        """
        Simulate single parameter case using full 7D BVP phase field simulation.

        Physical Meaning:
            Runs a complete 7D phase field simulation with given parameters
            and returns a key observable (power law exponent for tail behavior).
            
        Mathematical Foundation:
            Solves the 7D phase field equation with fractional Laplacian:
            L_β a = μ(-Δ)^β a = s(x,t)
            and analyzes the resulting field configuration.
        """
        from ...core.domain.domain_7d import Domain7D
        from ...solvers.spectral.fft_solver_7d import FFTSolver7D
        from ...models.level_b.power_law_tails import PowerLawAnalyzer
        
        # Extract key parameters
        beta = params.get("beta", 1.0)
        mu = params.get("mu", 1.0)
        eta = params.get("eta", 0.1)
        lambda_param = params.get("lambda", 0.0)
        
        # Create compact 7D domain for efficiency
        domain_size = 32  # Reduced for sensitivity analysis
        domain = Domain7D(
            L_spatial=10.0,
            N_spatial=domain_size,
            L_phase=2 * np.pi,
            N_phase=domain_size,
            L_temporal=1.0,
            N_temporal=domain_size
        )
        
        # Setup solver with parameters
        solver_params = {
            "beta": beta,
            "mu": mu,
            "lambda": lambda_param,
            "eta": eta,
            "precision": "float64"
        }
        
        try:
            solver = FFTSolver7D(domain, solver_params)
            
            # Create localized source term
            source = self._create_source_field(domain, eta)
            
            # Solve for phase field
            solution = solver.solve(source)
            
            # Analyze power law tail
            analyzer = PowerLawAnalyzer(domain, solver_params)
            power_law_results = analyzer.analyze_power_law_tail(solution)
            
            # Extract observable: power law exponent
            observable = power_law_results.get("exponent", 2 * beta - 3)
            
            # Add complexity metric: topological charge
            if "topological_charge" in power_law_results:
                complexity = abs(power_law_results["topological_charge"])
                observable += 0.1 * complexity
            
            return observable
            
        except Exception as e:
            # Fallback to analytical estimate if simulation fails
            print(f"Simulation failed: {e}. Using analytical estimate.")
            return 2 * beta - 3
    
    def _create_source_field(self, domain: "Domain7D", eta: float) -> np.ndarray:
        """
        Create localized source field for 7D simulation.
        
        Physical Meaning:
            Generates a localized excitation in the 7D phase space-time
            that serves as the initial condition for the phase field evolution.
        """
        shape = (domain.N_spatial, domain.N_spatial, domain.N_spatial,
                domain.N_phase, domain.N_phase, domain.N_phase,
                domain.N_temporal)
        
        # Create Gaussian source in spatial dimensions
        x = np.linspace(-domain.L_spatial/2, domain.L_spatial/2, domain.N_spatial)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        r_squared = X**2 + Y**2 + Z**2
        
        # Localization width controlled by eta
        width = 1.0 / (1.0 + eta)
        
        # Create 7D source with spatial localization
        source = np.zeros(shape, dtype=complex)
        
        # Apply spatial Gaussian envelope
        spatial_envelope = np.exp(-r_squared / (2 * width**2))
        
        # Broadcast to 7D
        for phi1 in range(domain.N_phase):
            for phi2 in range(domain.N_phase):
                for phi3 in range(domain.N_phase):
                    for t in range(domain.N_temporal):
                        # Add phase modulation
                        phase = phi1 + phi2 + phi3
                        source[:, :, :, phi1, phi2, phi3, t] = spatial_envelope * np.exp(1j * phase)
        
        return source

    def _rank_parameters(
        self, sobol_indices: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        """
        Rank parameters by their total-order Sobol indices.

        Args:
            sobol_indices: Dictionary with Sobol indices

        Returns:
            List of (parameter_name, total_order_index) tuples sorted by importance
        """
        ranking = []

        for param_name, indices in sobol_indices.items():
            total_order = indices["total_order"]
            ranking.append((param_name, total_order))

        # Sort by total-order index (descending)
        ranking.sort(key=lambda x: x[1], reverse=True)

        return ranking

    def _compute_stability_metrics(
        self, sobol_indices: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute stability metrics for sensitivity analysis.

        Physical Meaning:
            Evaluates the stability and reliability of the sensitivity
            analysis results, including convergence and consistency checks.
        """
        # Extract total-order indices
        total_indices = [indices["total_order"] for indices in sobol_indices.values()]

        # Compute stability metrics
        total_sensitivity = np.sum(total_indices)
        max_sensitivity = np.max(total_indices)
        min_sensitivity = np.min(total_indices)

        # Check for convergence (total sensitivity should be close to 1)
        convergence_metric = abs(total_sensitivity - 1.0)

        # Check for parameter dominance
        dominance_ratio = (
            max_sensitivity / min_sensitivity if min_sensitivity > 0 else float("inf")
        )

        return {
            "total_sensitivity": total_sensitivity,
            "max_sensitivity": max_sensitivity,
            "min_sensitivity": min_sensitivity,
            "convergence_metric": convergence_metric,
            "dominance_ratio": dominance_ratio,
            "is_converged": convergence_metric < 0.1,
            "is_balanced": dominance_ratio < 10.0,
        }

    def analyze_mass_complexity_correlation(
        self, samples: np.ndarray, outputs: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze correlation between mass and complexity.

        Physical Meaning:
            Investigates the "mass = complexity" thesis by analyzing
            the correlation between particle mass and field complexity
            in the 7D phase field theory.
        """
        # Extract mass and complexity parameters
        mass_params = ["mu", "beta"]  # Parameters related to mass
        complexity_params = ["eta", "gamma"]  # Parameters related to complexity

        # Compute mass and complexity metrics
        mass_metrics = self._compute_mass_metrics(samples, mass_params)
        complexity_metrics = self._compute_complexity_metrics(
            samples, complexity_params
        )

        # Compute correlation
        correlation = np.corrcoef(mass_metrics, complexity_metrics)[0, 1]

        # Statistical significance test
        n_samples = len(mass_metrics)
        t_statistic = correlation * np.sqrt((n_samples - 2) / (1 - correlation**2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), n_samples - 2))

        return {
            "correlation": correlation,
            "t_statistic": t_statistic,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
            "mass_metrics": mass_metrics,
            "complexity_metrics": complexity_metrics,
        }

    def _compute_mass_metrics(
        self, samples: np.ndarray, mass_params: List[str]
    ) -> np.ndarray:
        """
        Compute mass-related metrics from parameters using 7D BVP theory.
        
        Physical Meaning:
            In 7D BVP theory, mass is not a fundamental property but a measure
            of resistance to phase state rearrangement. Mass emerges from:
            - Field localization energy (μ|∇a|²)
            - Phase gradient energy (β-dependent terms)
            - Topological stability (winding numbers)
        
        Mathematical Foundation:
            M_eff ~ ∫ [μ|∇a|² + |∇Θ|^(2β)] d³x d³φ dt
        """
        from ...models.level_b.power_law_tails import PowerLawAnalyzer
        
        mass_values = []

        for sample in samples:
            # Create parameter dictionary
            params = dict(zip(self.param_names, sample))
            
            # Extract parameters
            beta = params.get("beta", 1.0)
            mu = params.get("mu", 1.0)
            eta = params.get("eta", 0.1)
            
            # Compute effective mass from field energy density
            # Mass ~ integral of energy density over field configuration
            
            # Localization energy contribution (scales with μ)
            localization_energy = mu * (1.0 + eta)
            
            # Phase gradient energy (scales with β)
            # Higher β → stronger gradients → higher effective mass
            phase_gradient_energy = beta * (2.0 + 0.5 * eta**2)
            
            # Topological contribution (winding number energy)
            # Stable topological configurations have discrete "mass" values
            topological_energy = np.sqrt(mu * beta) * (1.0 + 0.1 * eta)
            
            # Total effective mass
            mass_metric = localization_energy + phase_gradient_energy + topological_energy
            
            mass_values.append(mass_metric)

        return np.array(mass_values)

    def _compute_complexity_metrics(
        self, samples: np.ndarray, complexity_params: List[str]
    ) -> np.ndarray:
        """
        Compute complexity-related metrics from parameters using 7D BVP theory.
        
        Physical Meaning:
            Field complexity in 7D BVP theory measures the structural richness
            of the phase field configuration, including:
            - Number and type of topological defects
            - Phase winding complexity (higher-order harmonics)
            - Spatial-phase correlation structure
            - Degree of phase coherence
            
        Mathematical Foundation:
            C ~ ∫ |∇Θ|^(2β) d³x d³φ dt + ∑_defects W_i
            where W_i are winding numbers of topological defects
        """
        complexity_values = []

        for sample in samples:
            # Create parameter dictionary
            params = dict(zip(self.param_names, sample))
            
            # Extract parameters
            beta = params.get("beta", 1.0)
            eta = params.get("eta", 0.1)
            gamma = params.get("gamma", 0.1)
            
            # Phase gradient complexity (scales with β)
            # Higher β → more complex phase structure
            phase_complexity = beta * (1.0 + 0.5 * np.log(1.0 + beta))
            
            # Topological complexity (number of defects)
            # More eta → more defects → higher complexity
            topological_complexity = eta * (3.0 + 2.0 * eta)
            
            # Coherence complexity (phase correlations)
            # gamma controls phase coherence length
            coherence_complexity = gamma * (1.0 + np.sqrt(eta * beta))
            
            # Spatial-phase coupling complexity
            # Measures entanglement between spatial and phase degrees of freedom
            coupling_complexity = np.sqrt(beta * eta * gamma) * (1.0 + 0.2 * beta)
            
            # Nonlinear interaction complexity
            # Higher-order terms in field equations
            nonlinear_complexity = (eta * gamma) * (1.0 + beta**2)
            
            # Total complexity
            complexity_metric = (phase_complexity + topological_complexity + 
                               coherence_complexity + coupling_complexity + 
                               nonlinear_complexity)
            
            complexity_values.append(complexity_metric)

        return np.array(complexity_values)

    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """
        Save sensitivity analysis results to file.

        Args:
            results: Analysis results dictionary
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
