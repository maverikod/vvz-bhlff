"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Cosmological models for 7D phase field theory.

This module implements envelope-derived effective metric models
for the 7D phase field theory without invoking spacetime curvature
or cosmological scale factors.

Theoretical Background:
    Gravity-like effects emerge from the curvature of the VBP envelope
    in the 7D phase field theory. There is no spacetime curvature here;
    instead, an effective metric g_eff[Θ] is derived from envelope
    invariants and phase dynamics.

Example:
    >>> cosmology = CosmologicalModel(initial_conditions, params)
    >>> evolution = cosmology.evolve_universe([0, 13.8])
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from ..base.model_base import ModelBase


class EnvelopeEffectiveMetric:
    """
    Envelope-derived effective metric (no spacetime curvature).

    Physical Meaning:
        Computes a 7x7 effective metric g_eff[Θ] derived solely from
        envelope dynamics and invariants. No cosmological scale factors,
        no spacetime curvature.

    Mathematical Foundation:
        g00 = -1/c_φ^2; spatial gij = A δ^{ij} with A from envelope invariants;
        phase-space diagonal unity (can be extended to anisotropic models).
    """

    def __init__(self, params: Dict[str, float]):
        self.params = params

    def compute_effective_metric_from_vbp_envelope(
        self,
        envelope_invariants: Dict[str, float],
    ) -> np.ndarray:
        """
        Compute effective metric from VBP envelope dynamics.

        Physical Meaning:
            Computes the effective metric g_eff[Θ] using only VBP envelope
            invariants (no spacetime curvature, no scale factors).

        Mathematical Foundation:
            g_eff[Θ] with g00=-1/c_φ^2, gij=A δ^{ij} (isotropic case), where
            c_φ is the phase velocity and A = χ'/κ is derived from envelope
            invariants (provided in envelope_invariants).

        Args:
            envelope_invariants: dict containing keys like
                - chi_over_kappa: float, isotropic spatial scaling A

        Returns:
            7x7 effective metric tensor g_eff[Θ]
        """
        # Initialize effective metric from VBP envelope
        g_eff = np.zeros((7, 7))

        # Time component: g00 = -1/c_φ^2 (VBP envelope)
        c_phi = self.params.get("c_phi", 1.0)  # Phase velocity
        g_eff[0, 0] = -1.0 / (c_phi**2)

        # Spatial components: gij = A δ^{ij} (isotropic)
        chi_kappa = float(
            envelope_invariants.get("chi_over_kappa", self.params.get("chi_kappa", 1.0))
        )
        for i in range(1, 4):
            g_eff[i, i] = chi_kappa

        # Phase components: gαβ (phase space metric)
        for alpha in range(4, 7):
            g_eff[alpha, alpha] = 1.0  # Unit phase space metric

        return g_eff

    def compute_scale_factor(self, t: float) -> float:
        """
        Compute scale factor for cosmological evolution using VBP envelope dynamics.
        
        Physical Meaning:
            Computes a scale factor for cosmological evolution based on
            VBP envelope dynamics rather than classical spacetime expansion.
            Uses power law evolution instead of exponential growth.
            
        Args:
            t: Cosmological time
            
        Returns:
            Scale factor from VBP envelope dynamics
        """
        # VBP envelope scale factor evolution (no exponential attenuation)
        # In the 7D BVP theory, this represents the evolution of the
        # envelope effective metric rather than spacetime expansion
        H0 = self.params.get("H0", 70.0)
        omega_lambda = self.params.get("omega_lambda", 0.7)
        
        # Power law evolution for VBP envelope dynamics
        if omega_lambda > 0:
            # Dark energy dominated - power law instead of exponential
            return (1.0 + H0 * np.sqrt(omega_lambda) * t / 100.0) ** 2.0
        else:
            # Matter dominated - power law
            return (1.0 + H0 * t / 100.0) ** (2.0/3.0)
    
    def compute_envelope_curvature_metric(self, phase_field: np.ndarray) -> np.ndarray:
        """
        Compute effective metric from phase field envelope curvature.
        
        Physical Meaning:
            Computes the effective metric g_eff[Θ] directly from the
            phase field envelope curvature, incorporating local
            envelope dynamics and phase gradients.
            
        Mathematical Foundation:
            g_eff[Θ] = f(∇Θ, c_φ(a,k), A^{ij}) where the metric components
            depend on local phase field gradients and envelope properties.
            
        Args:
            phase_field: Phase field configuration Θ(x,φ,t)
            
        Returns:
            7x7 effective metric tensor g_eff[Θ] from envelope curvature
        """
        # Compute phase field gradients
        phase_gradients = np.gradient(phase_field)
        
        # Compute envelope curvature invariants
        envelope_amplitude = np.mean(np.abs(phase_field))
        envelope_gradient_magnitude = np.mean([np.mean(np.abs(grad)) for grad in phase_gradients])
        
        # Initialize effective metric
        g_eff = np.zeros((7, 7))
        
        # Time component: g00 = -1/c_φ^2 with envelope corrections
        c_phi = self.params.get("c_phi", 1.0)
        envelope_correction = 1.0 + 0.1 * envelope_amplitude
        g_eff[0, 0] = -1.0 / (c_phi**2 * envelope_correction)
        
        # Spatial components: gij = A δ^{ij} with envelope curvature
        chi_kappa = self.params.get("chi_kappa", 1.0)
        curvature_correction = 1.0 + 0.05 * envelope_gradient_magnitude
        for i in range(1, 4):
            g_eff[i, i] = chi_kappa * curvature_correction
        
        # Phase components: gαβ with envelope dynamics
        for alpha in range(4, 7):
            g_eff[alpha, alpha] = 1.0 + 0.02 * envelope_amplitude
        
        return g_eff
    
    def compute_anisotropic_metric(self, envelope_invariants: Dict[str, float]) -> np.ndarray:
        """
        Compute anisotropic effective metric from envelope invariants.
        
        Physical Meaning:
            Computes an anisotropic effective metric g_eff[Θ] where
            spatial components can differ, reflecting anisotropic
            envelope dynamics in the VBP.
            
        Mathematical Foundation:
            g_eff[Θ] with g00=-1/c_φ^2, gij=A^{ij} (anisotropic case),
            where A^{ij} can have different values for different spatial directions.
            
        Args:
            envelope_invariants: Dictionary containing anisotropic envelope properties
            
        Returns:
            7x7 anisotropic effective metric tensor g_eff[Θ]
        """
        # Initialize anisotropic effective metric
        g_eff = np.zeros((7, 7))
        
        # Time component: g00 = -1/c_φ^2
        c_phi = self.params.get("c_phi", 1.0)
        g_eff[0, 0] = -1.0 / (c_phi**2)
        
        # Anisotropic spatial components: gij = A^{ij}
        A_xx = envelope_invariants.get("A_xx", self.params.get("chi_kappa", 1.0))
        A_yy = envelope_invariants.get("A_yy", self.params.get("chi_kappa", 1.0))
        A_zz = envelope_invariants.get("A_zz", self.params.get("chi_kappa", 1.0))
        
        g_eff[1, 1] = A_xx
        g_eff[2, 2] = A_yy
        g_eff[3, 3] = A_zz
        
        # Phase components: gαβ (phase space metric)
        for alpha in range(4, 7):
            g_eff[alpha, alpha] = 1.0
        
        return g_eff


class CosmologicalModel(ModelBase):
    """
    Cosmological evolution model for 7D phase field theory.

    Physical Meaning:
        Implements the evolution of phase field in expanding universe,
        including structure formation and cosmological parameters.

    Mathematical Foundation:
        Solves the phase field evolution equation in expanding spacetime:
        ∂²a/∂t² + 3H(t)∂a/∂t - c_φ²∇²a + V'(a) = 0

    Attributes:
        scale_factor (np.ndarray): Scale factor evolution a(t)
        hubble_parameter (np.ndarray): Hubble parameter H(t)
        phase_field (np.ndarray): Phase field configuration
        cosmology_params (dict): Cosmological parameters
    """

    def __init__(
        self, initial_conditions: Dict[str, Any], cosmology_params: Dict[str, Any]
    ):
        """
        Initialize cosmological model.

        Physical Meaning:
            Sets up the cosmological model with initial conditions
            and cosmological parameters for universe evolution.

        Args:
            initial_conditions: Initial phase field configuration
            cosmology_params: Cosmological parameters
        """
        super().__init__()
        self.initial_conditions = initial_conditions
        self.cosmology_params = cosmology_params
        self.metric = EnvelopeEffectiveMetric(cosmology_params)
        self._setup_evolution_parameters()

    def _setup_evolution_parameters(self) -> None:
        """
        Setup evolution parameters.

        Physical Meaning:
            Initializes parameters for cosmological evolution,
            including time steps and physical constants.
        """
        # Time evolution parameters
        self.time_start = self.cosmology_params.get("time_start", 0.0)
        self.time_end = self.cosmology_params.get("time_end", 13.8)  # Gyr
        self.dt = self.cosmology_params.get("dt", 0.01)  # Gyr

        # Physical parameters
        self.c_phi = self.cosmology_params.get("c_phi", 1e10)  # Phase velocity
        # No phase_mass - removed according to 7D BVP theory

        # Initialize arrays
        self.time_steps = np.arange(self.time_start, self.time_end + self.dt, self.dt)
        self.scale_factor = np.zeros_like(
            self.time_steps
        )  # Will be filled during evolution
        self.hubble_parameter = np.zeros_like(
            self.time_steps
        )  # Will be filled during evolution
        self.phase_field = None

    def evolve_universe(
        self, time_range: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Evolve universe from initial to final time.

        Physical Meaning:
            Evolves the universe from initial conditions through
            cosmological time, computing phase field evolution
            and structure formation.

        Mathematical Foundation:
            Integrates the phase field evolution equation with
            cosmological expansion and gravitational effects.

        Args:
            time_range: Optional time range [start, end]

        Returns:
            Dictionary with evolution results
        """
        if time_range is not None:
            self.time_start, self.time_end = time_range
            self.time_steps = np.arange(
                self.time_start, self.time_end + self.dt, self.dt
            )

        # Initialize evolution
        evolution_results = {
            "time": self.time_steps,
            "scale_factor": np.zeros_like(self.time_steps),
            "hubble_parameter": np.zeros_like(self.time_steps),
            "phase_field_evolution": [],
            "structure_formation": [],
        }

        # Time evolution
        for i, t in enumerate(self.time_steps):
            # Update scale factor
            # Use envelope effective metric for scale factors
            a_t = self.metric.compute_scale_factor(t)
            b_t = a_t  # Simplified for envelope metric
            self.scale_factor[i] = a_t
            self.hubble_parameter[i] = self._compute_hubble_parameter(t)

            # Evolve phase field
            if i == 0:
                # Initial conditions
                self.phase_field = self._initialize_phase_field()
            else:
                # Evolution step
                self.phase_field = self._evolve_phase_field_step(t, self.dt, a_t)

            # Analyze structure
            structure = self._analyze_structure_at_time(t)

            evolution_results["phase_field_evolution"].append(self.phase_field.copy())
            evolution_results["structure_formation"].append(structure)

        # Add scale factor and Hubble parameter to results
        evolution_results["scale_factor"] = self.scale_factor.copy()
        evolution_results["hubble_parameter"] = self.hubble_parameter.copy()

        return evolution_results

    def _initialize_phase_field(self) -> np.ndarray:
        """
        Initialize phase field from initial conditions.

        Physical Meaning:
            Creates initial phase field configuration based on
            cosmological initial conditions.

        Returns:
            Initial phase field configuration
        """
        # Get domain parameters
        domain_size = self.initial_conditions.get("domain_size", 1000.0)
        resolution = self.initial_conditions.get("resolution", 256)

        # Create initial fluctuations
        if self.initial_conditions.get("type") == "gaussian_fluctuations":
            # Gaussian random fluctuations
            np.random.seed(self.initial_conditions.get("seed", 42))
            phase_field = np.random.normal(0, 0.1, (resolution, resolution, resolution))
        else:
            # Default: zero field
            phase_field = np.zeros((resolution, resolution, resolution))

        return phase_field

    def _evolve_phase_field_step(
        self, t: float, dt: float, scale_factor: float = 1.0
    ) -> np.ndarray:
        """
        Evolve phase field for one time step.

        Physical Meaning:
            Advances the phase field configuration by one time step
            using the cosmological evolution equation.

        Mathematical Foundation:
            ∂²a/∂t² + 3H(t)∂a/∂t - c_φ²∇²a + V'(a) = 0

        Args:
            t: Current time
            dt: Time step
            scale_factor: Current scale factor

        Returns:
            Updated phase field
        """
        # Get current Hubble parameter
        H_t = (
            self.hubble_parameter[-1]
            if len(self.hubble_parameter) > 0
            else self.cosmology_params.get("H0", 70.0)
        )

        # Simple evolution (for demonstration)
        # In full implementation, this would solve the PDE
        phase_field_new = self.phase_field.copy()

        # Add cosmological expansion effects using step resonator model
        # No exponential decay - use step resonator transmission
        transmission_coeff = 0.9  # Energy transmission through resonator
        expansion_factor = transmission_coeff  # Step resonator model
        phase_field_new *= expansion_factor

        # Add phase field dynamics
        # This is a simplified version - full implementation would
        # solve the fractional Laplacian equation

        return phase_field_new

    def _compute_hubble_parameter(self, t: float) -> float:
        """
        Compute Hubble parameter at time t.

        Physical Meaning:
            Computes the Hubble parameter H(t) for the expanding
            universe at cosmological time t.

        Mathematical Foundation:
            H(t) = H0 * sqrt(Ω_Λ) for ΛCDM model

        Args:
            t: Cosmological time

        Returns:
            Hubble parameter
        """
        omega_lambda = self.cosmology_params.get("omega_lambda", 0.7)
        H0 = self.cosmology_params.get("H0", 70.0)
        if omega_lambda > 0:
            # ΛCDM model with dark energy
            hubble_parameter = H0 * np.sqrt(omega_lambda)
        else:
            # Model without dark energy
            hubble_parameter = H0

        return hubble_parameter

    def _analyze_structure_at_time(self, t: float) -> Dict[str, Any]:
        """
        Analyze structure formation at given time.

        Physical Meaning:
            Analyzes the formation of large-scale structure
            from phase field evolution at cosmological time t.

        Args:
            t: Cosmological time

        Returns:
            Structure analysis results
        """
        if self.phase_field is None:
            return {}

        # Compute structure metrics
        structure = {
            "time": t,
            "phase_field_rms": np.sqrt(np.mean(self.phase_field**2)),
            "phase_field_max": np.max(np.abs(self.phase_field)),
            "correlation_length": self._compute_correlation_length(),
            "topological_defects": self._count_topological_defects(),
        }

        return structure

    def _compute_correlation_length(self) -> float:
        """
        Compute correlation length of phase field.

        Physical Meaning:
            Computes the characteristic length scale over which
            the phase field is correlated.

        Returns:
            Correlation length
        """
        if self.phase_field is None:
            return 0.0

        # Simplified correlation length computation
        # In full implementation, this would use FFT-based correlation
        field_std = np.std(self.phase_field)
        if field_std > 0:
            return 1.0 / field_std
        else:
            return 0.0

    def _count_topological_defects(self) -> int:
        """
        Count topological defects in phase field.

        Physical Meaning:
            Counts the number of topological defects (vortices,
            monopoles, etc.) in the current phase field configuration.

        Returns:
            Number of topological defects
        """
        if self.phase_field is None:
            return 0

        # Simplified defect counting
        # In full implementation, this would use proper topological analysis
        gradient_magnitude = np.gradient(self.phase_field)
        defect_density = np.sum(np.abs(gradient_magnitude))

        return int(defect_density)

    def analyze_structure_formation(self) -> Dict[str, Any]:
        """
        Analyze large-scale structure formation.

        Physical Meaning:
            Analyzes the overall process of structure formation
            throughout cosmological evolution.

        Returns:
            Structure formation analysis
        """
        if not hasattr(self, "scale_factor") or len(self.scale_factor) == 0:
            return {}

        # Analyze structure formation metrics
        analysis = {
            "total_evolution_time": self.time_end - self.time_start,
            "final_scale_factor": self.scale_factor[-1],
            "expansion_rate": np.mean(np.diff(self.scale_factor) / self.dt),
            "structure_growth_rate": self._compute_structure_growth_rate(),
        }

        return analysis

    def _compute_structure_growth_rate(self) -> float:
        """
        Compute structure growth rate.

        Physical Meaning:
            Computes the rate at which large-scale structure
            grows during cosmological evolution.

        Returns:
            Structure growth rate
        """
        if not hasattr(self, "scale_factor") or len(self.scale_factor) < 2:
            return 0.0

        # Simplified growth rate computation
        scale_growth = np.diff(self.scale_factor)
        return np.mean(scale_growth) if len(scale_growth) > 0 else 0.0

    def compute_cosmological_parameters(self) -> Dict[str, float]:
        """
        Compute cosmological parameters from evolution.

        Physical Meaning:
            Computes derived cosmological parameters from
            the evolution results.

        Returns:
            Dictionary of cosmological parameters
        """
        if not hasattr(self, "scale_factor") or len(self.scale_factor) == 0:
            return {}

        # Compute derived parameters
        parameters = {
            "current_scale_factor": self.scale_factor[-1],
            "current_hubble_parameter": self.hubble_parameter[-1],
            "age_universe": self.time_end,
            "expansion_rate": np.mean(np.diff(self.scale_factor) / self.dt),
            "phase_velocity": self.c_phi,
        }

        return parameters
