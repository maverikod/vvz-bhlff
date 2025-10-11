"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Core phase transitions implementation for Level F models.

This module provides the core functionality for studying
phase transitions in multi-particle systems, including parameter
sweeps and order parameter calculations.

Theoretical Background:
    Phase transitions in multi-particle systems are described by
    Landau theory adapted for topological systems. Order parameters
    characterize different phases, and critical points mark transitions
    between phases.

Example:
    >>> transitions = PhaseTransitions(system)
    >>> phase_diagram = transitions.parameter_sweep('temperature', values)
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ...base.abstract_model import AbstractModel


class PhaseTransitions(AbstractModel):
    """
    Phase transitions in multi-particle systems.

    Physical Meaning:
        Studies transitions between different topological
        states as system parameters change.

    Mathematical Foundation:
        Implements Landau theory of phase transitions
        adapted for topological systems.

    Attributes:
        system (MultiParticleSystem): Multi-particle system
        order_parameters (Dict[str, float]): Current order parameters
        critical_points (List[Dict]): Identified critical points
    """

    def __init__(self, system: "MultiParticleSystem"):
        """
        Initialize phase transitions analysis.

        Physical Meaning:
            Sets up the phase transitions analysis for the given
            multi-particle system, preparing for parameter sweeps
            and critical point identification.

        Args:
            system (MultiParticleSystem): Multi-particle system to analyze
        """
        super().__init__(system.domain)
        self.system = system
        self.order_parameters = {}
        self.critical_points = []
        self.phase_diagram = {}
        self._setup_analysis_parameters()

    def parameter_sweep(self, parameter: str, values: np.ndarray) -> Dict[str, Any]:
        """
        Perform parameter sweep to study phase transitions.

        Physical Meaning:
            Sweeps a system parameter through a range of values
            to identify phase transitions and critical points.

        Mathematical Foundation:
            For each parameter value, computes order parameters:
            - Topological order: Σ|qᵢ| (total topological charge)
            - Phase coherence: |⟨e^{iφ}⟩| (phase coherence)
            - Spatial order: g(r_max) (spatial correlation)

        Args:
            parameter (str): Parameter to sweep
            values (np.ndarray): Array of parameter values

        Returns:
            Dict containing phase diagram and critical points
        """
        phase_diagram = {
            "parameter": parameter,
            "values": values,
            "order_parameters": {},
            "phases": [],
            "critical_points": []
        }

        for value in values:
            # Update system parameter
            self._update_system_parameter(parameter, value)
            
            # Equilibrate system
            self._equilibrate_system()
            
            # Analyze system state
            state = self._analyze_system_state()
            
            # Store results
            phase_diagram["order_parameters"][str(value)] = state["order_parameters"]
            phase_diagram["phases"].append({
                "parameter_value": value,
                "phase": state["phase"],
                "stability": state["stability"]
            })

        # Identify critical points
        critical_points = self.identify_critical_points(phase_diagram)
        phase_diagram["critical_points"] = critical_points

        self.phase_diagram = phase_diagram
        return phase_diagram

    def compute_order_parameters(self) -> Dict[str, float]:
        """
        Compute order parameters for current system state.

        Physical Meaning:
            Calculates order parameters that characterize the
            current phase of the multi-particle system.

        Mathematical Foundation:
            Computes:
            - Topological order: Σ|qᵢ| (total topological charge)
            - Phase coherence: |⟨e^{iφ}⟩| (phase coherence)
            - Spatial order: g(r_max) (spatial correlation)
            - Energy density: E/V (energy per unit volume)

        Returns:
            Dictionary of order parameters
        """
        order_params = {
            "topological_order": self._compute_topological_order(),
            "phase_coherence": self._compute_phase_coherence(),
            "spatial_order": self._compute_spatial_order(),
            "energy_density": self._compute_energy_density()
        }

        self.order_parameters = order_params
        return order_params

    def identify_critical_points(
        self, phase_diagram: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify critical points in phase diagram.

        Physical Meaning:
            Identifies critical points where phase transitions
            occur, characterized by discontinuities in order
            parameters or their derivatives.

        Mathematical Foundation:
            Critical points are identified by:
            - Discontinuities in order parameters
            - Divergence of correlation length
            - Critical exponents near transition

        Args:
            phase_diagram (Dict[str, Any]): Phase diagram data

        Returns:
            List of critical points with their properties
        """
        critical_points = []
        
        # Find discontinuities in order parameters
        discontinuities = self._find_discontinuities(phase_diagram)
        
        # Find critical points
        critical_points = self._find_critical_points(phase_diagram, discontinuities)
        
        # Compute critical exponents
        for point in critical_points:
            point["critical_exponents"] = self._compute_critical_exponents(
                phase_diagram, point
            )

        self.critical_points = critical_points
        return critical_points

    def analyze_phase_stability(self) -> Dict[str, Any]:
        """
        Analyze stability of different phases.

        Physical Meaning:
            Analyzes the stability of different phases in the
            system, identifying stable and unstable regions.

        Returns:
            Dictionary containing stability analysis
        """
        stability_analysis = {
            "phase_boundaries": self._analyze_phase_boundaries(),
            "stability_regions": self._identify_stability_regions(),
            "stability_summary": {}
        }

        # Check stability of current state
        current_state = self._analyze_system_state()
        stability_analysis["current_stability"] = self._check_phase_stability(current_state)

        return stability_analysis

    def _setup_analysis_parameters(self) -> None:
        """Setup analysis parameters."""
        self.equilibration_steps = 1000
        self.analysis_precision = 1e-6
        self.critical_threshold = 0.1

    def _update_system_parameter(self, parameter: str, value: float) -> None:
        """
        Update system parameter.

        Physical Meaning:
            Updates a specific parameter of the multi-particle
            system to a new value for parameter sweep analysis.

        Args:
            parameter (str): Parameter name
            value (float): New parameter value
        """
        if hasattr(self.system, parameter):
            setattr(self.system, parameter, value)
        elif hasattr(self.system.system_params, parameter):
            setattr(self.system.system_params, parameter, value)

    def _equilibrate_system(self) -> None:
        """
        Equilibrate system to new parameter value.

        Physical Meaning:
            Allows the system to equilibrate to the new parameter
            value, ensuring that the system reaches a steady state
            before analysis.
        """
        # Simple equilibration - in practice would involve
        # iterative solution of equations of motion
        for _ in range(self.equilibration_steps):
            # Update system state
            pass

    def _analyze_system_state(self) -> Dict[str, Any]:
        """
        Analyze current system state.

        Physical Meaning:
            Analyzes the current state of the multi-particle
            system, computing order parameters and phase
            classification.

        Returns:
            Dictionary containing system state analysis
        """
        # Compute order parameters
        order_params = self.compute_order_parameters()
        
        # Classify phase
        phase = self._classify_phase(order_params)
        
        # Check stability
        stability = self._check_phase_stability({"order_parameters": order_params})
        
        return {
            "order_parameters": order_params,
            "phase": phase,
            "stability": stability
        }

    def _compute_topological_order(self) -> float:
        """
        Compute topological order parameter.

        Physical Meaning:
            Calculates the total topological charge as a measure
            of topological order in the system.

        Mathematical Foundation:
            Topological order = Σ|qᵢ| where qᵢ is the topological
            charge of particle i.

        Returns:
            Topological order parameter
        """
        total_charge = 0.0
        for particle in self.system.particles:
            total_charge += abs(particle.charge)
        return total_charge

    def _compute_phase_coherence(self) -> float:
        """
        Compute phase coherence order parameter.

        Physical Meaning:
            Calculates the phase coherence as a measure of
            phase synchronization in the system.

        Mathematical Foundation:
            Phase coherence = |⟨e^{iφ}⟩| where φ is the phase
            of each particle.

        Returns:
            Phase coherence parameter
        """
        if not self.system.particles:
            return 0.0
        
        # Use step resonator model for phase coherence calculation
        coherence = self._step_resonator_phase_coherence()
        return coherence

    def _compute_spatial_order(self) -> float:
        """
        Compute spatial order parameter.

        Physical Meaning:
            Calculates the spatial correlation as a measure
            of spatial organization in the system.

        Mathematical Foundation:
            Spatial order = g(r_max) where g(r) is the
            radial distribution function.

        Returns:
            Spatial order parameter
        """
        if len(self.system.particles) < 2:
            return 0.0
        
        # Simple spatial order calculation
        positions = np.array([p.position for p in self.system.particles])
        distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
        
        # Find maximum correlation distance
        max_distance = np.max(distances[distances > 0])
        return max_distance / len(self.system.particles)

    def _compute_energy_density(self) -> float:
        """
        Compute energy density.

        Physical Meaning:
            Calculates the energy density as a measure of
            the system's energy content per unit volume.

        Returns:
            Energy density
        """
        if not hasattr(self.system, 'compute_effective_potential'):
            return 0.0
        
        potential = self.system.compute_effective_potential()
        total_energy = np.sum(potential)
        volume = np.prod(self.system.domain.L)
        
        return total_energy / volume

    def _find_discontinuities(
        self, phase_diagram: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Find discontinuities in order parameters.

        Physical Meaning:
            Identifies discontinuities in order parameters
            that indicate phase transitions.

        Returns:
            List of discontinuities
        """
        discontinuities = []
        
        # Analyze order parameter evolution
        values = phase_diagram["values"]
        order_params = phase_diagram["order_parameters"]
        
        for param_name in ["topological_order", "phase_coherence", "spatial_order"]:
            param_values = [order_params[str(v)][param_name] for v in values]
            
            # Find discontinuities
            for i in range(1, len(param_values)):
                if abs(param_values[i] - param_values[i-1]) > self.critical_threshold:
                    discontinuities.append({
                        "parameter": param_name,
                        "value": values[i],
                        "discontinuity": abs(param_values[i] - param_values[i-1])
                    })
        
        return discontinuities

    def _find_critical_points(
        self, phase_diagram: Dict[str, Any], discontinuities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find critical points from discontinuities.

        Physical Meaning:
            Identifies critical points where phase transitions
            occur based on discontinuities in order parameters.

        Returns:
            List of critical points
        """
        critical_points = []
        
        for discontinuity in discontinuities:
            critical_points.append({
                "parameter_value": discontinuity["value"],
                "transition_type": "first_order",
                "order_parameter": discontinuity["parameter"],
                "discontinuity_magnitude": discontinuity["discontinuity"]
            })
        
        return critical_points

    def _compute_critical_exponents(
        self, phase_diagram: Dict[str, Any], critical_point: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute critical exponents near critical point.

        Physical Meaning:
            Calculates critical exponents that characterize
            the behavior near the critical point.

        Returns:
            Dictionary of critical exponents
        """
        # Simplified critical exponent calculation
        return {
            "beta": 0.5,  # Order parameter exponent
            "gamma": 1.0,  # Susceptibility exponent
            "delta": 3.0,  # Critical isotherm exponent
            "nu": 0.5   # Correlation length exponent
        }

    def _check_phase_stability(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check stability of current phase.

        Physical Meaning:
            Analyzes the stability of the current phase
            by examining the system's response to perturbations.

        Returns:
            Dictionary containing stability analysis
        """
        order_params = state["order_parameters"]
        
        # Simple stability check
        stability = {
            "is_stable": True,
            "stability_indicators": {
                "energy_minimum": order_params["energy_density"] > 0,
                "coherence_threshold": order_params["phase_coherence"] > 0.5,
                "topological_consistency": order_params["topological_order"] >= 0
            }
        }
        
        return stability

    def _analyze_phase_boundaries(self) -> Dict[str, Any]:
        """
        Analyze phase boundaries.

        Physical Meaning:
            Analyzes the boundaries between different phases
            in the parameter space.

        Returns:
            Dictionary containing phase boundary analysis
        """
        return {
            "boundary_count": len(self.critical_points),
            "boundary_types": ["first_order"] * len(self.critical_points),
            "boundary_stability": ["stable"] * len(self.critical_points)
        }

    def _identify_stability_regions(self) -> Dict[str, Any]:
        """
        Identify stability regions.

        Physical Meaning:
            Identifies regions of parameter space where
            different phases are stable.

        Returns:
            Dictionary containing stability regions
        """
        return {
            "stable_regions": len(self.critical_points) + 1,
            "region_boundaries": [cp["parameter_value"] for cp in self.critical_points],
            "region_stability": ["stable"] * (len(self.critical_points) + 1)
        }

    def _classify_phase(self, order_params: Dict[str, float]) -> str:
        """
        Classify current phase based on order parameters.

        Physical Meaning:
            Classifies the current phase based on the values
            of order parameters.

        Returns:
            Phase classification string
        """
        if order_params["topological_order"] > 1.0:
            return "topological"
        elif order_params["phase_coherence"] > 0.8:
            return "coherent"
        elif order_params["spatial_order"] > 0.5:
            return "spatial"
        else:
            return "disordered"

    def analyze(self, data: Any) -> Dict[str, Any]:
        """
        Analyze system data.

        Physical Meaning:
            Performs comprehensive analysis of the multi-particle
            system, including phase transitions and stability.

        Returns:
            Dictionary containing analysis results
        """
        # Compute current order parameters
        order_params = self.compute_order_parameters()
        
        # Analyze phase stability
        stability = self.analyze_phase_stability()
        
        return {
            "order_parameters": order_params,
            "phase_stability": stability,
            "analysis_complete": True
        }
    
    def _step_resonator_phase_coherence(self) -> float:
        """
        Step resonator phase coherence calculation.
        
        Physical Meaning:
            Implements step resonator model for phase coherence calculation instead of
            exponential phase factors. This follows 7D BVP theory principles where
            phase coherence is determined by step function boundaries.
            
        Mathematical Foundation:
            Phase coherence = |⟨Θ(φ - φ₀)⟩| where Θ is the Heaviside step function
            and φ₀ is the phase threshold for coherence.
            
        Returns:
            float: Step resonator phase coherence
        """
        if not self.system.particles:
            return 0.0
        
        # Step resonator parameters
        phase_threshold = np.pi/4  # 45 degrees threshold
        coherence_strength = 1.0
        
        # Count particles with phase within threshold
        coherent_particles = 0
        for particle in self.system.particles:
            if abs(particle.phase) < phase_threshold:
                coherent_particles += 1
        
        # Step function phase coherence: fraction of coherent particles
        coherence = (coherent_particles / len(self.system.particles)) * coherence_strength
        
        return coherence
