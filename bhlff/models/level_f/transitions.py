"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Phase transitions implementation for Level F models.

This module implements the PhaseTransitions class for studying
phase transitions in multi-particle systems. It includes methods
for parameter sweeps, order parameter calculations, and critical
point identification.

Theoretical Background:
    Phase transitions in multi-particle systems are described by
    Landau theory adapted for topological systems. Order parameters
    characterize different phases, and critical points mark transitions
    between phases.
    
    The order parameters include:
    - Topological order: Σ|qᵢ| (total topological charge)
    - Phase coherence: |⟨e^{iφ}⟩| (phase coherence)
    - Spatial order: g(r_max) (spatial correlation)

Example:
    >>> transitions = PhaseTransitions(system)
    >>> phase_diagram = transitions.parameter_sweep('temperature', values)
    >>> critical_points = transitions.identify_critical_points(phase_diagram)
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ..base.abstract_model import AbstractModel


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
    
    def __init__(self, system: 'MultiParticleSystem'):
        """
        Initialize phase transitions model.
        
        Physical Meaning:
            Sets up the model for studying phase transitions
            in the multi-particle system.
            
        Args:
            system (MultiParticleSystem): Multi-particle system
        """
        super().__init__(system.domain)
        self.system = system
        self.order_parameters = {}
        self.critical_points = []
        self._setup_analysis_parameters()
    
    def parameter_sweep(self, parameter: str, 
                       values: np.ndarray) -> Dict[str, Any]:
        """
        Perform parameter sweep to study phase transitions.
        
        Physical Meaning:
            Varies a system parameter and monitors
            the system state for phase transitions.
            
        Args:
            parameter (str): Parameter to vary
            values (np.ndarray): Parameter values
            
        Returns:
            Dict containing:
                - parameter_values: parameter values
                - order_parameters: O(parameter)
                - critical_points: critical parameter values
                - phase_diagram: complete phase diagram
        """
        phase_diagram = []
        
        for param_value in values:
            # Update system parameter
            self._update_system_parameter(parameter, param_value)
            
            # Equilibrate system
            self._equilibrate_system()
            
            # Analyze system state
            state = self._analyze_system_state()
            
            # Compute order parameters
            order_params = self.compute_order_parameters()
            
            # Store results
            phase_diagram.append({
                'parameter_value': param_value,
                'state': state,
                'order_parameters': order_params
            })
        
        # Identify critical points
        critical_points = self.identify_critical_points(phase_diagram)
        
        return {
            'parameter_values': values,
            'phase_diagram': phase_diagram,
            'critical_points': critical_points
        }
    
    def compute_order_parameters(self) -> Dict[str, float]:
        """
        Compute order parameters for the system.
        
        Physical Meaning:
            Calculates order parameters that characterize
            different phases of the system.
            
        Returns:
            Dict containing:
                - topological_order: Σ|qᵢ| (total topological charge)
                - phase_coherence: |⟨e^{iφ}⟩| (phase coherence)
                - spatial_order: g(r_max) (spatial correlation)
                - energy_density: ⟨E⟩ (average energy density)
        """
        # Topological order parameter
        topological_order = self._compute_topological_order()
        
        # Phase coherence
        phase_coherence = self._compute_phase_coherence()
        
        # Spatial order
        spatial_order = self._compute_spatial_order()
        
        # Energy density
        energy_density = self._compute_energy_density()
        
        order_parameters = {
            'topological_order': topological_order,
            'phase_coherence': phase_coherence,
            'spatial_order': spatial_order,
            'energy_density': energy_density
        }
        
        self.order_parameters = order_parameters
        return order_parameters
    
    def identify_critical_points(self, 
                               phase_diagram: List[Dict[str, Any]]) -> List[Dict]:
        """
        Identify critical points in phase diagram.
        
        Physical Meaning:
            Finds critical points where phase transitions
            occur based on order parameter behavior.
            
        Args:
            phase_diagram (List[Dict]): Phase diagram data
            
        Returns:
            List of critical points with:
                - parameter_value: critical value
                - transition_type: "first_order", "second_order"
                - critical_exponents: α, β, γ, δ
        """
        critical_points = []
        
        # Extract parameter values and order parameters
        param_values = [point['parameter_value'] for point in phase_diagram]
        
        for order_param_name in ['topological_order', 'phase_coherence', 'spatial_order']:
            order_values = [point['order_parameters'][order_param_name] 
                          for point in phase_diagram]
            
            # Find discontinuities (first-order transitions)
            discontinuities = self._find_discontinuities(param_values, order_values)
            
            # Find critical points (second-order transitions)
            critical_points_param = self._find_critical_points(param_values, order_values)
            
            # Combine results
            for disc in discontinuities:
                critical_points.append({
                    'parameter_value': disc,
                    'transition_type': 'first_order',
                    'order_parameter': order_param_name,
                    'critical_exponents': self._compute_critical_exponents(
                        param_values, order_values, disc)
                })
            
            for crit in critical_points_param:
                critical_points.append({
                    'parameter_value': crit,
                    'transition_type': 'second_order',
                    'order_parameter': order_param_name,
                    'critical_exponents': self._compute_critical_exponents(
                        param_values, order_values, crit)
                })
        
        self.critical_points = critical_points
        return critical_points
    
    def analyze_phase_stability(self) -> Dict[str, Any]:
        """
        Analyze stability of different phases.
        
        Physical Meaning:
            Analyzes the stability of different phases
            in the system.
            
        Returns:
            Dict containing stability analysis
        """
        # Get current system state
        current_state = self._analyze_system_state()
        
        # Check stability of current phase
        stability = self._check_phase_stability(current_state)
        
        # Analyze phase boundaries
        boundaries = self._analyze_phase_boundaries()
        
        return {
            'current_stability': stability,
            'phase_boundaries': boundaries,
            'stability_regions': self._identify_stability_regions()
        }
    
    def _setup_analysis_parameters(self) -> None:
        """
        Setup analysis parameters for phase transitions.
        
        Physical Meaning:
            Initializes parameters needed for analysis
            of phase transitions.
        """
        self.equilibration_time = 50.0  # Time for equilibration
        self.measurement_time = 100.0  # Time for measurements
        self.discontinuity_threshold = 0.1  # Threshold for discontinuities
        self.critical_threshold = 0.05  # Threshold for critical points
        self.stability_threshold = 0.01  # Stability threshold
    
    def _update_system_parameter(self, parameter: str, value: float) -> None:
        """
        Update system parameter.
        
        Physical Meaning:
            Updates the specified parameter in the system
            to the given value.
        """
        if parameter == 'temperature':
            # Update temperature (affects thermal fluctuations)
            self.system.temperature = value
        elif parameter == 'interaction_strength':
            # Update interaction strength
            self.system.interaction_strength = value
        elif parameter == 'interaction_range':
            # Update interaction range
            self.system.interaction_range = value
        else:
            raise ValueError(f"Unknown parameter: {parameter}")
    
    def _equilibrate_system(self) -> None:
        """
        Equilibrate system to new parameter values.
        
        Physical Meaning:
            Allows the system to reach equilibrium
            under the new parameter values.
        """
        # This is a placeholder for equilibration
        # In practice, would run time evolution
        pass
    
    def _analyze_system_state(self) -> Dict[str, Any]:
        """
        Analyze current system state.
        
        Physical Meaning:
            Analyzes the current state of the system
            including particle positions, phases, and energies.
        """
        # Get particle information
        positions = np.array([p.position for p in self.system.particles])
        charges = np.array([p.charge for p in self.system.particles])
        phases = np.array([p.phase for p in self.system.particles])
        
        # Compute system properties
        total_charge = np.sum(charges)
        charge_dispersion = np.std(charges)
        phase_dispersion = np.std(phases)
        
        # Spatial properties
        if len(positions) > 1:
            distances = []
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    distances.append(dist)
            mean_distance = np.mean(distances)
            distance_dispersion = np.std(distances)
        else:
            mean_distance = 0.0
            distance_dispersion = 0.0
        
        return {
            'total_charge': total_charge,
            'charge_dispersion': charge_dispersion,
            'phase_dispersion': phase_dispersion,
            'mean_distance': mean_distance,
            'distance_dispersion': distance_dispersion,
            'n_particles': len(self.system.particles)
        }
    
    def _compute_topological_order(self) -> float:
        """
        Compute topological order parameter.
        
        Physical Meaning:
            Calculates the total topological charge
            as a measure of topological order.
        """
        charges = [p.charge for p in self.system.particles]
        return np.sum(np.abs(charges))
    
    def _compute_phase_coherence(self) -> float:
        """
        Compute phase coherence order parameter.
        
        Physical Meaning:
            Calculates the phase coherence |⟨e^{iφ}⟩|
            as a measure of phase ordering.
        """
        phases = [p.phase for p in self.system.particles]
        complex_phases = np.exp(1j * np.array(phases))
        coherence = np.abs(np.mean(complex_phases))
        return coherence
    
    def _compute_spatial_order(self) -> float:
        """
        Compute spatial order parameter.
        
        Physical Meaning:
            Calculates spatial correlation as a measure
            of spatial ordering.
        """
        # Get spatial correlations
        correlations = self.system.analyze_correlations()
        spatial_corr = correlations['spatial_correlations']
        
        # Use correlation length as spatial order
        return spatial_corr['correlation_length']
    
    def _compute_energy_density(self) -> float:
        """
        Compute average energy density.
        
        Physical Meaning:
            Calculates the average energy density
            of the system.
        """
        # Compute effective potential
        potential = self.system.compute_effective_potential()
        
        # Average energy density
        energy_density = np.mean(potential)
        return energy_density
    
    def _find_discontinuities(self, param_values: List[float],
                            order_values: List[float]) -> List[float]:
        """
        Find discontinuities in order parameter.
        
        Physical Meaning:
            Identifies first-order phase transitions
            by finding discontinuities in order parameters.
        """
        discontinuities = []
        
        for i in range(1, len(order_values)):
            # Check for large jumps
            jump = abs(order_values[i] - order_values[i-1])
            if jump > self.discontinuity_threshold:
                discontinuities.append(param_values[i])
        
        return discontinuities
    
    def _find_critical_points(self, param_values: List[float],
                            order_values: List[float]) -> List[float]:
        """
        Find critical points in order parameter.
        
        Physical Meaning:
            Identifies second-order phase transitions
            by finding critical points in order parameters.
        """
        critical_points = []
        
        # Compute derivatives
        derivatives = []
        for i in range(1, len(order_values)):
            deriv = (order_values[i] - order_values[i-1]) / (param_values[i] - param_values[i-1])
            derivatives.append(deriv)
        
        # Find points where derivative changes sign
        for i in range(1, len(derivatives)):
            if (derivatives[i] * derivatives[i-1] < 0 and 
                abs(derivatives[i] - derivatives[i-1]) > self.critical_threshold):
                critical_points.append(param_values[i])
        
        return critical_points
    
    def _compute_critical_exponents(self, param_values: List[float],
                                  order_values: List[float],
                                  critical_point: float) -> Dict[str, float]:
        """
        Compute critical exponents for phase transition.
        
        Physical Meaning:
            Calculates critical exponents α, β, γ, δ
            for the phase transition.
        """
        # Find index of critical point
        crit_idx = np.argmin(np.abs(np.array(param_values) - critical_point))
        
        # Fit power law near critical point
        # This is simplified - in practice would use proper fitting
        alpha = 0.1  # Heat capacity exponent
        beta = 0.3   # Order parameter exponent
        gamma = 1.2  # Susceptibility exponent
        delta = 4.0  # Critical isotherm exponent
        
        return {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'delta': delta
        }
    
    def _check_phase_stability(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check stability of current phase.
        
        Physical Meaning:
            Analyzes the stability of the current
            phase of the system.
        """
        # Check system stability
        stability = self.system.check_stability()
        
        # Additional phase stability checks
        is_stable = stability['is_stable']
        stability_margin = stability['stability_margin']
        
        return {
            'is_stable': is_stable,
            'stability_margin': stability_margin,
            'phase_stability': is_stable and stability_margin > self.stability_threshold
        }
    
    def _analyze_phase_boundaries(self) -> Dict[str, Any]:
        """
        Analyze phase boundaries.
        
        Physical Meaning:
            Identifies boundaries between different
            phases in parameter space.
        """
        # This would analyze the phase diagram
        # to identify phase boundaries
        return {
            'boundaries': [],
            'metastable_regions': [],
            'coexistence_regions': []
        }
    
    def _identify_stability_regions(self) -> Dict[str, Any]:
        """
        Identify stability regions in parameter space.
        
        Physical Meaning:
            Identifies regions of parameter space
            where different phases are stable.
        """
        return {
            'stable_regions': [],
            'unstable_regions': [],
            'metastable_regions': []
        }
    
    def analyze(self, data: Any) -> Dict[str, Any]:
        """
        Analyze data for this model.
        
        Physical Meaning:
            Performs comprehensive analysis of phase transitions,
            including order parameters and critical points.
            
        Args:
            data (Any): Input data to analyze (not used for this model)
            
        Returns:
            Dict: Analysis results including order parameters and stability
        """
        # Compute order parameters
        order_params = self.compute_order_parameters()
        
        # Analyze phase stability
        stability = self.analyze_phase_stability()
        
        return {
            'order_parameters': order_params,
            'stability': stability,
            'critical_points': self.critical_points
        }

