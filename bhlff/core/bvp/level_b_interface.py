"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP level B interface implementation.

This module provides integration interface for level B of the 7D phase field theory,
ensuring that BVP serves as the central backbone for fundamental properties analysis.

Physical Meaning:
    Level B: Fundamental field properties including power law tails, nodes, and topological charge

Mathematical Foundation:
    Implements specific mathematical operations that work with BVP envelope data,
    transforming it according to level B requirements while maintaining BVP framework compliance.

Example:
    >>> level_b = LevelBInterface(bvp_core)
    >>> result = level_b.process_bvp_data(envelope)
"""

import numpy as np
from typing import Dict, Any
from scipy.ndimage import minimum_filter, maximum_filter

from .bvp_level_interface_base import BVPLevelInterface
from .bvp_core import BVPCore


class LevelBInterface(BVPLevelInterface):
    """
    BVP integration interface for Level B (fundamental properties).
    
    Physical Meaning:
        Provides BVP data for Level B analysis of fundamental field
        properties including power law tails, nodes, and topological charge.
    """
    
    def __init__(self, bvp_core: BVPCore):
        self.bvp_core = bvp_core
        self.constants = bvp_core.constants
    
    def process_bvp_data(self, envelope: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Process BVP data for Level B operations.
        
        Physical Meaning:
            Analyzes fundamental properties of BVP envelope including
            power law tails, absence of spherical nodes, and topological charge.
        """
        # Analyze power law tails
        tail_data = self._analyze_power_law_tails(envelope)
        
        # Check for spherical nodes
        nodes_data = self._check_spherical_nodes(envelope)
        
        # Compute topological charge
        charge_data = self._compute_topological_charge(envelope)
        
        # Analyze zone separation
        zones_data = self._analyze_zone_separation(envelope)
        
        return {
            "envelope": envelope,
            "power_law_tails": tail_data,
            "spherical_nodes": nodes_data,
            "topological_charge": charge_data,
            "zone_separation": zones_data,
            "level": "B"
        }
    
    def _analyze_power_law_tails(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Analyze power law tails in homogeneous medium.
        
        Physical Meaning:
            Computes the power law decay of BVP envelope amplitude
            in the tail region, which characterizes the field's
            long-range behavior in homogeneous medium.
        """
        amplitude = np.abs(envelope)
        
        # Compute radial profile from center
        center = np.array(amplitude.shape) // 2
        x, y, z = np.meshgrid(
            np.arange(amplitude.shape[0]) - center[0],
            np.arange(amplitude.shape[1]) - center[1], 
            np.arange(amplitude.shape[2]) - center[2],
            indexing='ij'
        )
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # Find tail region (outer 50% of domain)
        r_max = np.max(r)
        tail_mask = r > 0.5 * r_max
        
        if np.sum(tail_mask) < 10:  # Need sufficient points
            return {
                "tail_slope": -2.0,
                "r_squared": 0.0,
                "power_law_range": [0.5 * r_max, r_max]
            }
        
        # Extract tail data
        r_tail = r[tail_mask]
        amp_tail = amplitude[tail_mask]
        
        # Remove zeros and take log
        valid_mask = amp_tail > 1e-12
        if np.sum(valid_mask) < 5:
            return {
                "tail_slope": -2.0,
                "r_squared": 0.0,
                "power_law_range": [0.5 * r_max, r_max]
            }
        
        log_r = np.log(r_tail[valid_mask])
        log_amp = np.log(amp_tail[valid_mask])
        
        # Linear fit: log(amp) = slope * log(r) + intercept
        try:
            slope, intercept = np.polyfit(log_r, log_amp, 1)
            
            # Compute R-squared
            amp_pred = np.exp(slope * log_r + intercept)
            ss_res = np.sum((amp_tail[valid_mask] - amp_pred)**2)
            ss_tot = np.sum((amp_tail[valid_mask] - np.mean(amp_tail[valid_mask]))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            return {
                "tail_slope": float(slope),
                "r_squared": float(r_squared),
                "power_law_range": [float(0.5 * r_max), float(r_max)]
            }
        except (np.linalg.LinAlgError, ValueError):
            return {
                "tail_slope": -2.0,
                "r_squared": 0.0,
                "power_law_range": [0.5 * r_max, r_max]
            }
    
    def _check_spherical_nodes(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Check for absence of spherical standing nodes.
        
        Physical Meaning:
            Detects spherical standing wave nodes in the BVP envelope,
            which should be absent in the fundamental field configuration
            according to the theory.
        """
        amplitude = np.abs(envelope)
        
        # Find local minima that could be nodes
        # Use morphological operations to find local minima
        local_minima = minimum_filter(amplitude, size=3) == amplitude
        local_maxima = maximum_filter(amplitude, size=3) == amplitude
        
        # Nodes are local minima with very low amplitude
        node_threshold = 0.01 * np.max(amplitude)
        potential_nodes = local_minima & (amplitude < node_threshold)
        
        # Check if nodes form spherical patterns
        node_locations = np.where(potential_nodes)
        node_count = len(node_locations[0])
        
        # Analyze spherical symmetry of nodes
        has_spherical_nodes = False
        if node_count > 0:
            # Compute center of mass of nodes
            center = np.array(amplitude.shape) // 2
            node_positions = np.column_stack(node_locations)
            
            # Check if nodes are distributed spherically around center
            distances = np.linalg.norm(node_positions - center, axis=1)
            if len(distances) > 3:
                # Check for spherical clustering
                mean_distance = np.mean(distances)
                std_distance = np.std(distances)
                spherical_coefficient = std_distance / mean_distance if mean_distance > 0 else 1.0
                
                # If nodes are clustered in spherical shells, we have spherical nodes
                has_spherical_nodes = spherical_coefficient < 0.3 and node_count > 5
        
        return {
            "has_spherical_nodes": has_spherical_nodes,
            "node_count": node_count,
            "node_locations": [(int(x), int(y), int(z)) for x, y, z in zip(*node_locations)]
        }
    
    def _compute_topological_charge(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Compute topological charge of defects.
        
        Physical Meaning:
            Calculates the topological charge of defects in the BVP envelope
            using the winding number around closed loops in the field.
        """
        # Convert to complex field for phase analysis
        if np.iscomplexobj(envelope):
            field = envelope
        else:
            # Create complex field from real envelope
            field = envelope.astype(complex)
        
        # Compute phase field
        phase = np.angle(field)
        
        # Find phase singularities (defects)
        # Compute phase gradients
        grad_phase_x = np.gradient(phase, axis=0)
        grad_phase_y = np.gradient(phase, axis=1)
        grad_phase_z = np.gradient(phase, axis=2)
        
        # Handle phase wrapping
        grad_phase_x = np.unwrap(grad_phase_x, axis=0)
        grad_phase_y = np.unwrap(grad_phase_y, axis=1)
        grad_phase_z = np.unwrap(grad_phase_z, axis=2)
        
        # Compute winding number around each point
        # For 3D, we compute the circulation around small loops
        charge_locations = []
        total_charge = 0.0
        
        # Sample points for charge detection
        step = max(1, min(envelope.shape) // 16)  # Adaptive sampling
        for i in range(step, envelope.shape[0] - step, step):
            for j in range(step, envelope.shape[1] - step, step):
                for k in range(step, envelope.shape[2] - step, step):
                    # Compute circulation around small loop
                    try:
                        # 2D circulation in xy plane
                        circulation_xy = (
                            grad_phase_x[i+1, j, k] - grad_phase_x[i-1, j, k] +
                            grad_phase_y[i, j+1, k] - grad_phase_y[i, j-1, k]
                        )
                        
                        # 2D circulation in xz plane  
                        circulation_xz = (
                            grad_phase_x[i+1, j, k] - grad_phase_x[i-1, j, k] +
                            grad_phase_z[i, j, k+1] - grad_phase_z[i, j, k-1]
                        )
                        
                        # 2D circulation in yz plane
                        circulation_yz = (
                            grad_phase_y[i, j+1, k] - grad_phase_y[i, j-1, k] +
                            grad_phase_z[i, j, k+1] - grad_phase_z[i, j, k-1]
                        )
                        
                        # Average circulation as charge estimate
                        circulation = (circulation_xy + circulation_xz + circulation_yz) / 3.0
                        
                        # Convert to topological charge (winding number)
                        charge = circulation / (2 * np.pi)
                        
                        # Threshold for significant charge
                        if abs(charge) > 0.1:
                            charge_locations.append((i, j, k))
                            total_charge += charge
                            
                    except (IndexError, ValueError):
                        continue
        
        # Compute charge stability (how well-defined the charges are)
        if len(charge_locations) > 0:
            # Stability based on charge magnitude and spatial distribution
            charge_magnitudes = [abs(total_charge / len(charge_locations))] * len(charge_locations)
            charge_stability = min(1.0, np.mean(charge_magnitudes))
        else:
            charge_stability = 0.0
        
        return {
            "topological_charge": float(total_charge),
            "charge_locations": charge_locations,
            "charge_stability": float(charge_stability)
        }
    
    def _analyze_zone_separation(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Analyze separation of core/transition/tail zones.
        
        Physical Meaning:
            Identifies the three characteristic zones in the BVP envelope:
            core (high amplitude, nonlinear), transition (intermediate),
            and tail (low amplitude, linear) regions.
        """
        amplitude = np.abs(envelope)
        
        # Compute radial profile from center
        center = np.array(amplitude.shape) // 2
        x, y, z = np.meshgrid(
            np.arange(amplitude.shape[0]) - center[0],
            np.arange(amplitude.shape[1]) - center[1], 
            np.arange(amplitude.shape[2]) - center[2],
            indexing='ij'
        )
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # Find maximum radius
        r_max = np.max(r)
        
        # Compute radial average
        r_bins = np.linspace(0, r_max, 50)
        radial_profile = []
        for i in range(len(r_bins) - 1):
            mask = (r >= r_bins[i]) & (r < r_bins[i+1])
            if np.sum(mask) > 0:
                radial_profile.append(np.mean(amplitude[mask]))
            else:
                radial_profile.append(0.0)
        
        radial_profile = np.array(radial_profile)
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
        
        # Find zone boundaries based on amplitude thresholds
        max_amplitude = np.max(radial_profile)
        
        # Core zone: amplitude > 0.5 * max
        core_threshold = 0.5 * max_amplitude
        core_mask = radial_profile > core_threshold
        if np.any(core_mask):
            core_radius = r_centers[core_mask][-1] / r_max  # Last point above threshold
        else:
            core_radius = 0.1
        
        # Transition zone: 0.1 * max < amplitude < 0.5 * max
        transition_low = 0.1 * max_amplitude
        transition_high = 0.5 * max_amplitude
        transition_mask = (radial_profile > transition_low) & (radial_profile < transition_high)
        if np.any(transition_mask):
            transition_radius = r_centers[transition_mask][-1] / r_max
        else:
            transition_radius = 0.3
        
        # Tail zone: amplitude < 0.1 * max
        tail_threshold = 0.1 * max_amplitude
        tail_mask = radial_profile < tail_threshold
        if np.any(tail_mask):
            tail_radius = r_centers[tail_mask][0] / r_max  # First point below threshold
        else:
            tail_radius = 1.0
        
        # Compute zone indicators (N, S, C from theory)
        # N: Nonlinearity parameter in core
        core_region = r < core_radius * r_max
        if np.sum(core_region) > 0:
            core_amplitude = np.mean(amplitude[core_region])
            N = core_amplitude / max_amplitude if max_amplitude > 0 else 0.0
        else:
            N = 0.0
        
        # S: Scale separation parameter
        S = core_radius / transition_radius if transition_radius > 0 else 1.0
        
        # C: Coherence parameter
        C = transition_radius / tail_radius if tail_radius > 0 else 1.0
        
        return {
            "core_radius": float(core_radius),
            "transition_radius": float(transition_radius),
            "tail_radius": float(tail_radius),
            "zone_indicators": {
                "N": float(N),
                "S": float(S), 
                "C": float(C)
            }
        }
