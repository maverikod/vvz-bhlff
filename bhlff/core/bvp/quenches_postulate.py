"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Quenches Postulate implementation for BVP framework.

This module implements Postulate 5 of the BVP framework, which states that
BVP field exhibits "quenches" - localized regions where field amplitude
drops significantly, creating energy dumps and phase discontinuities.

Theoretical Background:
    Quenches represent localized energy dissipation events in the BVP field
    where field amplitude drops below critical thresholds. These events
    are essential for understanding field dynamics and energy transport.

Example:
    >>> postulate = QuenchesPostulate(domain, constants)
    >>> results = postulate.apply(envelope)
"""

import numpy as np
from typing import Dict, Any, List
from ..domain.domain import Domain
from .bvp_constants import BVPConstants
from .bvp_postulate_base import BVPPostulate


class QuenchesPostulate(BVPPostulate):
    """
    Postulate 5: Quenches.
    
    Physical Meaning:
        BVP field exhibits "quenches" - localized regions where
        field amplitude drops significantly, creating energy dumps.
    """
    
    def __init__(self, domain: Domain, constants: BVPConstants):
        """
        Initialize quenches postulate.
        
        Physical Meaning:
            Sets up the postulate with domain and constants for
            detecting and analyzing quench events.
            
        Args:
            domain (Domain): Computational domain for analysis.
            constants (BVPConstants): BVP physical constants.
        """
        self.domain = domain
        self.constants = constants
        self.quench_threshold = constants.get_quench_parameter("quench_threshold", 0.1)
        self.energy_dump_threshold = constants.get_quench_parameter("energy_dump_threshold", 0.01)
        self.min_quench_size = constants.get_quench_parameter("min_quench_size", 5)
    
    def apply(self, envelope: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Apply quenches postulate.
        
        Physical Meaning:
            Detects and analyzes quench events in the BVP field,
            including energy dumps and phase discontinuities.
            
        Mathematical Foundation:
            Identifies regions where field amplitude drops below
            critical thresholds and analyzes energy dissipation.
            
        Args:
            envelope (np.ndarray): BVP envelope to analyze.
            
        Returns:
            Dict[str, Any]: Results including quench detection,
                energy analysis, and quench validation.
        """
        # Detect quench events
        quench_detection = self._detect_quenches(envelope)
        
        # Analyze quench properties
        quench_analysis = self._analyze_quench_properties(envelope, quench_detection)
        
        # Compute energy dumps
        energy_analysis = self._analyze_energy_dumps(envelope, quench_detection)
        
        # Validate quenches
        satisfies_postulate = self._validate_quenches(quench_analysis, energy_analysis)
        
        return {
            "quench_detection": quench_detection,
            "quench_analysis": quench_analysis,
            "energy_analysis": energy_analysis,
            "satisfies_postulate": satisfies_postulate,
            "postulate_satisfied": satisfies_postulate
        }
    
    def _detect_quenches(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Detect quench events in the field.
        
        Physical Meaning:
            Identifies localized regions where field amplitude
            drops below critical thresholds.
            
        Args:
            envelope (np.ndarray): BVP envelope.
            
        Returns:
            Dict[str, Any]: Quench detection results.
        """
        amplitude = np.abs(envelope)
        
        # Compute amplitude statistics
        mean_amplitude = np.mean(amplitude)
        std_amplitude = np.std(amplitude)
        
        # Define quench threshold
        quench_threshold_value = mean_amplitude - self.quench_threshold * std_amplitude
        
        # Find quench regions
        quench_mask = amplitude < quench_threshold_value
        
        # Filter small quenches
        quench_mask = self._filter_small_quenches(quench_mask)
        
        # Find quench locations
        quench_locations = self._find_quench_locations(quench_mask)
        
        # Compute quench statistics
        num_quenches = len(quench_locations)
        quench_volume = np.sum(quench_mask)
        quench_fraction = quench_volume / amplitude.size
        
        return {
            "quench_mask": quench_mask,
            "quench_locations": quench_locations,
            "num_quenches": num_quenches,
            "quench_volume": quench_volume,
            "quench_fraction": quench_fraction,
            "quench_threshold_value": quench_threshold_value
        }
    
    def _filter_small_quenches(self, quench_mask: np.ndarray) -> np.ndarray:
        """
        Filter out small quench regions.
        
        Physical Meaning:
            Removes quench regions that are too small to be
            physically significant.
            
        Args:
            quench_mask (np.ndarray): Binary quench mask.
            
        Returns:
            np.ndarray: Filtered quench mask.
        """
        # Use morphological operations to filter small regions
        from scipy import ndimage
        
        # Remove small connected components
        filtered_mask = ndimage.binary_opening(quench_mask, 
                                            structure=ndimage.generate_binary_structure(3, 1))
        
        # Remove very small regions
        labeled_mask, num_features = ndimage.label(filtered_mask)
        for i in range(1, num_features + 1):
            region_size = np.sum(labeled_mask == i)
            if region_size < self.min_quench_size:
                filtered_mask[labeled_mask == i] = False
        
        return filtered_mask
    
    def _find_quench_locations(self, quench_mask: np.ndarray) -> List[tuple]:
        """
        Find center locations of quench regions.
        
        Physical Meaning:
            Identifies center coordinates of each quench region
            for further analysis.
            
        Args:
            quench_mask (np.ndarray): Binary quench mask.
            
        Returns:
            List[tuple]: List of quench center coordinates.
        """
        from scipy import ndimage
        
        # Label connected components
        labeled_mask, num_features = ndimage.label(quench_mask)
        
        # Find centers of mass for each quench
        quench_locations = []
        for i in range(1, num_features + 1):
            region_mask = labeled_mask == i
            center = ndimage.center_of_mass(region_mask)
            quench_locations.append(tuple(center))
        
        return quench_locations
    
    def _analyze_quench_properties(self, envelope: np.ndarray, 
                                 quench_detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze properties of detected quenches.
        
        Physical Meaning:
            Computes detailed properties of quench events
            including size, shape, and amplitude characteristics.
            
        Args:
            envelope (np.ndarray): BVP envelope.
            quench_detection (Dict[str, Any]): Quench detection results.
            
        Returns:
            Dict[str, Any]: Quench properties analysis.
        """
        amplitude = np.abs(envelope)
        quench_mask = quench_detection["quench_mask"]
        quench_locations = quench_detection["quench_locations"]
        
        # Analyze individual quenches
        quench_properties = []
        for i, location in enumerate(quench_locations):
            properties = self._analyze_individual_quench(amplitude, quench_mask, location, i)
            quench_properties.append(properties)
        
        # Compute overall quench statistics
        if quench_properties:
            avg_quench_size = np.mean([q["size"] for q in quench_properties])
            avg_quench_amplitude = np.mean([q["min_amplitude"] for q in quench_properties])
            avg_quench_depth = np.mean([q["depth"] for q in quench_properties])
        else:
            avg_quench_size = 0
            avg_quench_amplitude = 0
            avg_quench_depth = 0
        
        return {
            "individual_quenches": quench_properties,
            "avg_quench_size": avg_quench_size,
            "avg_quench_amplitude": avg_quench_amplitude,
            "avg_quench_depth": avg_quench_depth
        }
    
    def _analyze_individual_quench(self, amplitude: np.ndarray, quench_mask: np.ndarray, 
                                 location: tuple, quench_id: int) -> Dict[str, Any]:
        """
        Analyze properties of individual quench.
        
        Physical Meaning:
            Computes detailed properties of a single quench
            event including size, amplitude, and depth.
            
        Args:
            amplitude (np.ndarray): Field amplitude.
            quench_mask (np.ndarray): Binary quench mask.
            location (tuple): Quench center location.
            quench_id (int): Quench identifier.
            
        Returns:
            Dict[str, Any]: Individual quench properties.
        """
        # Find quench region around location
        quench_region = self._extract_quench_region(amplitude, quench_mask, location)
        
        # Compute quench properties
        quench_amplitude = amplitude[quench_region]
        min_amplitude = np.min(quench_amplitude)
        max_amplitude = np.max(quench_amplitude)
        mean_amplitude = np.mean(quench_amplitude)
        
        # Compute quench depth
        surrounding_amplitude = self._compute_surrounding_amplitude(amplitude, location)
        depth = surrounding_amplitude - min_amplitude
        
        # Compute quench size
        size = np.sum(quench_region)
        
        return {
            "id": quench_id,
            "location": location,
            "size": size,
            "min_amplitude": min_amplitude,
            "max_amplitude": max_amplitude,
            "mean_amplitude": mean_amplitude,
            "depth": depth
        }
    
    def _extract_quench_region(self, amplitude: np.ndarray, quench_mask: np.ndarray, 
                             location: tuple) -> np.ndarray:
        """
        Extract region around quench location.
        
        Physical Meaning:
            Creates mask for quench region around specified
            location for detailed analysis.
            
        Args:
            amplitude (np.ndarray): Field amplitude.
            quench_mask (np.ndarray): Binary quench mask.
            location (tuple): Quench center location.
            
        Returns:
            np.ndarray: Quench region mask.
        """
        # Create region mask around location
        region_size = 10  # Adjust based on domain size
        region_mask = np.zeros_like(amplitude, dtype=bool)
        
        # Define region bounds
        bounds = []
        for i, coord in enumerate(location):
            lower = max(0, int(coord - region_size))
            upper = min(amplitude.shape[i], int(coord + region_size))
            bounds.append((lower, upper))
        
        # Create region mask
        region_mask[bounds[0][0]:bounds[0][1], 
                   bounds[1][0]:bounds[1][1], 
                   bounds[2][0]:bounds[2][1]] = True
        
        # Intersect with quench mask
        quench_region = region_mask & quench_mask
        
        return quench_region
    
    def _compute_surrounding_amplitude(self, amplitude: np.ndarray, location: tuple) -> float:
        """
        Compute amplitude in surrounding region.
        
        Physical Meaning:
            Calculates average amplitude in region surrounding
            quench to determine quench depth.
            
        Args:
            amplitude (np.ndarray): Field amplitude.
            location (tuple): Quench center location.
            
        Returns:
            float: Surrounding amplitude.
        """
        # Define surrounding region
        region_size = 20  # Larger than quench region
        surrounding_amplitude = []
        
        # Sample surrounding region
        for i in range(-region_size, region_size + 1):
            for j in range(-region_size, region_size + 1):
                for k in range(-region_size, region_size + 1):
                    x = int(location[0] + i)
                    y = int(location[1] + j)
                    z = int(location[2] + k)
                    
                    if (0 <= x < amplitude.shape[0] and 
                        0 <= y < amplitude.shape[1] and 
                        0 <= z < amplitude.shape[2]):
                        surrounding_amplitude.append(amplitude[x, y, z])
        
        return np.mean(surrounding_amplitude) if surrounding_amplitude else 0
    
    def _analyze_energy_dumps(self, envelope: np.ndarray, 
                            quench_detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze energy dumps at quench locations.
        
        Physical Meaning:
            Computes energy dissipation at quench locations
            to quantify energy dump events.
            
        Args:
            envelope (np.ndarray): BVP envelope.
            quench_detection (Dict[str, Any]): Quench detection results.
            
        Returns:
            Dict[str, Any]: Energy dump analysis.
        """
        amplitude = np.abs(envelope)
        quench_mask = quench_detection["quench_mask"]
        
        # Compute energy in quench regions
        quench_energy = amplitude[quench_mask]**2
        total_quench_energy = np.sum(quench_energy)
        
        # Compute energy dump rate
        energy_dump_rate = total_quench_energy / (amplitude.size * np.mean(amplitude**2))
        
        # Check if energy dumps are significant
        significant_dumps = energy_dump_rate > self.energy_dump_threshold
        
        return {
            "total_quench_energy": total_quench_energy,
            "energy_dump_rate": energy_dump_rate,
            "significant_dumps": significant_dumps
        }
    
    def _validate_quenches(self, quench_analysis: Dict[str, Any], 
                         energy_analysis: Dict[str, Any]) -> bool:
        """
        Validate quenches postulate.
        
        Physical Meaning:
            Checks that quench events are present and
            energy dumps are significant.
            
        Args:
            quench_analysis (Dict[str, Any]): Quench analysis.
            energy_analysis (Dict[str, Any]): Energy analysis.
            
        Returns:
            bool: True if quenches postulate is satisfied.
        """
        # Check if quenches are present
        has_quenches = quench_analysis["avg_quench_size"] > 0
        
        # Check if energy dumps are significant
        significant_dumps = energy_analysis["significant_dumps"]
        
        return has_quenches and significant_dumps
