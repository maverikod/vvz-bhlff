"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Collective modes analysis module for multi-particle systems.

This module implements collective modes analysis functionality
for Level F models in 7D phase field theory.

Physical Meaning:
    Implements collective modes analysis including mode finding,
    stability analysis, and mode interactions.

Example:
    >>> analyzer = CollectiveModesAnalyzer(domain, particles, system_params)
    >>> modes = analyzer.find_collective_modes()
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy.linalg import eig
from ..base.abstract_model import AbstractModel
from .data_structures import Particle, SystemParameters


class CollectiveModesAnalyzer(AbstractModel):
    """
    Collective modes analysis for multi-particle systems.

    Physical Meaning:
        Analyzes collective modes in multi-particle systems,
        including mode finding, stability analysis, and
        mode interactions.

    Mathematical Foundation:
        Implements collective modes analysis methods:
        - Mode finding: diagonalization of dynamics matrix M⁻¹K
        - Stability analysis: eigenvalue analysis
        - Mode interactions: coupling analysis
    """

    def __init__(self, domain, particles: List[Particle], system_params: SystemParameters):
        """
        Initialize collective modes analyzer.

        Physical Meaning:
            Sets up the collective modes analysis system with
            domain, particles, and system parameters.

        Args:
            domain: Computational domain
            particles (List[Particle]): List of particles
            system_params (SystemParameters): System parameters
        """
        super().__init__()
        self.domain = domain
        self.particles = particles
        self.system_params = system_params
        
        # Initialize collective modes analysis
        self._initialize_collective_modes_analysis()

    def _initialize_collective_modes_analysis(self) -> None:
        """
        Initialize collective modes analysis.

        Physical Meaning:
            Initializes the collective modes analysis system with
            dynamics matrices and mode analysis methods.
        """
        # Set up dynamics matrices
        self._setup_dynamics_matrices()
        
        # Set up mode analysis methods
        self._setup_mode_analysis_methods()

    def _setup_dynamics_matrices(self) -> None:
        """
        Setup dynamics matrices.

        Physical Meaning:
            Sets up the dynamics matrices for the system
            based on particle properties and interactions.
        """
        n_particles = len(self.particles)
        
        # Initialize mass matrix
        self.mass_matrix = np.zeros((n_particles, n_particles))
        for i, particle in enumerate(self.particles):
            self.mass_matrix[i, i] = particle.mass
        
        # Initialize stiffness matrix
        self.stiffness_matrix = np.zeros((n_particles, n_particles))
        for i, particle_i in enumerate(self.particles):
            for j, particle_j in enumerate(self.particles):
                if i != j:
                    # Stiffness from interaction
                    distance = particle_i.distance_to(particle_j)
                    if distance <= self.system_params.interaction_range:
                        interaction_strength = particle_i.interaction_strength(
                            particle_j, self.system_params.interaction_range
                        )
                        self.stiffness_matrix[i, j] = -interaction_strength
                        self.stiffness_matrix[i, i] += interaction_strength

    def _setup_mode_analysis_methods(self) -> None:
        """
        Setup mode analysis methods.

        Physical Meaning:
            Sets up the methods for collective mode analysis
            including mode finding and stability analysis.
        """
        # Mode finding method
        self.mode_finder = self._find_collective_modes
        
        # Stability analysis method
        self.stability_analyzer = self._analyze_mode_stability
        
        # Mode interaction analyzer
        self.interaction_analyzer = self._analyze_mode_interactions

    def find_collective_modes(self) -> Dict[str, Any]:
        """
        Find collective modes of the system.

        Physical Meaning:
            Identifies collective excitations that involve
            coordinated motion of multiple particles.

        Returns:
            Dict[str, Any]: Collective modes analysis including:
                - frequencies: ω_n (mode frequencies)
                - amplitudes: A_n (mode amplitudes)
                - eigenvectors: v_n (mode shapes)
                - stability: stability analysis
        """
        # Compute dynamics matrix
        dynamics_matrix = self._compute_dynamics_matrix()

        # Find eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eig(dynamics_matrix)

        # Analyze modes
        modes = self._analyze_collective_modes(eigenvalues, eigenvectors)

        # Analyze stability
        stability = self._analyze_mode_stability(eigenvalues)

        # Analyze mode interactions
        interactions = self._analyze_mode_interactions(eigenvalues, eigenvectors)

        return {
            "frequencies": modes["frequencies"],
            "amplitudes": modes["amplitudes"],
            "eigenvectors": modes["eigenvectors"],
            "stability": stability,
            "interactions": interactions,
            "num_modes": len(modes["frequencies"]),
        }

    def _compute_dynamics_matrix(self) -> np.ndarray:
        """
        Compute dynamics matrix.

        Physical Meaning:
            Computes the dynamics matrix M⁻¹K for the system.

        Returns:
            np.ndarray: Dynamics matrix.
        """
        # Compute inverse mass matrix
        mass_inv = np.linalg.inv(self.mass_matrix)

        # Compute dynamics matrix
        dynamics_matrix = np.dot(mass_inv, self.stiffness_matrix)

        return dynamics_matrix

    def _analyze_collective_modes(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> Dict[str, Any]:
        """
        Analyze collective modes.

        Physical Meaning:
            Analyzes the collective modes from eigenvalues
            and eigenvectors.

        Args:
            eigenvalues (np.ndarray): Eigenvalues.
            eigenvectors (np.ndarray): Eigenvectors.

        Returns:
            Dict[str, Any]: Collective modes analysis.
        """
        # Extract frequencies
        frequencies = np.sqrt(-eigenvalues.real)
        frequencies = frequencies[np.isfinite(frequencies)]

        # Extract amplitudes
        amplitudes = np.abs(eigenvectors)
        amplitudes = amplitudes[:, np.isfinite(frequencies)]

        # Extract mode shapes
        mode_shapes = eigenvectors[:, np.isfinite(frequencies)]

        return {
            "frequencies": frequencies.tolist(),
            "amplitudes": amplitudes.tolist(),
            "eigenvectors": mode_shapes.tolist(),
        }

    def _analyze_mode_stability(self, eigenvalues: np.ndarray) -> Dict[str, Any]:
        """
        Analyze mode stability.

        Physical Meaning:
            Analyzes the stability of collective modes
            based on eigenvalues.

        Args:
            eigenvalues (np.ndarray): Eigenvalues.

        Returns:
            Dict[str, Any]: Stability analysis.
        """
        # Analyze stability
        stable_modes = np.sum(eigenvalues.real < 0)
        unstable_modes = np.sum(eigenvalues.real > 0)
        marginal_modes = np.sum(np.abs(eigenvalues.real) < 1e-12)

        # Determine overall stability
        if unstable_modes == 0:
            stability = "stable"
        elif stable_modes > unstable_modes:
            stability = "mostly_stable"
        else:
            stability = "unstable"

        return {
            "stable_modes": int(stable_modes),
            "unstable_modes": int(unstable_modes),
            "marginal_modes": int(marginal_modes),
            "stability": stability,
            "max_growth_rate": float(np.max(eigenvalues.real)),
        }

    def _analyze_mode_interactions(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> Dict[str, Any]:
        """
        Analyze mode interactions.

        Physical Meaning:
            Analyzes interactions between collective modes
            based on eigenvalues and eigenvectors.

        Args:
            eigenvalues (np.ndarray): Eigenvalues.
            eigenvectors (np.ndarray): Eigenvectors.

        Returns:
            Dict[str, Any]: Mode interaction analysis.
        """
        # Calculate mode coupling
        mode_coupling = self._calculate_mode_coupling(eigenvalues, eigenvectors)

        # Calculate mode overlap
        mode_overlap = self._calculate_mode_overlap(eigenvectors)

        # Calculate mode correlation
        mode_correlation = self._calculate_mode_correlation(eigenvectors)

        return {
            "mode_coupling": mode_coupling,
            "mode_overlap": mode_overlap,
            "mode_correlation": mode_correlation,
            "interactions_detected": mode_coupling > 0.1,
        }

    def _calculate_mode_coupling(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> float:
        """
        Calculate mode coupling.

        Physical Meaning:
            Calculates the coupling between collective modes
            based on eigenvalues and eigenvectors.

        Args:
            eigenvalues (np.ndarray): Eigenvalues.
            eigenvectors (np.ndarray): Eigenvectors.

        Returns:
            float: Mode coupling strength.
        """
        # Calculate coupling based on eigenvalue spacing
        frequencies = np.sqrt(-eigenvalues.real)
        frequencies = frequencies[np.isfinite(frequencies)]
        
        if len(frequencies) < 2:
            return 0.0
        
        # Calculate frequency spacing
        frequency_spacing = np.diff(np.sort(frequencies))
        
        # Calculate coupling strength
        coupling_strength = np.mean(frequency_spacing) / np.mean(frequencies)
        
        return float(coupling_strength)

    def _calculate_mode_overlap(self, eigenvectors: np.ndarray) -> float:
        """
        Calculate mode overlap.

        Physical Meaning:
            Calculates the overlap between collective modes
            based on eigenvectors.

        Args:
            eigenvectors (np.ndarray): Eigenvectors.

        Returns:
            float: Mode overlap strength.
        """
        # Calculate overlap between modes
        n_modes = eigenvectors.shape[1]
        if n_modes < 2:
            return 0.0
        
        # Calculate pairwise overlaps
        overlaps = []
        for i in range(n_modes):
            for j in range(i + 1, n_modes):
                overlap = np.abs(np.dot(eigenvectors[:, i], eigenvectors[:, j]))
                overlaps.append(overlap)
        
        # Calculate average overlap
        average_overlap = np.mean(overlaps) if overlaps else 0.0
        
        return float(average_overlap)

    def _calculate_mode_correlation(self, eigenvectors: np.ndarray) -> float:
        """
        Calculate mode correlation.

        Physical Meaning:
            Calculates the correlation between collective modes
            based on eigenvectors.

        Args:
            eigenvectors (np.ndarray): Eigenvectors.

        Returns:
            float: Mode correlation strength.
        """
        # Calculate correlation between modes
        n_modes = eigenvectors.shape[1]
        if n_modes < 2:
            return 0.0
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(n_modes):
            for j in range(i + 1, n_modes):
                correlation = np.corrcoef(eigenvectors[:, i], eigenvectors[:, j])[0, 1]
                if not np.isnan(correlation):
                    correlations.append(correlation)
        
        # Calculate average correlation
        average_correlation = np.mean(correlations) if correlations else 0.0
        
        return float(average_correlation)

    def analyze_mode_spectrum(self, modes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze mode spectrum.

        Physical Meaning:
            Analyzes the spectrum of collective modes
            to identify characteristic frequencies and
            spectral features.

        Args:
            modes (Dict[str, Any]): Collective modes.

        Returns:
            Dict[str, Any]: Mode spectrum analysis.
        """
        # Extract frequencies
        frequencies = np.array(modes["frequencies"])

        # Analyze frequency distribution
        frequency_analysis = self._analyze_frequency_distribution(frequencies)

        # Analyze spectral features
        spectral_features = self._analyze_spectral_features(frequencies)

        # Analyze mode spacing
        spacing_analysis = self._analyze_mode_spacing(frequencies)

        return {
            "frequency_analysis": frequency_analysis,
            "spectral_features": spectral_features,
            "spacing_analysis": spacing_analysis,
            "spectrum_complete": True,
        }

    def _analyze_frequency_distribution(self, frequencies: np.ndarray) -> Dict[str, Any]:
        """
        Analyze frequency distribution.

        Physical Meaning:
            Analyzes the distribution of mode frequencies
            to identify characteristic patterns.

        Args:
            frequencies (np.ndarray): Mode frequencies.

        Returns:
            Dict[str, Any]: Frequency distribution analysis.
        """
        # Calculate frequency statistics
        mean_frequency = np.mean(frequencies)
        std_frequency = np.std(frequencies)
        min_frequency = np.min(frequencies)
        max_frequency = np.max(frequencies)

        # Analyze frequency range
        frequency_range = max_frequency - min_frequency

        return {
            "mean_frequency": float(mean_frequency),
            "std_frequency": float(std_frequency),
            "min_frequency": float(min_frequency),
            "max_frequency": float(max_frequency),
            "frequency_range": float(frequency_range),
        }

    def _analyze_spectral_features(self, frequencies: np.ndarray) -> Dict[str, Any]:
        """
        Analyze spectral features.

        Physical Meaning:
            Analyzes spectral features in the mode
            frequency spectrum.

        Args:
            frequencies (np.ndarray): Mode frequencies.

        Returns:
            Dict[str, Any]: Spectral features analysis.
        """
        # Find spectral gaps
        gaps = self._find_spectral_gaps(frequencies)

        # Find spectral clusters
        clusters = self._find_spectral_clusters(frequencies)

        # Find spectral peaks
        peaks = self._find_spectral_peaks(frequencies)

        return {
            "gaps": gaps,
            "clusters": clusters,
            "peaks": peaks,
            "num_gaps": len(gaps),
            "num_clusters": len(clusters),
            "num_peaks": len(peaks),
        }

    def _find_spectral_gaps(self, frequencies: np.ndarray) -> List[Dict[str, Any]]:
        """
        Find spectral gaps.

        Physical Meaning:
            Identifies gaps in the frequency spectrum
            where no modes are present.

        Args:
            frequencies (np.ndarray): Mode frequencies.

        Returns:
            List[Dict[str, Any]]: Spectral gaps.
        """
        # Sort frequencies
        sorted_frequencies = np.sort(frequencies)
        
        # Find gaps
        gaps = []
        for i in range(len(sorted_frequencies) - 1):
            gap_size = sorted_frequencies[i + 1] - sorted_frequencies[i]
            if gap_size > 2 * np.std(frequencies):
                gaps.append({
                    "start": float(sorted_frequencies[i]),
                    "end": float(sorted_frequencies[i + 1]),
                    "size": float(gap_size),
                })
        
        return gaps

    def _find_spectral_clusters(self, frequencies: np.ndarray) -> List[Dict[str, Any]]:
        """
        Find spectral clusters.

        Physical Meaning:
            Identifies clusters of modes in the frequency
            spectrum.

        Args:
            frequencies (np.ndarray): Mode frequencies.

        Returns:
            List[Dict[str, Any]]: Spectral clusters.
        """
        # Simplified clustering
        # In practice, this would involve proper clustering algorithms
        clusters = []
        
        # Group frequencies by proximity
        sorted_frequencies = np.sort(frequencies)
        cluster_threshold = np.std(frequencies)
        
        current_cluster = [sorted_frequencies[0]]
        for i in range(1, len(sorted_frequencies)):
            if sorted_frequencies[i] - sorted_frequencies[i - 1] < cluster_threshold:
                current_cluster.append(sorted_frequencies[i])
            else:
                if len(current_cluster) > 1:
                    clusters.append({
                        "frequencies": current_cluster,
                        "center": float(np.mean(current_cluster)),
                        "size": len(current_cluster),
                    })
                current_cluster = [sorted_frequencies[i]]
        
        # Add final cluster
        if len(current_cluster) > 1:
            clusters.append({
                "frequencies": current_cluster,
                "center": float(np.mean(current_cluster)),
                "size": len(current_cluster),
            })
        
        return clusters

    def _find_spectral_peaks(self, frequencies: np.ndarray) -> List[Dict[str, Any]]:
        """
        Find spectral peaks.

        Physical Meaning:
            Identifies peaks in the frequency spectrum
            where modes are concentrated.

        Args:
            frequencies (np.ndarray): Mode frequencies.

        Returns:
            List[Dict[str, Any]]: Spectral peaks.
        """
        # Simplified peak finding
        # In practice, this would involve proper peak detection
        peaks = []
        
        # Find local maxima
        for i in range(1, len(frequencies) - 1):
            if (frequencies[i] > frequencies[i - 1] and 
                frequencies[i] > frequencies[i + 1]):
                peaks.append({
                    "frequency": float(frequencies[i]),
                    "height": float(frequencies[i]),
                    "index": i,
                })
        
        return peaks

    def _analyze_mode_spacing(self, frequencies: np.ndarray) -> Dict[str, Any]:
        """
        Analyze mode spacing.

        Physical Meaning:
            Analyzes the spacing between collective modes
            to identify characteristic patterns.

        Args:
            frequencies (np.ndarray): Mode frequencies.

        Returns:
            Dict[str, Any]: Mode spacing analysis.
        """
        # Calculate mode spacing
        sorted_frequencies = np.sort(frequencies)
        spacings = np.diff(sorted_frequencies)
        
        # Analyze spacing statistics
        mean_spacing = np.mean(spacings)
        std_spacing = np.std(spacings)
        min_spacing = np.min(spacings)
        max_spacing = np.max(spacings)
        
        # Analyze spacing distribution
        spacing_distribution = {
            "mean": float(mean_spacing),
            "std": float(std_spacing),
            "min": float(min_spacing),
            "max": float(max_spacing),
        }
        
        return {
            "spacings": spacings.tolist(),
            "distribution": spacing_distribution,
            "regular_spacing": std_spacing < 0.1 * mean_spacing,
        }
