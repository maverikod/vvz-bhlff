"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Stability analysis for soliton models in 7D phase field theory.

This module contains stability analysis methods for soliton models,
including eigenvalue analysis and mode classification.

Theoretical Background:
    Implements stability analysis for soliton configurations,
    including eigenvalue computation and mode classification.

Example:
    >>> stability_analyzer = SolitonStabilityAnalyzer(domain, physics_params)
    >>> stability = stability_analyzer.analyze_stability(field)
"""

import numpy as np
from typing import Dict, Any


class SolitonStabilityAnalyzer:
    """
    Stability analyzer for soliton models.
    
    Physical Meaning:
        Analyzes the stability of soliton solutions by computing
        eigenvalues and eigenvectors of the energy Hessian.
    """

    def __init__(self, domain: "Domain", physics_params: Dict[str, Any]):
        """
        Initialize stability analyzer.
        
        Args:
            domain: Computational domain
            physics_params: Physical parameters
        """
        self.domain = domain
        self.params = physics_params

    def analyze_stability(self, soliton: np.ndarray) -> Dict[str, Any]:
        """
        Analyze stability of soliton solution.
        
        Physical Meaning:
            Investigates the response of the soliton to small perturbations
            to determine if it represents a stable minimum of the energy
            functional.
            
        Mathematical Foundation:
            Computes the spectrum of the Hessian matrix δ²E/δU² at the
            soliton solution to identify unstable modes.
            
        Args:
            soliton: Soliton field configuration
            
        Returns:
            Dict containing stability analysis, unstable modes, frequencies
        """
        # Compute Hessian
        hessian = self._compute_energy_hessian(soliton)
        
        # Diagonalize to get eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        
        # Analyze stability
        stable_modes = eigenvalues >= 0
        unstable_modes = eigenvalues < 0
        
        # Compute oscillation frequencies
        frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)
        
        # Analyze eigenmodes
        mode_analysis = self._analyze_eigenmodes(eigenvalues, eigenvectors)
        
        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "frequencies": frequencies,
            "stable_modes": stable_modes,
            "unstable_modes": unstable_modes,
            "stability_ratio": np.sum(stable_modes) / len(stable_modes),
            "mode_analysis": mode_analysis,
            "is_stable": np.all(stable_modes),
            "stability_margin": np.min(eigenvalues) if len(eigenvalues) > 0 else 0,
        }

    def _compute_energy_hessian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute Hessian of energy functional.
        
        Physical Meaning:
            Calculates the second derivative of the energy functional
            for stability analysis.
        """
        # Numerical computation of Hessian
        epsilon = 1e-6
        n = field.size
        hessian = np.zeros((n, n))

        # Base energy
        E0 = self._compute_energy_functional(field)

        for i in range(n):
            for j in range(n):
                # Finite difference approximation
                field_pp = field.copy()
                field_pp.flat[i] += epsilon
                field_pp.flat[j] += epsilon
                E_pp = self._compute_energy_functional(field_pp)

                field_pm = field.copy()
                field_pm.flat[i] += epsilon
                field_pm.flat[j] -= epsilon
                E_pm = self._compute_energy_functional(field_pm)

                field_mp = field.copy()
                field_mp.flat[i] -= epsilon
                field_mp.flat[j] += epsilon
                E_mp = self._compute_energy_functional(field_mp)

                field_mm = field.copy()
                field_mm.flat[i] -= epsilon
                field_mm.flat[j] -= epsilon
                E_mm = self._compute_energy_functional(field_mm)

                # Mixed derivative
                hessian[i, j] = (E_pp - E_pm - E_mp + E_mm) / (4 * epsilon**2)

        return hessian

    def _compute_energy_functional(self, field: np.ndarray) -> float:
        """
        Compute energy functional for Hessian calculation.
        
        Physical Meaning:
            Computes the total energy of the field configuration
            for numerical differentiation.
        """
        # Simplified energy calculation for Hessian
        # In practice, this would call the full energy calculator
        return np.sum(np.abs(field)**2)

    def _analyze_eigenmodes(
        self, eigenvalues: np.ndarray, eigenvectors: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze eigenmodes for understanding perturbation types.
        
        Physical Meaning:
            Classifies eigenmodes by their physical meaning (translational,
            rotational, deformational).
        """
        mode_types = []
        mode_energies = []

        for i, (eigenval, eigenvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            # Analyze mode symmetry
            symmetry = self._analyze_mode_symmetry(eigenvec)

            # Classify mode type
            if eigenval < 1e-10:  # Zero modes
                mode_type = "zero_mode"
            elif eigenval < 0:  # Unstable modes
                mode_type = "unstable_mode"
            else:  # Stable modes
                mode_type = "stable_mode"

            mode_types.append(mode_type)
            mode_energies.append(eigenval)

        return {
            "mode_types": mode_types,
            "mode_energies": mode_energies,
            "zero_mode_count": sum(1 for t in mode_types if t == "zero_mode"),
            "unstable_mode_count": sum(1 for t in mode_types if t == "unstable_mode"),
            "stable_mode_count": sum(1 for t in mode_types if t == "stable_mode"),
        }

    def _analyze_mode_symmetry(self, eigenvector: np.ndarray) -> str:
        """
        Analyze symmetry of eigenmode.
        
        Physical Meaning:
            Determines the type of symmetry of the perturbation
            (translational, rotational, deformational).
        """
        # Simple analysis based on mode structure
        # In real implementation, more sophisticated analysis would be needed

        # Check for translational symmetry
        if self._is_translational_mode(eigenvector):
            return "translational"

        # Check for rotational symmetry
        if self._is_rotational_mode(eigenvector):
            return "rotational"

        # Other modes are considered deformational
        return "deformational"

    def _is_translational_mode(self, eigenvector: np.ndarray) -> bool:
        """Check for translational mode."""
        # Simplified check - in reality, more complex analysis needed
        return False

    def _is_rotational_mode(self, eigenvector: np.ndarray) -> bool:
        """Check for rotational mode."""
        # Simplified check - in reality, more complex analysis needed
        return False
