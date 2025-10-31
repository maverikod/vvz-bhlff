"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Nonlinear effects implementation for Level F models (facade).

Provides the facade class `NonlinearEffects` while avoiding package/module
name collision with `nonlinear/` package. This mirrors the pattern used for
the collective module (`collective_facade.py`).

Theoretical Background:
    Nonlinear effects in multi-particle systems arise from higher-order
    terms that lead to solitonic solutions and nonlinear modes.

Example:
    >>> nonlinear = NonlinearEffects(system, nonlinear_params)
    >>> nonlinear.add_nonlinear_interactions(nonlinear_params)
    >>> modes = nonlinear.find_nonlinear_modes()
    >>> solitons = nonlinear.find_soliton_solutions()
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, List
from ..base.abstract_model import AbstractModel
from .nonlinear.basic_effects import BasicNonlinearEffects
from .nonlinear.soliton_analysis import SolitonAnalyzer
from .nonlinear.mode_analysis import NonlinearModeAnalyzer


class NonlinearEffects(AbstractModel):
    """
    Nonlinear effects in collective systems.

    Physical Meaning:
        Studies nonlinear interactions in multi-particle systems,
        including solitonic solutions and nonlinear modes.
    """

    def __init__(self, system, nonlinear_params: Dict[str, Any]):
        super().__init__()
        self.system = system
        self.nonlinear_params = nonlinear_params

        self.nonlinear_strength = nonlinear_params.get("strength", 1.0)
        self.nonlinear_order = nonlinear_params.get("order", 3)
        self.nonlinear_type = nonlinear_params.get("type", "cubic")

        self.basic_effects = BasicNonlinearEffects(system, nonlinear_params)
        self.soliton_analyzer = SolitonAnalyzer(system, nonlinear_params)
        self.mode_analyzer = NonlinearModeAnalyzer(system, nonlinear_params)

    def add_nonlinear_interactions(self, nonlinear_params: Dict[str, Any]) -> None:
        self.nonlinear_strength = nonlinear_params.get(
            "strength", self.nonlinear_strength
        )
        self.nonlinear_order = nonlinear_params.get("order", self.nonlinear_order)
        self.nonlinear_type = nonlinear_params.get("type", self.nonlinear_type)
        if hasattr(self.system, "add_potential"):
            self.system.add_potential(self._nonlinear_potential)
        if hasattr(self.system, "add_force"):
            self.system.add_force(self._nonlinear_force)

    def _nonlinear_potential(self, psi: np.ndarray) -> np.ndarray:
        if self.nonlinear_type == "cubic":
            return self.nonlinear_strength * np.abs(psi) ** 3
        if self.nonlinear_type == "quartic":
            return self.nonlinear_strength * np.abs(psi) ** 4
        if self.nonlinear_type == "sine_gordon":
            return self.nonlinear_strength * (1 - np.cos(psi))
        raise ValueError(f"Unknown nonlinear type: {self.nonlinear_type}")

    def _nonlinear_force(self, psi: np.ndarray) -> np.ndarray:
        if self.nonlinear_type == "cubic":
            return -3 * self.nonlinear_strength * np.abs(psi) * np.sign(psi)
        if self.nonlinear_type == "quartic":
            return -4 * self.nonlinear_strength * np.abs(psi) ** 2 * np.sign(psi)
        if self.nonlinear_type == "sine_gordon":
            return -self.nonlinear_strength * np.sin(psi)
        raise ValueError(f"Unknown nonlinear type: {self.nonlinear_type}")

    def find_nonlinear_modes(self) -> Dict[str, Any]:
        return self.mode_analyzer.find_nonlinear_modes()

    def find_soliton_solutions(self) -> Dict[str, Any]:
        return self.soliton_analyzer.find_soliton_solutions()

    def analyze_nonlinear_strength(self, field: np.ndarray) -> Dict[str, Any]:
        return self.basic_effects.analyze_nonlinear_strength(field)

    def compute_nonlinear_energy(self, field: np.ndarray) -> float:
        return self.basic_effects.compute_nonlinear_energy(field)

    def compute_nonlinear_force(self, field: np.ndarray) -> np.ndarray:
        return self.basic_effects.compute_nonlinear_force(field)

    def analyze_nonlinear_stability(self) -> Dict[str, Any]:
        return self.mode_analyzer._analyze_nonlinear_stability()

    def find_bifurcation_points(self) -> List[Dict[str, Any]]:
        return self.mode_analyzer._find_bifurcation_points()

    def compute_nonlinear_corrections(
        self, linear_modes: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self.mode_analyzer._compute_nonlinear_corrections(linear_modes)

    def analyze_soliton_stability(
        self, soliton_profiles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return self.soliton_analyzer._analyze_soliton_stability(soliton_profiles)

    def analyze_soliton_interactions(
        self, soliton_profiles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return self.soliton_analyzer._analyze_soliton_interactions(soliton_profiles)

    def compute_soliton_energies(
        self, soliton_profiles: List[Dict[str, Any]]
    ) -> List[float]:
        return self.soliton_analyzer._compute_soliton_energies(soliton_profiles)

    def validate_nonlinear_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        basic_validation = self._validate_basic_effects(
            results.get("basic_effects", {})
        )
        soliton_validation = self._validate_soliton_analysis(
            results.get("soliton_analysis", {})
        )
        mode_validation = self._validate_mode_analysis(results.get("mode_analysis", {}))
        overall_validation = self._calculate_overall_validation(
            basic_validation, soliton_validation, mode_validation
        )
        return {
            "basic_validation": basic_validation,
            "soliton_validation": soliton_validation,
            "mode_validation": mode_validation,
            "overall_validation": overall_validation,
            "validation_complete": True,
        }

    def _validate_basic_effects(self, basic_effects: Dict[str, Any]) -> Dict[str, Any]:
        is_present = len(basic_effects) > 0
        quality_metrics = basic_effects.get("quality_metrics", {})
        quality_score = quality_metrics.get("overall_quality", 0.0)
        return {
            "is_present": is_present,
            "quality_score": quality_score,
            "validation_passed": is_present and quality_score > 0.7,
        }

    def _validate_soliton_analysis(
        self, soliton_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        is_present = len(soliton_analysis) > 0
        profiles = soliton_analysis.get("profiles", [])
        num_profiles = len(profiles)
        stability = soliton_analysis.get("stability", {})
        stability_score = stability.get("overall_stability", 0.0)
        return {
            "is_present": is_present,
            "num_profiles": num_profiles,
            "stability_score": stability_score,
            "validation_passed": is_present
            and num_profiles > 0
            and stability_score > 0.5,
        }

    def _validate_mode_analysis(self, mode_analysis: Dict[str, Any]) -> Dict[str, Any]:
        is_present = len(mode_analysis) > 0
        frequencies = mode_analysis.get("nonlinear_frequencies", [])
        num_frequencies = len(frequencies)
        stability = mode_analysis.get("stability", {})
        stability_score = stability.get("overall_stability", 0.0)
        return {
            "is_present": is_present,
            "num_frequencies": num_frequencies,
            "stability_score": stability_score,
            "validation_passed": is_present
            and num_frequencies > 0
            and stability_score > 0.5,
        }

    def _calculate_overall_validation(
        self,
        basic_validation: Dict[str, Any],
        soliton_validation: Dict[str, Any],
        mode_validation: Dict[str, Any],
    ) -> Dict[str, Any]:
        overall_quality = np.mean(
            [
                basic_validation["quality_score"],
                soliton_validation["stability_score"],
                mode_validation["stability_score"],
            ]
        )
        overall_passed = all(
            [
                basic_validation["validation_passed"],
                soliton_validation["validation_passed"],
                mode_validation["validation_passed"],
            ]
        )
        return {
            "overall_quality": overall_quality,
            "overall_passed": overall_passed,
            "validation_summary": {
                "basic_validation": basic_validation["validation_passed"],
                "soliton_validation": soliton_validation["validation_passed"],
                "mode_validation": mode_validation["validation_passed"],
            },
        }
