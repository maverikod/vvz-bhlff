"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

C3 test acceptance criteria validator.

Physical Meaning:
    Validates C3 test results against acceptance criteria:
    1. At γ=0: v_cell ≈ Δω/|Δk|, deviation ≤ 10%
    2. At γ≥γ*: v_cell ≤ 10⁻³ L/T_0 (frozen)
    3. Jaccard index ≥ 0.95 on long windows
"""

import numpy as np
from typing import Dict, Any
import logging

from .data_structures import C3AcceptanceResults


class C3AcceptanceValidator:
    """
    C3 test acceptance criteria validator.

    Physical Meaning:
        Validates C3 test results according to acceptance criteria
        for quench memory and pinning effects analysis.

    Mathematical Foundation:
        Validates:
        - Drift velocity: v_cell = Δx_max / Δt from cross-correlation
        - Freezing threshold: v_cell(γ*) ≤ 10⁻³ L/T_0
        - Jaccard index: J = |A ∩ B| / |A ∪ B| ≥ 0.95
    """

    def __init__(self):
        """Initialize C3 acceptance validator."""
        self.logger = logging.getLogger(__name__)

    def validate(self, c3_results: Dict[str, Any]) -> C3AcceptanceResults:
        """
        Validate C3 test results against acceptance criteria.

        Args:
            c3_results (Dict[str, Any]): C3 test results.

        Returns:
            C3AcceptanceResults: Validation results.
        """
        failures = []

        # Extract data
        memory_analysis = c3_results.get("memory_analysis", {})
        gamma_sweep_results = memory_analysis.get("gamma_sweep_results", {})

        # 1. Check drift velocity at zero memory (γ=0)
        zero_gamma_result = gamma_sweep_results.get("0.0", {})
        drift_velocity_at_zero = zero_gamma_result.get("v_cell", 0.0)
        delta_omega = memory_analysis.get("delta_omega", 0.0)
        delta_k = memory_analysis.get("delta_k", 0.0)

        if delta_k > 0:
            predicted_velocity = delta_omega / abs(delta_k)
            drift_velocity_error_at_zero = (
                abs(drift_velocity_at_zero - predicted_velocity)
                / predicted_velocity
                if predicted_velocity > 0
                else float("inf")
            )

            if drift_velocity_error_at_zero > 0.10:
                failures.append(
                    f"C3.1: Drift velocity error at γ=0: {drift_velocity_error_at_zero:.4f} > 0.10"
                )
        else:
            drift_velocity_error_at_zero = float("inf")
            failures.append("C3.1: delta_k not available for validation")

        # 2. Check freezing threshold
        domain_params = memory_analysis.get("domain", {})
        L = domain_params.get("L", 1.0)
        T0 = memory_analysis.get("T0", 1.0)
        freezing_threshold = 1e-3 * L / T0

        freezing_threshold_gamma_star = None
        drift_velocity_at_threshold = None

        # Find gamma star (first gamma where v_cell ≤ freezing_threshold)
        gamma_values = sorted(
            [float(g) for g in gamma_sweep_results.keys() if g != "0.0"]
        )
        for gamma_str in gamma_values:
            gamma = float(gamma_str) if isinstance(gamma_str, str) else gamma_str
            gamma_result = gamma_sweep_results.get(str(gamma), {})
            v_cell = gamma_result.get("v_cell", float("inf"))

            if v_cell <= freezing_threshold:
                freezing_threshold_gamma_star = gamma
                drift_velocity_at_threshold = v_cell
                break

        if freezing_threshold_gamma_star is None:
            failures.append(
                f"C3.2: No freezing threshold found: v_cell never ≤ {freezing_threshold:.6f}"
            )
            freezing_threshold_gamma_star = float("inf")

        # 3. Check Jaccard index
        jaccard_index = memory_analysis.get("jaccard_index", 0.0)

        if jaccard_index < 0.95:
            failures.append(
                f"C3.3: Jaccard index {jaccard_index:.4f} < 0.95"
            )

        all_passed = (
            drift_velocity_error_at_zero <= 0.10
            and freezing_threshold_gamma_star != float("inf")
            and drift_velocity_at_threshold is not None
            and drift_velocity_at_threshold <= freezing_threshold
            and jaccard_index >= 0.95
        )

        return C3AcceptanceResults(
            drift_velocity_at_zero_memory=drift_velocity_at_zero,
            drift_velocity_error_at_zero=drift_velocity_error_at_zero,
            freezing_threshold_gamma_star=freezing_threshold_gamma_star,
            drift_velocity_at_threshold=drift_velocity_at_threshold,
            jaccard_index=jaccard_index,
            all_passed=all_passed,
            failures=failures,
        )

