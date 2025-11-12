"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

C2 test acceptance criteria validator.

Physical Meaning:
    Validates C2 test results against acceptance criteria:
    1. ≥3 peaks in Y(ω) with prominence ≥ 8 dB
    2. ABCD prediction errors ≤10% overall, ≤5% at peaks
    3. Frequency errors |ω^ABCD - ω^sim|/ω^sim ≤ 5%
    4. Quality factor errors |Q^ABCD - Q^sim|/Q^sim ≤ 10%
    5. Passivity check
"""

import numpy as np
from typing import Dict, Any
import logging

from .data_structures import C2AcceptanceResults


class C2AcceptanceValidator:
    """
    C2 test acceptance criteria validator.

    Physical Meaning:
        Validates C2 test results according to acceptance criteria
        for resonator chain analysis with ABCD model validation.

    Mathematical Foundation:
        Validates:
        - Peak detection: count peaks with prominence ≥ 8 dB
        - ABCD errors: |A_sim - A_ABCD|/|A_sim| ≤ 0.10 (overall), ≤ 0.05 (at peaks)
        - Frequency errors: |ω_ABCD - ω_sim|/ω_sim ≤ 0.05
        - Q errors: |Q_ABCD - Q_sim|/Q_sim ≤ 0.10
    """

    def __init__(self):
        """Initialize C2 acceptance validator."""
        self.logger = logging.getLogger(__name__)

    def validate(self, c2_results: Dict[str, Any]) -> C2AcceptanceResults:
        """
        Validate C2 test results against acceptance criteria.

        Args:
            c2_results (Dict[str, Any]): C2 test results.

        Returns:
            C2AcceptanceResults: Validation results.
        """
        failures = []

        # Extract data
        abcd_analysis = c2_results.get("abcd_analysis", {})
        comparison_results = abcd_analysis.get("comparison_with_numerical", {})

        # 1. Check minimum peaks count
        peaks = abcd_analysis.get("peaks", [])
        prominence_dB = [peak.get("prominence_dB", 0.0) for peak in peaks]
        peaks_above_threshold = [p for p in prominence_dB if p >= 8.0]
        minimum_peaks_count = len(peaks_above_threshold)

        if minimum_peaks_count < 3:
            failures.append(
                f"C2.1: Only {minimum_peaks_count} peaks ≥ 8 dB, need ≥3"
            )

        # 2. Check ABCD prediction errors
        admittance_errors = comparison_results.get("admittance_errors", [])
        max_admittance_error = (
            max(admittance_errors) if admittance_errors else 0.0
        )
        abcd_errors_overall = max_admittance_error

        if abcd_errors_overall > 0.10:
            failures.append(
                f"C2.2: ABCD overall error {abcd_errors_overall:.4f} > 0.10"
            )

        # Check errors at peaks
        peak_indices = [i for i, p in enumerate(prominence_dB) if p >= 8.0]
        peak_errors = [
            admittance_errors[i]
            for i in peak_indices
            if i < len(admittance_errors)
        ]
        abcd_errors_at_peaks = max(peak_errors) if peak_errors else 0.0

        if abcd_errors_at_peaks > 0.05:
            failures.append(
                f"C2.2: ABCD peak error {abcd_errors_at_peaks:.4f} > 0.05"
            )

        # 3. Check frequency errors
        frequency_errors = comparison_results.get("frequency_errors", [])
        max_frequency_error = (
            max(frequency_errors) if frequency_errors else 0.0
        )

        if max_frequency_error > 0.05:
            failures.append(
                f"C2.3: Max frequency error {max_frequency_error:.4f} > 0.05"
            )

        # 4. Check quality factor errors
        quality_errors = comparison_results.get("quality_errors", [])
        max_quality_error = max(quality_errors) if quality_errors else 0.0

        if max_quality_error > 0.10:
            failures.append(
                f"C2.4: Max Q error {max_quality_error:.4f} > 0.10"
            )

        # 5. Check passivity
        passivity_results = abcd_analysis.get("passivity_results", {})
        P_omega_values = passivity_results.get("P_omega", [])

        passivity_check = True
        if len(P_omega_values) > 0:
            min_P = np.min(P_omega_values)
            if min_P < 0.0:
                passivity_check = False
                failures.append(f"C2.5: Passivity violation: min P_Ω = {min_P} < 0")

        all_passed = (
            minimum_peaks_count >= 3
            and abcd_errors_overall <= 0.10
            and abcd_errors_at_peaks <= 0.05
            and max_frequency_error <= 0.05
            and max_quality_error <= 0.10
            and passivity_check
        )

        return C2AcceptanceResults(
            minimum_peaks_count=minimum_peaks_count,
            abcd_errors_overall=abcd_errors_overall,
            abcd_errors_at_peaks=abcd_errors_at_peaks,
            frequency_errors=frequency_errors,
            quality_factor_errors=quality_errors,
            passivity_check=passivity_check,
            all_passed=all_passed,
            failures=failures,
        )

