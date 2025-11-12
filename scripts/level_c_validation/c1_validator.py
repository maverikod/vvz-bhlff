"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

C1 test acceptance criteria validator.

Physical Meaning:
    Validates C1 test results against acceptance criteria:
    1. At η=0: no peaks ≥ 8 dB
    2. At η≥0.1: ≥1 peak exists
    3. Maximum localization between core and wall
    4. Passivity P_Ω(ω)≥0 for all ω
    5. Convergence: ω_n change ≤3%, Q_n change ≤10% when N increases
"""

import numpy as np
from typing import Dict, Any, List
import logging

from .data_structures import C1AcceptanceResults


class C1AcceptanceValidator:
    """
    C1 test acceptance criteria validator.

    Physical Meaning:
        Validates C1 test results according to acceptance criteria
        for single wall boundary effects and resonance mode analysis.

    Mathematical Foundation:
        Validates:
        - Peak detection: peaks with prominence ≥ 8 dB
        - Passivity: P_Ω(ω) = ∫_V [ν|(-Δ)^(β/2)â|² + λ|â|² + Re Γ_mem|â|²] dV ≥ 0
        - Convergence: |ω_n(N₂) - ω_n(N₁)|/ω_n(N₁) ≤ 0.03
        - Convergence: |Q_n(N₂) - Q_n(N₁)|/Q_n(N₁) ≤ 0.10
    """

    def __init__(self):
        """Initialize C1 acceptance validator."""
        self.logger = logging.getLogger(__name__)

    def validate(self, c1_results: Dict[str, Any]) -> C1AcceptanceResults:
        """
        Validate C1 test results against acceptance criteria.

        Args:
            c1_results (Dict[str, Any]): C1 test results.

        Returns:
            C1AcceptanceResults: Validation results.
        """
        failures = []

        # Extract data
        boundary_analysis = c1_results.get("boundary_analysis", {})
        contrast_results = boundary_analysis.get("contrast_results", {})
        convergence_results = boundary_analysis.get("convergence_results", {})

        # 1. Check no peaks at zero contrast
        zero_contrast_result = contrast_results.get("0.0", {})
        zero_contrast_peaks = zero_contrast_result.get("peaks", [])
        zero_contrast_prominence_dB = [
            peak.get("prominence_dB", 0.0) for peak in zero_contrast_peaks
        ]
        no_peaks_at_zero = all(p < 8.0 for p in zero_contrast_prominence_dB)

        if not no_peaks_at_zero:
            failures.append(
                f"C1.1: Found peaks ≥ 8 dB at η=0: {[p for p in zero_contrast_prominence_dB if p >= 8.0]}"
            )

        # 2. Check resonance birth threshold
        resonance_birth_threshold = None
        for contrast_str in ["0.1", "0.2", "0.3"]:
            contrast_value = float(contrast_str)
            contrast_result = contrast_results.get(contrast_str, {})
            peaks = contrast_result.get("peaks", [])
            prominence_dB = [peak.get("prominence_dB", 0.0) for peak in peaks]

            if any(p >= 8.0 for p in prominence_dB):
                resonance_birth_threshold = contrast_value
                break

        if resonance_birth_threshold is None:
            failures.append("C1.2: No peaks ≥ 8 dB found at η≥0.1")
            resonance_birth_threshold = float("inf")
        elif resonance_birth_threshold > 0.1:
            failures.append(
                f"C1.2: Resonance birth threshold {resonance_birth_threshold} > 0.1"
            )

        # 3. Check localization (maximum between core and wall)
        localization_correct = True
        for contrast_str, contrast_result in contrast_results.items():
            if float(contrast_str) < 0.1:
                continue

            radial_profile = contrast_result.get("radial_profile", {})
            if radial_profile is None:
                continue

            r_values = radial_profile.get("r", [])
            A_values = radial_profile.get("A", [])
            R_ref = boundary_analysis.get("R_ref", 0.0)
            R_layer = boundary_analysis.get("R_layer", float("inf"))

            if len(r_values) == 0 or len(A_values) == 0:
                continue

            # Find maximum between R_ref and R_layer
            max_idx = np.argmax(A_values)
            max_r = r_values[max_idx] if max_idx < len(r_values) else 0.0

            if not (R_ref < max_r < R_layer):
                localization_correct = False
                failures.append(
                    f"C1.3: Maximum at r={max_r} not between R_ref={R_ref} and R_layer={R_layer}"
                )

        # 4. Check passivity
        passivity_check = True
        passivity_results = boundary_analysis.get("passivity_results", {})
        P_omega_values = passivity_results.get("P_omega", [])

        if len(P_omega_values) > 0:
            min_P = np.min(P_omega_values)
            if min_P < 0.0:
                passivity_check = False
                failures.append(f"C1.4: Passivity violation: min P_Ω = {min_P} < 0")

        # 5. Check convergence
        convergence_omega = True
        convergence_q = True

        if convergence_results:
            omega_N256 = convergence_results.get("omega_N256", [])
            omega_N384 = convergence_results.get("omega_N384", [])
            Q_N256 = convergence_results.get("Q_N256", [])
            Q_N384 = convergence_results.get("Q_N384", [])

            # Check omega convergence
            if len(omega_N256) > 0 and len(omega_N384) > 0:
                for i, omega_256 in enumerate(omega_N256):
                    if i < len(omega_N384):
                        omega_384 = omega_N384[i]
                        if omega_256 > 0:
                            error = abs(omega_384 - omega_256) / omega_256
                            if error > 0.03:
                                convergence_omega = False
                                failures.append(
                                    f"C1.5: Omega convergence violation: error={error:.4f} > 0.03"
                                )

            # Check Q convergence
            if len(Q_N256) > 0 and len(Q_N384) > 0:
                for i, Q_256 in enumerate(Q_N256):
                    if i < len(Q_N384):
                        Q_384 = Q_N384[i]
                        if Q_256 > 0:
                            error = abs(Q_384 - Q_256) / Q_256
                            if error > 0.10:
                                convergence_q = False
                                failures.append(
                                    f"C1.5: Q convergence violation: error={error:.4f} > 0.10"
                                )

        all_passed = (
            no_peaks_at_zero
            and (resonance_birth_threshold <= 0.1)
            and localization_correct
            and passivity_check
            and convergence_omega
            and convergence_q
        )

        return C1AcceptanceResults(
            no_peaks_at_zero_contrast=no_peaks_at_zero,
            resonance_birth_threshold=resonance_birth_threshold,
            localization_correct=localization_correct,
            passivity_check=passivity_check,
            convergence_omega=convergence_omega,
            convergence_q=convergence_q,
            all_passed=all_passed,
            failures=failures,
        )

