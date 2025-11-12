"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

C4 test acceptance criteria validator.

Physical Meaning:
    Validates C4 test results against acceptance criteria:
    1. Without pinning: |v_cell^num - v_cell^pred|/v_cell^pred ≤ 10%
    2. With pinning: v_cell^num/v_cell^pred ≤ 0.1 (suppression ≥10×)
"""

from typing import Dict, Any
import logging

from .data_structures import C4AcceptanceResults


class C4AcceptanceValidator:
    """
    C4 test acceptance criteria validator.

    Physical Meaning:
        Validates C4 test results according to acceptance criteria
        for mode beating and drift velocity analysis.

    Mathematical Foundation:
        Validates:
        - Beating error: |v_num - v_pred|/v_pred ≤ 0.10 (no pinning)
        - Suppression: v_num/v_pred ≤ 0.1 (with pinning, i.e., suppression ≥10×)
    """

    def __init__(self):
        """Initialize C4 acceptance validator."""
        self.logger = logging.getLogger(__name__)

    def validate(self, c4_results: Dict[str, Any]) -> C4AcceptanceResults:
        """
        Validate C4 test results against acceptance criteria.

        Args:
            c4_results (Dict[str, Any]): C4 test results.

        Returns:
            C4AcceptanceResults: Validation results.
        """
        failures = []

        # Extract data
        beating_analysis = c4_results.get("beating_analysis", {})
        beating_results = beating_analysis.get("beating_results", {})

        # 1. Check beating error without pinning
        background_result = beating_results.get("background", {})
        error_analysis = background_result.get("error_analysis", {})
        beating_error_without_pinning = error_analysis.get(
            "background_error", float("inf")
        )

        if beating_error_without_pinning > 0.10:
            failures.append(
                f"C4.1: Beating error without pinning {beating_error_without_pinning:.4f} > 0.10"
            )

        # 2. Check suppression factor with pinning
        pinned_result = beating_results.get("pinned", {})
        pinned_error_analysis = pinned_result.get("error_analysis", {})
        suppression_factor_with_pinning = pinned_error_analysis.get(
            "suppression_factor", 1.0
        )

        if suppression_factor_with_pinning > 0.1:
            failures.append(
                f"C4.2: Suppression factor {suppression_factor_with_pinning:.4f} > 0.1 (suppression < 10×)"
            )

        all_passed = (
            beating_error_without_pinning <= 0.10
            and suppression_factor_with_pinning <= 0.1
        )

        return C4AcceptanceResults(
            beating_error_without_pinning=beating_error_without_pinning,
            suppression_factor_with_pinning=suppression_factor_with_pinning,
            all_passed=all_passed,
            failures=failures,
        )

