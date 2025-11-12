"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Main Level C acceptance criteria validator orchestrator.

Physical Meaning:
    Orchestrates validation of all Level C test results against
    acceptance criteria from the experiment plan document.
"""

from typing import Dict, Any, Optional
import logging

from .c1_validator import C1AcceptanceValidator
from .c2_validator import C2AcceptanceValidator
from .c3_validator import C3AcceptanceValidator
from .c4_validator import C4AcceptanceValidator


class LevelCAcceptanceValidator:
    """
    Level C acceptance criteria validator orchestrator.

    Physical Meaning:
        Orchestrates validation of all Level C test results against
        acceptance criteria specified in the Level C experiment plan document.

    Mathematical Foundation:
        Coordinates validation of:
        - C1: Peak detection, passivity, convergence criteria
        - C2: ABCD model validation, error thresholds
        - C3: Memory effects, drift velocity, Jaccard index
        - C4: Beating analysis, suppression factors
    """

    def __init__(self):
        """Initialize Level C acceptance validator."""
        self.logger = logging.getLogger(__name__)
        self.c1_validator = C1AcceptanceValidator()
        self.c2_validator = C2AcceptanceValidator()
        self.c3_validator = C3AcceptanceValidator()
        self.c4_validator = C4AcceptanceValidator()

    def validate_all(
        self,
        c1_results: Optional[Dict[str, Any]] = None,
        c2_results: Optional[Dict[str, Any]] = None,
        c3_results: Optional[Dict[str, Any]] = None,
        c4_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validate all Level C test results.

        Physical Meaning:
            Validates all Level C test results against acceptance criteria
            and generates a comprehensive validation report.

        Args:
            c1_results (Optional[Dict[str, Any]]): C1 test results.
            c2_results (Optional[Dict[str, Any]]): C2 test results.
            c3_results (Optional[Dict[str, Any]]): C3 test results.
            c4_results (Optional[Dict[str, Any]]): C4 test results.

        Returns:
            Dict[str, Any]: Complete validation results.
        """
        validation_results = {}

        if c1_results:
            validation_results["c1"] = self.c1_validator.validate(c1_results)
            self.logger.info(
                f"C1 validation: {'PASS' if validation_results['c1'].all_passed else 'FAIL'}"
            )

        if c2_results:
            validation_results["c2"] = self.c2_validator.validate(c2_results)
            self.logger.info(
                f"C2 validation: {'PASS' if validation_results['c2'].all_passed else 'FAIL'}"
            )

        if c3_results:
            validation_results["c3"] = self.c3_validator.validate(c3_results)
            self.logger.info(
                f"C3 validation: {'PASS' if validation_results['c3'].all_passed else 'FAIL'}"
            )

        if c4_results:
            validation_results["c4"] = self.c4_validator.validate(c4_results)
            self.logger.info(
                f"C4 validation: {'PASS' if validation_results['c4'].all_passed else 'FAIL'}"
            )

        # Overall validation
        all_tests_passed = all(
            result.all_passed
            for result in validation_results.values()
            if hasattr(result, "all_passed")
        )

        validation_results["overall"] = {
            "all_tests_passed": all_tests_passed,
            "validation_complete": True,
        }

        return validation_results

