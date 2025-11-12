"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Level C acceptance criteria validation CLI script.

This script validates that all Level C test results meet the acceptance
criteria specified in 7d-33-БВП_план_численных_экспериментов_C.md.

Example:
    >>> python scripts/validate_level_c_acceptance_criteria.py
    >>> python scripts/validate_level_c_acceptance_criteria.py --c1-results output/C1/results.json
"""

import json
import numpy as np
import argparse
import logging
import sys
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from level_c_validation.main_validator import LevelCAcceptanceValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Validate Level C acceptance criteria"
    )
    parser.add_argument(
        "--c1-results",
        type=str,
        help="Path to C1 test results JSON file",
    )
    parser.add_argument(
        "--c2-results",
        type=str,
        help="Path to C2 test results JSON file",
    )
    parser.add_argument(
        "--c3-results",
        type=str,
        help="Path to C3 test results JSON file",
    )
    parser.add_argument(
        "--c4-results",
        type=str,
        help="Path to C4 test results JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="level_c_validation_report.json",
        help="Path to output validation report JSON file",
    )

    args = parser.parse_args()

    validator = LevelCAcceptanceValidator()

    # Load results
    c1_results = None
    c2_results = None
    c3_results = None
    c4_results = None

    if args.c1_results:
        with open(args.c1_results, "r") as f:
            c1_results = json.load(f)

    if args.c2_results:
        with open(args.c2_results, "r") as f:
            c2_results = json.load(f)

    if args.c3_results:
        with open(args.c3_results, "r") as f:
            c3_results = json.load(f)

    if args.c4_results:
        with open(args.c4_results, "r") as f:
            c4_results = json.load(f)

    # Validate
    validation_results = validator.validate_all(
        c1_results, c2_results, c3_results, c4_results
    )

    # Print summary
    print("\n" + "=" * 80)
    print("Level C Acceptance Criteria Validation Report")
    print("=" * 80)

    for test_name, result in validation_results.items():
        if test_name == "overall":
            continue

        print(f"\n{test_name.upper()} Test:")
        if hasattr(result, "all_passed"):
            status = "PASS" if result.all_passed else "FAIL"
            print(f"  Status: {status}")

            if hasattr(result, "failures") and result.failures:
                print("  Failures:")
                for failure in result.failures:
                    print(f"    - {failure}")

    print("\n" + "=" * 80)
    overall_status = (
        "PASS"
        if validation_results["overall"]["all_tests_passed"]
        else "FAIL"
    )
    print(f"Overall Status: {overall_status}")
    print("=" * 80 + "\n")

    # Save results
    # Convert dataclass results to dict for JSON serialization
    output_results = {}
    for key, value in validation_results.items():
        if hasattr(value, "__dict__"):
            output_results[key] = {
                k: v
                for k, v in value.__dict__.items()
                if not isinstance(v, np.ndarray)
                or isinstance(v, (list, float, int, bool, str))
            }
            if hasattr(value, "failures"):
                output_results[key]["failures"] = value.failures
        else:
            output_results[key] = value

    with open(args.output, "w") as f:
        json.dump(output_results, f, indent=2)

    print(f"Validation report saved to: {args.output}")

    # Exit with error code if validation failed
    if not validation_results["overall"]["all_tests_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
