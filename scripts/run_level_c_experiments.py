"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Level C experiment runner script.

This script runs Level C tests (C1-C4) and validates results against
acceptance criteria from 7d-33-БВП_план_численных_экспериментов_C.md.

Physical Meaning:
    Executes Level C integration tests and validates results:
    - C1: Single wall boundary effects and resonance mode analysis
    - C2: Resonator chain analysis with ABCD model validation
    - C3: Quench memory and pinning effects analysis
    - C4: Mode beating and drift velocity analysis

Example:
    >>> python scripts/run_level_c_experiments.py
    >>> python scripts/run_level_c_experiments.py --validate-only
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LevelCExperimentRunner:
    """
    Level C experiment runner.

    Physical Meaning:
        Runs Level C integration tests and validates results
        against acceptance criteria from the experiment plan.
    """

    def __init__(self, output_dir: Path = Path("output/level_c_experiments")):
        """
        Initialize Level C experiment runner.

        Args:
            output_dir (Path): Output directory for test results.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.results = {}

    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all Level C tests (C1-C4).

        Physical Meaning:
            Executes all Level C tests through the integration system
            and saves results for validation.

        Returns:
            Dict[str, Any]: Comprehensive test results.
        """
        try:
            from bhlff.core.domain import Domain
            from bhlff.core.bvp import BVPCore
            from bhlff.models.level_c.level_c_integration import (
                LevelCIntegration,
            )
            from bhlff.models.level_c.level_c_integration_config import (
                TestConfiguration,
            )
        except ImportError as e:
            self.logger.error(f"Failed to import Level C modules: {e}")
            return {
                "status": "ERROR",
                "error": str(e),
                "message": "Level C modules not available",
            }

        self.logger.info("Starting Level C experiments")

        # Create domain and BVP core
        try:
            # Create 7D domain for Level C tests with block processing
            # Level C works with 3D spatial fields, but BVP requires 7D domain
            # Block processing should handle memory efficiently
            L = 32.0 * 3.14159  # Normal spatial size for Level C tests
            N = 256  # Normal resolution for Level C tests (block processing handles memory)
            domain = Domain(
                L=L,
                N=N,
                N_phi=8,  # Phase dimension (block processing handles memory)
                N_t=16,  # Temporal dimension (block processing handles memory)
                T=1.0,  # Temporal size
                dimensions=7,  # 7D domain required by BVP
            )

            bvp_config = {
                "carrier_frequency": 1.85e43,
                "envelope_equation": {
                    "kappa_0": 1.0,
                    "kappa_2": 0.1,
                    "chi_prime": 1.0,
                    "chi_double_prime_0": 0.01,
                    "k0_squared": 1.0,
                },
            }

            bvp_core = BVPCore(domain, bvp_config)
            integration = LevelCIntegration(bvp_core)

            # Create test configuration
            # Level C works with 3D spatial fields (extracted from 7D domain)
            domain_params = {
                "L": L,  # Use same L as domain
                "N": N,  # Use same N as domain
                "dimensions": 3,  # Level C works with 3D spatial fields
            }

            test_params = {
                "boundary_params": {
                    "contrast_values": [0.0, 0.05, 0.1, 0.2, 0.3],
                    "frequency_range": (0.05, 5.0),
                    "resonance_threshold": 8.0,
                },
                "abcd_params": {
                    "use_two_runs": True,
                    "epsilon_dipole": 0.08,
                },
                "memory_params": {
                    "gamma_list": [0.0, 0.2, 0.4, 0.6, 0.8],
                    "tau_list": [0.5, 1.0, 2.0],
                },
                "beating_params": {
                    "delta_omega_ratios": [0.02, 0.05],
                },
                "time_params": {
                    "dt": 5e-3,
                    "T": 400.0,
                    "avg_window": 0.8,
                },
            }

            test_config = integration.create_test_configuration(
                domain_params, test_params
            )

            # Run tests
            self.logger.info("Running Level C integration tests")
            results = integration.run_all_tests(test_config)

            # Save results
            self._save_results(results)

            return {
                "status": "SUCCESS",
                "results": {
                    "c1_complete": results.c1_results.get("test_complete", False),
                    "c2_complete": results.c2_results.get("test_complete", False),
                    "c3_complete": results.c3_results.get("test_complete", False),
                    "c4_complete": results.c4_results.get("test_complete", False),
                    "all_tests_complete": results.all_tests_complete,
                },
                "validation": results.overall_validation,
                "output_dir": str(self.output_dir),
            }

        except Exception as e:
            self.logger.error(f"Error running Level C experiments: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "error": str(e),
                "message": "Failed to run Level C experiments",
            }

    def _save_results(self, results) -> None:
        """
        Save test results to JSON files.

        Args:
            results: Level C test results.
        """
        # Save individual test results
        for test_name in ["c1", "c2", "c3", "c4"]:
            test_results = getattr(results, f"{test_name}_results", {})
            output_file = self.output_dir / f"{test_name.upper()}_results.json"
            with open(output_file, "w") as f:
                json.dump(test_results, f, indent=2, default=str)
            self.logger.info(f"Saved {test_name.upper()} results to {output_file}")

        # Save overall validation
        validation_file = self.output_dir / "overall_validation.json"
        with open(validation_file, "w") as f:
            json.dump(results.overall_validation, f, indent=2, default=str)
        self.logger.info(f"Saved validation results to {validation_file}")

    def validate_results(self) -> Dict[str, Any]:
        """
        Validate test results against acceptance criteria.

        Returns:
            Dict[str, Any]: Validation results.
        """
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from level_c_validation.main_validator import LevelCAcceptanceValidator

            validator = LevelCAcceptanceValidator()

            # Load results
            c1_results = None
            c2_results = None
            c3_results = None
            c4_results = None

            c1_file = self.output_dir / "C1_results.json"
            c2_file = self.output_dir / "C2_results.json"
            c3_file = self.output_dir / "C3_results.json"
            c4_file = self.output_dir / "C4_results.json"

            if c1_file.exists():
                with open(c1_file, "r") as f:
                    c1_results = json.load(f)

            if c2_file.exists():
                with open(c2_file, "r") as f:
                    c2_results = json.load(f)

            if c3_file.exists():
                with open(c3_file, "r") as f:
                    c3_results = json.load(f)

            if c4_file.exists():
                with open(c4_file, "r") as f:
                    c4_results = json.load(f)

            # Validate
            validation_results = validator.validate_all(
                c1_results, c2_results, c3_results, c4_results
            )

            # Save validation report
            report_file = self.output_dir / "acceptance_criteria_validation.json"
            validation_dict = {}
            for key, value in validation_results.items():
                if hasattr(value, "__dict__"):
                    validation_dict[key] = {
                        k: v
                        for k, v in value.__dict__.items()
                        if isinstance(v, (list, float, int, bool, str))
                    }
                    if hasattr(value, "failures"):
                        validation_dict[key]["failures"] = value.failures
                else:
                    validation_dict[key] = value

            with open(report_file, "w") as f:
                json.dump(validation_dict, f, indent=2)

            self.logger.info(
                f"Validation report saved to {report_file}"
            )

            return validation_results

        except Exception as e:
            self.logger.error(f"Error validating results: {e}", exc_info=True)
            return {"status": "ERROR", "error": str(e)}


def main():
    """Main function for command-line execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Level C experiments")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing results, don't run tests",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/level_c_experiments"),
        help="Output directory for test results",
    )

    args = parser.parse_args()

    runner = LevelCExperimentRunner(output_dir=args.output_dir)

    if args.validate_only:
        print("Validating existing Level C test results...")
        validation_results = runner.validate_results()
        print("\nValidation complete.")
        if validation_results.get("overall", {}).get("all_tests_passed", False):
            print("✓ All tests passed acceptance criteria")
        else:
            print("✗ Some tests failed acceptance criteria")
            for test_name, result in validation_results.items():
                if test_name != "overall" and hasattr(result, "failures"):
                    if result.failures:
                        print(f"\n{test_name.upper()} failures:")
                        for failure in result.failures:
                            print(f"  - {failure}")
        return 0

    print("Running Level C experiments...")
    results = runner.run_all_tests()

    if results["status"] == "SUCCESS":
        print("\n✓ Level C experiments completed successfully")
        print(f"Results saved to: {results['output_dir']}")

        # Validate results
        print("\nValidating results against acceptance criteria...")
        validation_results = runner.validate_results()

        if validation_results.get("overall", {}).get("all_tests_passed", False):
            print("✓ All tests passed acceptance criteria")
        else:
            print("✗ Some tests failed acceptance criteria")
        return 0
    else:
        print(f"\n✗ Level C experiments failed: {results.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

