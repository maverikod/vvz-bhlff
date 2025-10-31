"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Level A Numerical Experiments Runner (7d-31 Specification).

This script executes all Level A validation tests according to the
specification in 7d-31-БВП_план_численных_экспери_'ментов_A.md and
generates a comprehensive acceptance report.

Physical Meaning:
    Validates the numerical core of the 7D BVP Framework by running
    comprehensive tests covering spectral solvers, time integrators,
    energy balance, and dimensional invariance.

Mathematical Foundation:
    Tests validate the spectral solution formula â = ŝ / (μ|k|^(2β) + λ),
    time-dependent evolution ∂t a + ν(-Δ)^β a + λa = s, and dimensional
    scaling invariance of the framework.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import platform
import importlib.metadata

import pytest


class LevelAExperimentRunner:
    """
    Runner for Level A numerical experiments according to 7d-31 specification.

    Physical Meaning:
        Executes all required Level A tests and validates that the numerical
        core meets acceptance criteria for spectral accuracy, time integration,
        energy balance, and dimensional invariance.
    """

    def __init__(self, output_dir: Path = Path("output/level_a_experiments")):
        """
        Initialize experiment runner.

        Args:
            output_dir (Path): Directory for output files and reports.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Any] = {}
        self.acceptance_criteria = {
            "A0.1": {"E2_max": 1e-12, "anisotropy_max": 1e-12},
            "A0.2": {"E2_max": 1e-12},
            "A0.3": {"requires_exception": True},
            "A0.4": {"amplitude_error_max": 1e-8, "phase_error_max": 1e-8},
            "A0.5": {"residual_max": 1e-12, "orthogonality_max": 1e-12},
            "A1.1": {"invariance_max": 1e-12},
            "A1.2": {"invariance_max": 1e-12},
        }

    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all Level A tests (A0.1-A0.5, A1.1-A1.2).

        Returns:
            Dict[str, Any]: Comprehensive test results.
        """
        tests_root = Path("tests/unit/test_level_a")

        test_mappings = {
            "A0.1": "test_A01_plane_wave",
            "A0.2": "test_A02_multi_plane",
            "A0.3": "test_A03_zero_mode",
            "A0.4": "test_A04_time_harmonic",
            "A0.5": "test_A05_residual_energy",
            "A1.1": "test_A11_scale_length",
            "A1.2": "test_A12_units_invariance",
        }

        all_results = {}
        all_passed = True

        for test_id, test_pattern in test_mappings.items():
            print(f"\n{'='*60}")
            print(f"Running {test_id}: {test_pattern}")
            print(f"{'='*60}")

            result = self._run_single_test(test_id, tests_root, test_pattern)
            all_results[test_id] = result

            if result["status"] != "PASS":
                all_passed = False

            print(f"Status: {result['status']}")

        self.results = {
            "all_tests": all_results,
            "overall_status": "PASS" if all_passed else "FAIL",
            "timestamp": datetime.now().isoformat(),
            "system_info": self._collect_system_info(),
        }

        return self.results

    def _run_single_test(
        self, test_id: str, tests_root: Path, test_pattern: str
    ) -> Dict[str, Any]:
        """
        Run a single test and collect results.

        Args:
            test_id (str): Test identifier (e.g., "A0.1").
            test_pattern (str): Pytest pattern to match test.
            tests_root (Path): Root directory for tests.

        Returns:
            Dict[str, Any]: Test execution results.
        """
        try:
            test_file = tests_root / f"{test_pattern}.py"
            if not test_file.exists():
                return {
                    "test_id": test_id,
                    "status": "FAIL",
                    "error": f"Test file not found: {test_file}",
                    "acceptance_checked": False,
                }

            exit_code = pytest.main(
                [
                    str(test_file),
                    "-v",
                    "--tb=short",
                    "-q",
                ]
            )

            passed = exit_code == 0

            metrics = self._load_test_metrics(test_id)

            if metrics:
                acceptance_ok = self._check_acceptance(test_id, metrics)
                status = "PASS" if (passed and acceptance_ok) else "FAIL"
            else:
                acceptance_ok = None
                status = "PASS" if passed else "FAIL"

            return {
                "test_id": test_id,
                "status": status,
                "pytest_exit_code": exit_code,
                "pytest_passed": passed,
                "metrics": metrics,
                "acceptance_checked": acceptance_ok,
            }

        except Exception as e:
            return {
                "test_id": test_id,
                "status": "FAIL",
                "error": str(e),
                "acceptance_checked": False,
            }

    def _load_test_metrics(self, test_id: str) -> Dict[str, Any]:
        """
        Load metrics from test output JSON files.

        Args:
            test_id (str): Test identifier.

        Returns:
            Dict[str, Any]: Test metrics.
        """
        # Map test_id to output directory name (A0.1 -> A01)
        test_dir_map = {
            "A0.1": "A01",
            "A0.2": "A02",
            "A0.3": "A03",
            "A0.4": "A04",
            "A0.5": "A05",
            "A1.1": "A11",
            "A1.2": "A12",
        }

        output_dir_name = test_dir_map.get(test_id, test_id.replace(".", ""))
        metrics_path = Path("output") / output_dir_name / "metrics.json"

        if metrics_path.exists():
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load metrics from {metrics_path}: {e}")

        return {}

    def _check_acceptance(self, test_id: str, metrics: Dict[str, Any]) -> bool:
        """
        Check if test results meet acceptance criteria.

        Args:
            test_id (str): Test identifier.
            metrics (Dict[str, Any]): Test metrics.

        Returns:
            bool: True if acceptance criteria are met.
        """
        if test_id not in self.acceptance_criteria:
            return False

        criteria = self.acceptance_criteria[test_id]
        test_metrics = metrics.get("metrics", {})

        if test_id == "A0.1":
            error_L2 = test_metrics.get("error_L2", float("inf"))
            anisotropy = test_metrics.get("anisotropy", float("inf"))
            return (
                error_L2 <= criteria["E2_max"]
                and anisotropy <= criteria["anisotropy_max"]
            )

        elif test_id == "A0.2":
            error_L2 = test_metrics.get("error_L2", float("inf"))
            return error_L2 <= criteria["E2_max"]

        elif test_id == "A0.3":
            exception_raised = test_metrics.get("exception_raised", False)
            residual = test_metrics.get("residual_norm", float("inf"))
            return exception_raised and residual <= 1e-12

        elif test_id == "A0.4":
            amp_err = test_metrics.get("amplitude_error", float("inf"))
            phase_err = test_metrics.get("phase_error", float("inf"))
            return (
                amp_err <= criteria["amplitude_error_max"]
                and phase_err <= criteria["phase_error_max"]
            )

        elif test_id == "A0.5":
            residual = test_metrics.get("residual_norm", float("inf"))
            orthogonality = abs(test_metrics.get("orthogonality", float("inf")))
            return (
                residual <= criteria["residual_max"]
                and orthogonality <= criteria["orthogonality_max"]
            )

        elif test_id in ["A1.1", "A1.2"]:
            invariance = test_metrics.get("invariance_error", float("inf"))
            return invariance <= criteria["invariance_max"]

        return False

    def _collect_system_info(self) -> Dict[str, str]:
        """
        Collect system information for reproducibility.

        Returns:
            Dict[str, str]: System information.
        """
        try:
            numpy_version = importlib.metadata.version("numpy")
        except Exception:
            numpy_version = "unknown"

        try:
            scipy_version = importlib.metadata.version("scipy")
        except Exception:
            scipy_version = "unknown"

        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "numpy_version": numpy_version,
            "scipy_version": scipy_version,
        }

    def generate_report(self) -> None:
        """
        Generate comprehensive acceptance report.
        """
        report_path = self.output_dir / "acceptance_report.json"
        summary_path = self.output_dir / "acceptance_summary.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, default=str)

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("LEVEL A NUMERICAL EXPERIMENTS - ACCEPTANCE REPORT\n")
            f.write("Specification: 7d-31-БВП_план_численных_экспериментов_A.md\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Overall Status: {self.results['overall_status']}\n")
            f.write(f"Timestamp: {self.results['timestamp']}\n\n")

            f.write("System Information:\n")
            for key, value in self.results["system_info"].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            f.write("Test Results:\n")
            f.write("-" * 80 + "\n")
            for test_id, result in self.results["all_tests"].items():
                status_icon = "✅" if result["status"] == "PASS" else "❌"
                f.write(f"{status_icon} {test_id}: {result['status']}\n")
                if "metrics" in result and result["metrics"]:
                    metrics = result["metrics"].get("metrics", {})
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            f.write(f"    {key}: {value:.2e}\n")
                        else:
                            f.write(f"    {key}: {value}\n")
                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("ACCEPTANCE CRITERIA SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write("A0.1–A0.2: E₂ ≤ 10⁻¹², анизотропия ≤ 10⁻¹²\n")
            f.write("A0.3: корректный FAIL/сообщение при нарушении ŝ(0)=0 при λ=0\n")
            f.write("A0.4: ошибка амплитуды и фазы ≤ 10⁻⁸\n")
            f.write("A0.5: ‖r‖₂/‖s‖₂ ≤ 10⁻¹², ортогональность в допуске\n")
            f.write("A1.1–A1.2: инвариантность ≤ 10⁻¹²\n")
            f.write("\n")

            passed_count = sum(
                1 for r in self.results["all_tests"].values() if r["status"] == "PASS"
            )
            total_count = len(self.results["all_tests"])

            f.write(f"Tests Passed: {passed_count}/{total_count}\n")
            f.write(f"Overall Acceptance: {self.results['overall_status']}\n")
            f.write("=" * 80 + "\n")

        print(f"\n{'='*80}")
        print("ACCEPTANCE REPORT GENERATED")
        print(f"{'='*80}")
        print(f"Report: {report_path}")
        print(f"Summary: {summary_path}")
        print(f"\nOverall Status: {self.results['overall_status']}")

        passed_count = sum(
            1 for r in self.results["all_tests"].values() if r["status"] == "PASS"
        )
        total_count = len(self.results["all_tests"])
        print(f"Tests Passed: {passed_count}/{total_count}")


def main() -> int:
    """
    Main entry point for Level A experiments.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    runner = LevelAExperimentRunner()
    results = runner.run_all_tests()
    runner.generate_report()

    return 0 if results["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
