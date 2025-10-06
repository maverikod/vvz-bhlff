"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Failure detection and boundary analysis for Level E experiments.

This module implements comprehensive failure detection and boundary
analysis for the 7D phase field theory, identifying limits of
applicability and diagnosing system failures.

Theoretical Background:
    Failure detection investigates the boundaries of applicability
    of the 7D theory and diagnoses system failures. This includes
    detection of passivity violations, singular modes, and other
    physical inconsistencies.

Mathematical Foundation:
    Detects violations of physical principles:
    - Passivity: Re Y_out ≥ 0
    - Singular modes: λ = 0 with ŝ(0) ≠ 0
    - Energy conservation: |ΔE|/E < threshold

Example:
    >>> detector = FailureDetector(config)
    >>> failures = detector.detect_failures()
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import json
import logging


class FailureDetector:
    """
    Failure detection and boundary analysis.

    Physical Meaning:
        Identifies limits of applicability of the 7D theory and
        diagnoses system failures through comprehensive analysis
        of physical principles and numerical stability.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize failure detector.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._setup_logging()
        self._setup_failure_criteria()

    def _setup_logging(self) -> None:
        """Setup logging for failure detection."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _setup_failure_criteria(self) -> None:
        """Setup criteria for failure detection."""
        self.failure_criteria = {
            "passivity_violation": self._check_passivity_violation,
            "singular_mode": self._check_singular_mode,
            "energy_conservation": self._check_energy_conservation,
            "topological_charge": self._check_topological_charge,
            "numerical_stability": self._check_numerical_stability,
        }

    def detect_failures(self) -> Dict[str, Any]:
        """
        Detect all types of failures in the system.

        Physical Meaning:
            Comprehensive analysis of system failures including
            passivity violations, singular modes, energy conservation
            violations, and numerical instabilities.

        Returns:
            Dictionary containing all detected failures
        """
        failures = {}

        for failure_type, check_function in self.failure_criteria.items():
            try:
                result = check_function()
                failures[failure_type] = result

                if result["detected"]:
                    self.logger.warning(f"Failure detected: {failure_type}")
                else:
                    self.logger.info(f"No failure detected: {failure_type}")

            except Exception as e:
                self.logger.error(f"Error checking {failure_type}: {e}")
                failures[failure_type] = {
                    "detected": True,
                    "error": str(e),
                    "type": "check_error",
                }

        # Overall failure assessment
        overall_assessment = self._assess_overall_failures(failures)
        failures["overall_assessment"] = overall_assessment

        return failures

    def _check_passivity_violation(self) -> Dict[str, Any]:
        """
        Check for passivity violations.

        Physical Meaning:
            Verifies that the system remains passive (Re Y_out ≥ 0),
            which is a fundamental physical requirement for energy
            conservation and stability.
        """
        # Extract impedance data
        impedance_data = self._get_impedance_data()

        if impedance_data is None:
            return {
                "detected": False,
                "reason": "No impedance data available",
                "violations": [],
            }

        violations = []

        for freq, impedance in impedance_data.items():
            if isinstance(impedance, complex):
                real_part = impedance.real
                if real_part < 0:
                    violations.append(
                        {
                            "frequency": freq,
                            "impedance": impedance,
                            "real_part": real_part,
                            "violation_magnitude": abs(real_part),
                        }
                    )

        detected = len(violations) > 0

        return {
            "detected": detected,
            "violations": violations,
            "count": len(violations),
            "max_violation": (
                max([v["violation_magnitude"] for v in violations])
                if violations
                else 0.0
            ),
        }

    def _get_impedance_data(self) -> Optional[Dict[float, complex]]:
        """Get impedance data for passivity checking."""
        # Placeholder implementation - in real case, this would extract
        # impedance data from simulation results

        # Simulate some impedance data
        frequencies = np.logspace(0, 3, 100)
        impedances = []

        for freq in frequencies:
            # Simulate impedance with some chance of passivity violation
            if np.random.random() < 0.1:  # 10% chance of violation
                real_part = -np.random.uniform(0.01, 0.1)
                imag_part = np.random.uniform(-1, 1)
            else:
                real_part = np.random.uniform(0.01, 1.0)
                imag_part = np.random.uniform(-1, 1)

            impedances.append(complex(real_part, imag_part))

        return dict(zip(frequencies, impedances))

    def _check_singular_mode(self) -> Dict[str, Any]:
        """
        Check for singular modes.

        Physical Meaning:
            Detects singular modes where λ = 0 with ŝ(0) ≠ 0,
            which can lead to numerical instabilities and
            unphysical behavior.
        """
        # Extract mode data
        mode_data = self._get_mode_data()

        if mode_data is None:
            return {
                "detected": False,
                "reason": "No mode data available",
                "singular_modes": [],
            }

        singular_modes = []

        for mode_id, mode_info in mode_data.items():
            lambda_val = mode_info.get("lambda", 1.0)
            source_val = mode_info.get("source", 0.0)

            # Check for singular mode condition
            if abs(lambda_val) < 1e-10 and abs(source_val) > 1e-10:
                singular_modes.append(
                    {
                        "mode_id": mode_id,
                        "lambda": lambda_val,
                        "source": source_val,
                        "singularity_strength": (
                            abs(source_val) / abs(lambda_val)
                            if lambda_val != 0
                            else float("inf")
                        ),
                    }
                )

        detected = len(singular_modes) > 0

        return {
            "detected": detected,
            "singular_modes": singular_modes,
            "count": len(singular_modes),
            "max_singularity": (
                max([m["singularity_strength"] for m in singular_modes])
                if singular_modes
                else 0.0
            ),
        }

    def _get_mode_data(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Get mode data for singular mode checking."""
        # Placeholder implementation - in real case, this would extract
        # mode data from simulation results

        # Simulate some mode data
        modes = {}
        for i in range(10):
            lambda_val = np.random.uniform(0.001, 1.0)
            source_val = np.random.uniform(0.0, 0.1)

            # Occasionally create singular modes
            if np.random.random() < 0.05:  # 5% chance of singular mode
                lambda_val = np.random.uniform(0, 1e-10)
                source_val = np.random.uniform(0.01, 0.1)

            modes[f"mode_{i}"] = {"lambda": lambda_val, "source": source_val}

        return modes

    def _check_energy_conservation(self) -> Dict[str, Any]:
        """
        Check for energy conservation violations.

        Physical Meaning:
            Verifies that energy is conserved within acceptable
            tolerances, which is fundamental for physical consistency.
        """
        # Extract energy data
        energy_data = self._get_energy_data()

        if energy_data is None:
            return {
                "detected": False,
                "reason": "No energy data available",
                "violations": [],
            }

        violations = []
        threshold = 0.01  # 1% energy conservation threshold

        for time, energy in energy_data.items():
            if time > 0:
                # Check energy change
                energy_change = abs(energy - energy_data[0])
                relative_change = (
                    energy_change / abs(energy_data[0]) if energy_data[0] != 0 else 0
                )

                if relative_change > threshold:
                    violations.append(
                        {
                            "time": time,
                            "energy": energy,
                            "change": energy_change,
                            "relative_change": relative_change,
                        }
                    )

        detected = len(violations) > 0

        return {
            "detected": detected,
            "violations": violations,
            "count": len(violations),
            "max_violation": (
                max([v["relative_change"] for v in violations]) if violations else 0.0
            ),
        }

    def _get_energy_data(self) -> Optional[Dict[float, float]]:
        """Get energy data for conservation checking."""
        # Placeholder implementation - in real case, this would extract
        # energy data from simulation results

        # Simulate energy data
        times = np.linspace(0, 10, 100)
        energies = []

        for t in times:
            # Simulate energy with some conservation violations
            base_energy = 1.0
            if np.random.random() < 0.05:  # 5% chance of violation
                violation = np.random.uniform(0.01, 0.05)
                energy = base_energy + violation
            else:
                energy = base_energy + np.random.normal(0, 0.001)

            energies.append(energy)

        return dict(zip(times, energies))

    def _check_topological_charge(self) -> Dict[str, Any]:
        """
        Check for topological charge violations.

        Physical Meaning:
            Verifies that topological charge remains integer-valued
            within acceptable tolerances, which is fundamental for
            topological consistency.
        """
        # Extract topological charge data
        charge_data = self._get_topological_charge_data()

        if charge_data is None:
            return {
                "detected": False,
                "reason": "No topological charge data available",
                "violations": [],
            }

        violations = []
        threshold = 0.1  # 10% tolerance for topological charge

        for time, charge in charge_data.items():
            # Check if charge is approximately integer
            nearest_integer = round(charge)
            deviation = abs(charge - nearest_integer)

            if deviation > threshold:
                violations.append(
                    {
                        "time": time,
                        "charge": charge,
                        "nearest_integer": nearest_integer,
                        "deviation": deviation,
                    }
                )

        detected = len(violations) > 0

        return {
            "detected": detected,
            "violations": violations,
            "count": len(violations),
            "max_deviation": (
                max([v["deviation"] for v in violations]) if violations else 0.0
            ),
        }

    def _get_topological_charge_data(self) -> Optional[Dict[float, float]]:
        """Get topological charge data for checking."""
        # Placeholder implementation - in real case, this would extract
        # topological charge data from simulation results

        # Simulate topological charge data
        times = np.linspace(0, 10, 100)
        charges = []

        for t in times:
            # Simulate charge with some violations
            base_charge = 1.0
            if np.random.random() < 0.03:  # 3% chance of violation
                deviation = np.random.uniform(0.1, 0.3)
                charge = base_charge + deviation
            else:
                charge = base_charge + np.random.normal(0, 0.01)

            charges.append(charge)

        return dict(zip(times, charges))

    def _check_numerical_stability(self) -> Dict[str, Any]:
        """
        Check for numerical stability issues.

        Physical Meaning:
            Detects numerical instabilities such as NaN values,
            infinite values, and excessive growth rates.
        """
        # Extract numerical data
        numerical_data = self._get_numerical_data()

        if numerical_data is None:
            return {
                "detected": False,
                "reason": "No numerical data available",
                "instabilities": [],
            }

        instabilities = []

        for field_name, field_data in numerical_data.items():
            # Check for NaN values
            nan_count = np.isnan(field_data).sum()
            if nan_count > 0:
                instabilities.append(
                    {"field": field_name, "type": "NaN_values", "count": nan_count}
                )

            # Check for infinite values
            inf_count = np.isinf(field_data).sum()
            if inf_count > 0:
                instabilities.append(
                    {"field": field_name, "type": "infinite_values", "count": inf_count}
                )

            # Check for excessive growth
            if len(field_data) > 1:
                growth_rate = np.max(np.abs(np.diff(field_data)))
                if growth_rate > 10.0:  # Threshold for excessive growth
                    instabilities.append(
                        {
                            "field": field_name,
                            "type": "excessive_growth",
                            "growth_rate": growth_rate,
                        }
                    )

        detected = len(instabilities) > 0

        return {
            "detected": detected,
            "instabilities": instabilities,
            "count": len(instabilities),
        }

    def _get_numerical_data(self) -> Optional[Dict[str, np.ndarray]]:
        """Get numerical data for stability checking."""
        # Placeholder implementation - in real case, this would extract
        # numerical data from simulation results

        # Simulate numerical data
        data = {}

        # Simulate field data
        field_data = np.random.normal(0, 1, 1000)

        # Occasionally introduce instabilities
        if np.random.random() < 0.1:  # 10% chance of instability
            # Add NaN values
            nan_indices = np.random.choice(len(field_data), 10, replace=False)
            field_data[nan_indices] = np.nan

        if np.random.random() < 0.05:  # 5% chance of infinite values
            # Add infinite values
            inf_indices = np.random.choice(len(field_data), 5, replace=False)
            field_data[inf_indices] = np.inf

        data["field"] = field_data

        return data

    def _assess_overall_failures(self, failures: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall failure status."""
        detected_failures = [
            name
            for name, result in failures.items()
            if isinstance(result, dict) and result.get("detected", False)
        ]

        failure_count = len(detected_failures)

        if failure_count == 0:
            status = "healthy"
            severity = "none"
        elif failure_count == 1:
            status = "warning"
            severity = "low"
        elif failure_count <= 3:
            status = "critical"
            severity = "medium"
        else:
            status = "failed"
            severity = "high"

        return {
            "status": status,
            "severity": severity,
            "failure_count": failure_count,
            "detected_failures": detected_failures,
        }

    def analyze_failure_boundaries(
        self, parameter_ranges: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Any]:
        """
        Analyze boundaries where failures occur.

        Physical Meaning:
            Identifies parameter ranges where the system fails,
            establishing boundaries of applicability.
        """
        boundaries = {}

        for param_name, (min_val, max_val) in parameter_ranges.items():
            print(f"Analyzing failure boundaries for {param_name}")

            # Test parameter range
            test_values = np.linspace(min_val, max_val, 20)
            failure_points = []

            for value in test_values:
                # Create test configuration
                test_config = self.config.copy()
                test_config[param_name] = value

                # Test for failures
                test_detector = FailureDetector(test_config)
                test_failures = test_detector.detect_failures()

                # Check if any failures detected
                has_failures = any(
                    result.get("detected", False)
                    for result in test_failures.values()
                    if isinstance(result, dict)
                )

                if has_failures:
                    failure_points.append(value)

            # Analyze failure boundaries
            if failure_points:
                boundaries[param_name] = {
                    "failure_points": failure_points,
                    "min_failure": min(failure_points),
                    "max_failure": max(failure_points),
                    "failure_range": max(failure_points) - min(failure_points),
                }
            else:
                boundaries[param_name] = {
                    "failure_points": [],
                    "min_failure": None,
                    "max_failure": None,
                    "failure_range": 0,
                }

        return boundaries

    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """
        Save failure detection results to file.

        Args:
            results: Detection results dictionary
            filename: Output filename
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(results)

        with open(filename, "w") as f:
            json.dump(serializable_results, f, indent=2)

    def _make_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
