"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Structured logging system for the 7D phase field theory project.

This module provides structured logging with different levels of detail
and export to various formats for the 7D phase field theory implementation.

Physical Meaning:
    Logging system tracks the evolution of phase field simulations,
    providing detailed information about solver convergence, energy
    evolution, and physical quantities for analysis and debugging.

Example:
    >>> logger = StructuredLogger("solver")
    >>> logger.log_experiment_start("test_run", {"mu": 1.0, "beta": 1.5})
    >>> logger.log_solver_step(1, 1e-6, 0.1)
    >>> logger.log_experiment_end("test_run", {"energy": 0.5, "converged": True})
"""

import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime


class StructuredLogger:
    """
    Structured logger for the 7D phase field theory project.

    Physical Meaning:
        Provides structured logging with different levels of detail
        and export to various formats, tracking the evolution of
        phase field simulations and physical quantities.

    Features:
        - Console and file logging
        - Structured experiment logging
        - Solver step tracking
        - Validation result logging
        - Export to JSON format
    """

    def __init__(self, name: str, log_dir: Optional[Path] = None):
        """
        Initialize structured logger.

        Physical Meaning:
            Sets up the logger with appropriate handlers and formatters
            for tracking phase field simulation progress and results.

        Args:
            name (str): Logger name.
            log_dir (Optional[Path]): Directory for log files.
        """
        self.name = name
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Setup handlers
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """
        Setup logging handlers.

        Physical Meaning:
            Configures console and file handlers with appropriate
            formatters for different logging levels.
        """
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{self.name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def log_experiment_start(
        self, experiment_name: str, parameters: Dict[str, Any]
    ) -> None:
        """
        Log experiment start.

        Physical Meaning:
            Records the start of a phase field simulation experiment
            with all relevant parameters for reproducibility.

        Args:
            experiment_name (str): Name of the experiment.
            parameters (Dict[str, Any]): Experiment parameters.
        """
        self.logger.info(f"Starting experiment: {experiment_name}")
        self.logger.debug(f"Parameters: {json.dumps(parameters, indent=2)}")

    def log_experiment_end(self, experiment_name: str, results: Dict[str, Any]) -> None:
        """
        Log experiment end.

        Physical Meaning:
            Records the completion of a phase field simulation experiment
            with all relevant results and metrics.

        Args:
            experiment_name (str): Name of the experiment.
            results (Dict[str, Any]): Experiment results.
        """
        self.logger.info(f"Completed experiment: {experiment_name}")
        self.logger.debug(f"Results: {json.dumps(results, indent=2)}")

    def log_solver_step(self, step: int, residual: float, time: float) -> None:
        """
        Log solver step.

        Physical Meaning:
            Records solver convergence information for each iteration,
            tracking the evolution of the residual and computation time.

        Args:
            step (int): Solver step number.
            residual (float): Residual value.
            time (float): Computation time in seconds.
        """
        self.logger.debug(f"Step {step}: residual={residual:.2e}, time={time:.3f}s")

    def log_validation_result(
        self, test_name: str, passed: bool, metrics: Dict[str, float]
    ) -> None:
        """
        Log validation result.

        Physical Meaning:
            Records validation test results with detailed metrics,
            ensuring physical consistency and numerical accuracy.

        Args:
            test_name (str): Name of the validation test.
            passed (bool): Whether the test passed.
            metrics (Dict[str, float]): Validation metrics.
        """
        status = "PASS" if passed else "FAIL"
        self.logger.info(f"Validation {test_name}: {status}")
        self.logger.debug(f"Metrics: {json.dumps(metrics, indent=2)}")

    def log_energy_evolution(
        self, time: float, energy: float, energy_change: float
    ) -> None:
        """
        Log energy evolution.

        Physical Meaning:
            Records the evolution of the total energy of the phase field
            system, tracking energy conservation and dissipation.

        Args:
            time (float): Current time.
            energy (float): Total energy.
            energy_change (float): Energy change from previous step.
        """
        self.logger.debug(
            f"Time {time:.3f}: energy={energy:.6f}, change={energy_change:.2e}"
        )

    def log_topological_charge(self, charge: float, location: tuple) -> None:
        """
        Log topological charge calculation.

        Physical Meaning:
            Records topological charge calculations at specific locations,
            tracking the formation and evolution of topological defects.

        Args:
            charge (float): Topological charge value.
            location (tuple): Location of the charge calculation.
        """
        self.logger.info(f"Topological charge: {charge:.3f} at {location}")

    def log_phase_coherence(self, coherence: float, time: float) -> None:
        """
        Log phase coherence measurement.

        Physical Meaning:
            Records phase coherence measurements, tracking the degree
            of phase alignment across the field.

        Args:
            coherence (float): Phase coherence value.
            time (float): Current time.
        """
        self.logger.debug(f"Time {time:.3f}: phase coherence={coherence:.3f}")

    def log_error(
        self, error_type: str, message: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log error with details.

        Physical Meaning:
            Records errors that occur during phase field simulations,
            providing detailed information for debugging and analysis.

        Args:
            error_type (str): Type of error.
            message (str): Error message.
            details (Optional[Dict[str, Any]]): Additional error details.
        """
        self.logger.error(f"{error_type}: {message}")
        if details:
            self.logger.error(f"Details: {json.dumps(details, indent=2)}")

    def log_warning(
        self, warning_type: str, message: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log warning with details.

        Physical Meaning:
            Records warnings that occur during phase field simulations,
            providing information about potential issues or deviations
            from expected behavior.

        Args:
            warning_type (str): Type of warning.
            message (str): Warning message.
            details (Optional[Dict[str, Any]]): Additional warning details.
        """
        self.logger.warning(f"{warning_type}: {message}")
        if details:
            self.logger.warning(f"Details: {json.dumps(details, indent=2)}")

    def export_logs_to_json(self, output_path: Optional[Path] = None) -> Path:
        """
        Export logs to JSON format.

        Physical Meaning:
            Exports structured log data to JSON format for analysis
            and visualization of simulation results.

        Args:
            output_path (Optional[Path]): Path to save JSON file.

        Returns:
            Path: Path to the exported JSON file.
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.log_dir / f"{self.name}_export_{timestamp}.json"

        # This would export the log data to JSON
        # For now, just create an empty file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"logs": "exported"}, f, indent=2)

        return output_path

    def get_log_summary(self) -> Dict[str, Any]:
        """
        Get log summary statistics.

        Physical Meaning:
            Provides summary statistics of the logged data,
            including experiment counts, error rates, and
            performance metrics.

        Returns:
            Dict[str, Any]: Log summary statistics.
        """
        # This would analyze the log data and provide statistics
        # For now, return a basic summary
        return {
            "logger_name": self.name,
            "log_level": self.logger.level,
            "handlers": len(self.logger.handlers),
            "log_dir": str(self.log_dir),
        }

    def __repr__(self) -> str:
        """String representation of the logger."""
        return f"StructuredLogger(name={self.name}, log_dir={self.log_dir})"
