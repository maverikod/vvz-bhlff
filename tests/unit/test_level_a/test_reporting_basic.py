"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic test reporting and visualization system for Level A tests.

This module implements the basic reporting system for Level A validation tests,
including metrics collection and basic report generation.

Physical Meaning:
    Collects and reports basic test results for Level A validation,
    ensuring comprehensive documentation of solver performance
    and validation metrics.
"""

import numpy as np
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from bhlff.core.domain import Domain
from bhlff.core.domain.parameters import Parameters
from bhlff.core.fft.fft_solver_7d_basic import FFTSolver7DBasic


class TestReporterBasic:
    """
    Basic test reporter for Level A validation tests.

    Physical Meaning:
        Collects and reports basic test results for Level A validation,
        ensuring comprehensive documentation of solver performance
        and validation metrics.
    """

    def __init__(self, output_dir: str = "output/level_a"):
        """
        Initialize test reporter.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        self.results = {}
        self.metrics = {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def record_test_result(
        self,
        test_id: str,
        test_name: str,
        status: str,
        metrics: Dict[str, Any],
        parameters: Dict[str, Any],
        execution_time: float,
        memory_usage: float,
    ) -> None:
        """
        Record test result.

        Physical Meaning:
            Records the results of a validation test, including
            performance metrics and physical parameters.

        Args:
            test_id: Test identifier
            test_name: Test name
            status: Test status (PASS/FAIL)
            metrics: Test metrics
            parameters: Test parameters
            execution_time: Execution time in seconds
            memory_usage: Memory usage in MB
        """
        result = {
            "test_id": test_id,
            "test_name": test_name,
            "status": status,
            "metrics": metrics,
            "parameters": parameters,
            "execution_time": execution_time,
            "memory_usage": memory_usage,
            "timestamp": datetime.now().isoformat(),
        }

        self.results[test_id] = result
        self.metrics[test_id] = metrics

        self.logger.info(f"Recorded result for {test_id}: {status}")

    def generate_json_report(self) -> str:
        """
        Generate JSON report of all test results.

        Physical Meaning:
            Generates a comprehensive JSON report containing
            all test results, metrics, and parameters.

        Returns:
            Path to generated JSON report
        """
        report_data = {
            "summary": {
                "total_tests": len(self.results),
                "passed_tests": sum(1 for r in self.results.values() if r["status"] == "PASS"),
                "failed_tests": sum(1 for r in self.results.values() if r["status"] == "FAIL"),
                "generation_time": datetime.now().isoformat(),
            },
            "results": self.results,
            "metrics": self.metrics,
        }

        report_path = os.path.join(self.output_dir, "test_results.json")
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)

        self.logger.info(f"Generated JSON report: {report_path}")
        return report_path

    def generate_csv_log(self) -> str:
        """
        Generate CSV log of test results.

        Physical Meaning:
            Generates a CSV log containing test results
            for easy analysis and processing.

        Returns:
            Path to generated CSV log
        """
        csv_path = os.path.join(self.output_dir, "test_results.csv")
        
        with open(csv_path, "w") as f:
            # Write header
            f.write("test_id,test_name,status,execution_time,memory_usage,timestamp\n")
            
            # Write data
            for result in self.results.values():
                f.write(f"{result['test_id']},{result['test_name']},{result['status']},"
                       f"{result['execution_time']},{result['memory_usage']},{result['timestamp']}\n")

        self.logger.info(f"Generated CSV log: {csv_path}")
        return csv_path

    def get_test_summary(self) -> Dict[str, Any]:
        """
        Get summary of test results.

        Physical Meaning:
            Provides a summary of all test results,
            including statistics and key metrics.

        Returns:
            Dictionary containing test summary
        """
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r["status"] == "PASS")
        failed_tests = sum(1 for r in self.results.values() if r["status"] == "FAIL")
        
        total_execution_time = sum(r["execution_time"] for r in self.results.values())
        total_memory_usage = sum(r["memory_usage"] for r in self.results.values())
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_execution_time": total_execution_time,
            "total_memory_usage": total_memory_usage,
            "average_execution_time": total_execution_time / total_tests if total_tests > 0 else 0,
            "average_memory_usage": total_memory_usage / total_tests if total_tests > 0 else 0,
        }

    def get_failed_tests(self) -> List[Dict[str, Any]]:
        """
        Get list of failed tests.

        Physical Meaning:
            Provides a list of all failed tests
            for analysis and debugging.

        Returns:
            List of failed test results
        """
        return [result for result in self.results.values() if result["status"] == "FAIL"]

    def get_passed_tests(self) -> List[Dict[str, Any]]:
        """
        Get list of passed tests.

        Physical Meaning:
            Provides a list of all passed tests
            for verification and analysis.

        Returns:
            List of passed test results
        """
        return [result for result in self.results.values() if result["status"] == "PASS"]

    def get_test_metrics(self, test_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metrics for a specific test.

        Physical Meaning:
            Retrieves the metrics for a specific test
            for detailed analysis.

        Args:
            test_id: Test identifier

        Returns:
            Test metrics or None if not found
        """
        return self.metrics.get(test_id)

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all test metrics.

        Physical Meaning:
            Retrieves all test metrics for
            comprehensive analysis.

        Returns:
            Dictionary of all test metrics
        """
        return self.metrics

    def clear_results(self) -> None:
        """
        Clear all test results.

        Physical Meaning:
            Clears all stored test results
            for a fresh start.
        """
        self.results.clear()
        self.metrics.clear()
        self.logger.info("Cleared all test results")

    def export_results(self, format: str = "json") -> str:
        """
        Export results in specified format.

        Physical Meaning:
            Exports test results in the specified
            format for external analysis.

        Args:
            format: Export format ("json" or "csv")

        Returns:
            Path to exported file
        """
        if format.lower() == "json":
            return self.generate_json_report()
        elif format.lower() == "csv":
            return self.generate_csv_log()
        else:
            raise ValueError(f"Unsupported format: {format}")


if __name__ == "__main__":
    # Example usage
    reporter = TestReporterBasic()
    
    # Record some test results
    reporter.record_test_result(
        test_id="test_001",
        test_name="Basic FFT Test",
        status="PASS",
        metrics={"accuracy": 0.99, "speed": 1.5},
        parameters={"N": 64, "L": 1.0},
        execution_time=0.1,
        memory_usage=10.5
    )
    
    # Generate reports
    json_report = reporter.generate_json_report()
    csv_log = reporter.generate_csv_log()
    
    print(f"Generated reports: {json_report}, {csv_log}")
    print(f"Test summary: {reporter.get_test_summary()}")
