"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test reporting and visualization system for Level A tests.

This module implements the reporting system for Level A validation tests,
including metrics collection, visualization, and report generation.
"""

import numpy as np
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import logging

from bhlff.core.domain import Domain
from bhlff.core.domain.parameters import Parameters
from bhlff.core.fft.fft_solver_7d_basic import FFTSolver7DBasic


class TestReporter:
    """
    Test reporter for Level A validation tests.

    Physical Meaning:
        Collects and reports test results for Level A validation,
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
        report_path = os.path.join(self.output_dir, "level_a_test_report.json")

        report = {
            "summary": {
                "total_tests": len(self.results),
                "passed_tests": sum(
                    1 for r in self.results.values() if r["status"] == "PASS"
                ),
                "failed_tests": sum(
                    1 for r in self.results.values() if r["status"] == "FAIL"
                ),
                "generation_time": datetime.now().isoformat(),
            },
            "results": self.results,
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Generated JSON report: {report_path}")
        return report_path

    def generate_csv_log(self) -> str:
        """
        Generate CSV log of test results.

        Physical Meaning:
            Generates a CSV log for trend analysis and
            automated processing of test results.

        Returns:
            Path to generated CSV log
        """
        csv_path = os.path.join(self.output_dir, "level_a_test_log.csv")

        with open(csv_path, "w") as f:
            # Write header
            f.write(
                "test_id,status,error_L2,error_inf,anisotropy,residual_norm,execution_time,memory_usage\n"
            )

            # Write data
            for test_id, result in self.results.items():
                metrics = result["metrics"]
                f.write(
                    f"{test_id},{result['status']},"
                    f"{metrics.get('error_L2', 'N/A')},"
                    f"{metrics.get('error_inf', 'N/A')},"
                    f"{metrics.get('anisotropy', 'N/A')},"
                    f"{metrics.get('residual_norm', 'N/A')},"
                    f"{result['execution_time']},"
                    f"{result['memory_usage']}\n"
                )

        self.logger.info(f"Generated CSV log: {csv_path}")
        return csv_path

    def generate_html_report(self) -> str:
        """
        Generate HTML report with visualizations.

        Physical Meaning:
            Generates a comprehensive HTML report with
            visualizations and analysis of test results.

        Returns:
            Path to generated HTML report
        """
        html_path = os.path.join(self.output_dir, "level_a_test_report.html")

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Level A Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .test-result {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                .pass {{ background-color: #d4edda; border-color: #c3e6cb; }}
                .fail {{ background-color: #f8d7da; border-color: #f5c6cb; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }}
                .metric {{ background-color: #f8f9fa; padding: 10px; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Level A Test Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total Tests: {len(self.results)}</p>
                <p>Passed: {sum(1 for r in self.results.values() if r['status'] == 'PASS')}</p>
                <p>Failed: {sum(1 for r in self.results.values() if r['status'] == 'FAIL')}</p>
            </div>
        """

        # Add test results
        for test_id, result in self.results.items():
            status_class = "pass" if result["status"] == "PASS" else "fail"
            html_content += f"""
            <div class="test-result {status_class}">
                <h3>{result['test_name']} ({test_id})</h3>
                <p><strong>Status:</strong> {result['status']}</p>
                <p><strong>Execution Time:</strong> {result['execution_time']:.3f} seconds</p>
                <p><strong>Memory Usage:</strong> {result['memory_usage']:.1f} MB</p>
                <div class="metrics">
            """

            # Add metrics
            for metric_name, metric_value in result["metrics"].items():
                html_content += f"""
                    <div class="metric">
                        <strong>{metric_name}:</strong> {metric_value}
                    </div>
                """

            html_content += """
                </div>
            </div>
            """

        html_content += """
        </body>
        </html>
        """

        with open(html_path, "w") as f:
            f.write(html_content)

        self.logger.info(f"Generated HTML report: {html_path}")
        return html_path

    def create_visualizations(self) -> List[str]:
        """
        Create visualizations for test results.

        Physical Meaning:
            Creates visualizations to analyze test results,
            including error trends, performance metrics,
            and validation quality.

        Returns:
            List of paths to generated visualization files
        """
        visualization_paths = []

        # Create error trend plot
        if len(self.results) > 1:
            error_plot_path = self._create_error_trend_plot()
            visualization_paths.append(error_plot_path)

        # Create performance plot
        if len(self.results) > 1:
            performance_plot_path = self._create_performance_plot()
            visualization_paths.append(performance_plot_path)

        # Create metrics summary plot
        if len(self.results) > 1:
            metrics_plot_path = self._create_metrics_summary_plot()
            visualization_paths.append(metrics_plot_path)

        return visualization_paths

    def _create_error_trend_plot(self) -> str:
        """Create error trend plot."""
        plot_path = os.path.join(self.output_dir, "error_trends.png")

        test_ids = list(self.results.keys())
        error_L2 = [self.results[tid]["metrics"].get("error_L2", 0) for tid in test_ids]
        error_inf = [
            self.results[tid]["metrics"].get("error_inf", 0) for tid in test_ids
        ]

        plt.figure(figsize=(10, 6))
        plt.plot(test_ids, error_L2, "b-o", label="L2 Error")
        plt.plot(test_ids, error_inf, "r-s", label="Infinity Error")
        plt.xlabel("Test ID")
        plt.ylabel("Error")
        plt.title("Error Trends Across Tests")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        return plot_path

    def _create_performance_plot(self) -> str:
        """Create performance plot."""
        plot_path = os.path.join(self.output_dir, "performance_metrics.png")

        test_ids = list(self.results.keys())
        execution_times = [self.results[tid]["execution_time"] for tid in test_ids]
        memory_usage = [self.results[tid]["memory_usage"] for tid in test_ids]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Execution time
        ax1.plot(test_ids, execution_times, "b-o")
        ax1.set_xlabel("Test ID")
        ax1.set_ylabel("Execution Time (s)")
        ax1.set_title("Execution Time Trends")
        ax1.grid(True)
        ax1.tick_params(axis="x", rotation=45)

        # Memory usage
        ax2.plot(test_ids, memory_usage, "r-s")
        ax2.set_xlabel("Test ID")
        ax2.set_ylabel("Memory Usage (MB)")
        ax2.set_title("Memory Usage Trends")
        ax2.grid(True)
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        return plot_path

    def _create_metrics_summary_plot(self) -> str:
        """Create metrics summary plot."""
        plot_path = os.path.join(self.output_dir, "metrics_summary.png")

        # Collect all metrics
        all_metrics = set()
        for result in self.results.values():
            all_metrics.update(result["metrics"].keys())

        # Create subplots for each metric
        n_metrics = len(all_metrics)
        if n_metrics == 0:
            return plot_path

        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        test_ids = list(self.results.keys())

        for i, metric in enumerate(sorted(all_metrics)):
            if i >= len(axes):
                break

            values = [self.results[tid]["metrics"].get(metric, 0) for tid in test_ids]

            axes[i].bar(test_ids, values)
            axes[i].set_xlabel("Test ID")
            axes[i].set_ylabel(metric)
            axes[i].set_title(f"{metric} Summary")
            axes[i].tick_params(axis="x", rotation=45)
            axes[i].grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(all_metrics), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        return plot_path

    def generate_summary_report(self) -> str:
        """
        Generate summary report.

        Physical Meaning:
            Generates a summary report with key findings
            and recommendations for Level A validation.

        Returns:
            Path to generated summary report
        """
        summary_path = os.path.join(self.output_dir, "level_a_summary.txt")

        with open(summary_path, "w") as f:
            f.write("Level A Test Summary Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Tests: {len(self.results)}\n")
            f.write(
                f"Passed: {sum(1 for r in self.results.values() if r['status'] == 'PASS')}\n"
            )
            f.write(
                f"Failed: {sum(1 for r in self.results.values() if r['status'] == 'FAIL')}\n\n"
            )

            f.write("Test Results:\n")
            f.write("-" * 20 + "\n")
            for test_id, result in self.results.items():
                f.write(f"{test_id}: {result['status']}\n")
                if result["status"] == "FAIL":
                    f.write(f"  - Error details: {result['metrics']}\n")

            f.write("\nKey Metrics:\n")
            f.write("-" * 15 + "\n")

            # Calculate summary statistics
            all_errors_L2 = [
                r["metrics"].get("error_L2", 0)
                for r in self.results.values()
                if "error_L2" in r["metrics"]
            ]
            if all_errors_L2:
                f.write(
                    f"L2 Error - Min: {min(all_errors_L2):.2e}, Max: {max(all_errors_L2):.2e}\n"
                )

            all_residuals = [
                r["metrics"].get("residual_norm", 0)
                for r in self.results.values()
                if "residual_norm" in r["metrics"]
            ]
            if all_residuals:
                f.write(
                    f"Residual Norm - Min: {min(all_residuals):.2e}, Max: {max(all_residuals):.2e}\n"
                )

            f.write("\nRecommendations:\n")
            f.write("-" * 20 + "\n")

            failed_tests = [
                tid for tid, r in self.results.items() if r["status"] == "FAIL"
            ]
            if failed_tests:
                f.write(f"Failed tests: {', '.join(failed_tests)}\n")
                f.write("Review failed tests and check tolerance settings.\n")
            else:
                f.write("All tests passed! Level A validation is successful.\n")

        self.logger.info(f"Generated summary report: {summary_path}")
        return summary_path


if __name__ == "__main__":
    # Example usage
    reporter = TestReporter("output/level_a")

    # Example test results
    reporter.record_test_result(
        test_id="A01",
        test_name="plane_wave",
        status="PASS",
        metrics={"error_L2": 1.2e-13, "error_inf": 2.1e-13, "anisotropy": 3.4e-14},
        parameters={"mu": 1.0, "beta": 1.0, "lambda": 0.1},
        execution_time=0.123,
        memory_usage=45.6,
    )

    # Generate reports
    json_report = reporter.generate_json_report()
    csv_log = reporter.generate_csv_log()
    html_report = reporter.generate_html_report()
    visualizations = reporter.create_visualizations()
    summary_report = reporter.generate_summary_report()

    print(f"Generated reports:")
    print(f"  JSON: {json_report}")
    print(f"  CSV: {csv_log}")
    print(f"  HTML: {html_report}")
    print(f"  Summary: {summary_report}")
    print(f"  Visualizations: {len(visualizations)} files")
