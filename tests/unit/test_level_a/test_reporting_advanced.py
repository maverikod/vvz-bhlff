"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Advanced test reporting and visualization system for Level A tests.

This module implements the advanced reporting system for Level A validation tests,
including advanced visualization, HTML reports, and comprehensive analysis.

Physical Meaning:
    Provides advanced reporting capabilities for Level A validation,
    including visualization, trend analysis, and comprehensive reporting.
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


class TestReporterAdvanced:
    """
    Advanced test reporter for Level A validation tests.

    Physical Meaning:
        Provides advanced reporting capabilities for Level A validation,
        including visualization, trend analysis, and comprehensive reporting.
    """

    def __init__(self, output_dir: str = "output/level_a"):
        """
        Initialize advanced test reporter.

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

    def generate_html_report(self) -> str:
        """
        Generate HTML report of all test results.

        Physical Meaning:
            Generates a comprehensive HTML report containing
            all test results, metrics, and visualizations.

        Returns:
            Path to generated HTML report
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Level A Test Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ background-color: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .test-result {{ border: 1px solid #ddd; margin: 10px 0; padding: 10px; border-radius: 5px; }}
                .pass {{ background-color: #d4edda; }}
                .fail {{ background-color: #f8d7da; }}
                .metrics {{ background-color: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Level A Test Results</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Tests: {len(self.results)}</p>
                <p>Passed: {sum(1 for r in self.results.values() if r['status'] == 'PASS')}</p>
                <p>Failed: {sum(1 for r in self.results.values() if r['status'] == 'FAIL')}</p>
            </div>
            
            <div class="test-results">
                <h2>Test Results</h2>
        """

        for result in self.results.values():
            status_class = "pass" if result["status"] == "PASS" else "fail"
            html_content += f"""
                <div class="test-result {status_class}">
                    <h3>{result['test_name']} ({result['test_id']})</h3>
                    <p>Status: {result['status']}</p>
                    <p>Execution Time: {result['execution_time']:.3f}s</p>
                    <p>Memory Usage: {result['memory_usage']:.1f}MB</p>
                    <div class="metrics">
                        <h4>Metrics:</h4>
                        <pre>{json.dumps(result['metrics'], indent=2)}</pre>
                    </div>
                </div>
            """

        html_content += """
            </div>
        </body>
        </html>
        """

        report_path = os.path.join(self.output_dir, "test_results.html")
        with open(report_path, "w") as f:
            f.write(html_content)

        self.logger.info(f"Generated HTML report: {report_path}")
        return report_path

    def create_visualizations(self) -> List[str]:
        """
        Create visualizations of test results.

        Physical Meaning:
            Creates visualizations to analyze test results
            and identify trends and patterns.

        Returns:
            List of paths to generated visualization files
        """
        visualization_paths = []
        
        # Create error trend plot
        error_plot_path = self._create_error_trend_plot()
        if error_plot_path:
            visualization_paths.append(error_plot_path)
        
        # Create performance plot
        performance_plot_path = self._create_performance_plot()
        if performance_plot_path:
            visualization_paths.append(performance_plot_path)
        
        # Create metrics summary plot
        metrics_plot_path = self._create_metrics_summary_plot()
        if metrics_plot_path:
            visualization_paths.append(metrics_plot_path)
        
        return visualization_paths

    def _create_error_trend_plot(self) -> str:
        """
        Create error trend plot.

        Physical Meaning:
            Creates a plot showing error trends
            across different tests.

        Returns:
            Path to generated plot
        """
        if not self.results:
            return None
        
        # Extract error data
        test_ids = list(self.results.keys())
        errors = []
        
        for result in self.results.values():
            if "error" in result["metrics"]:
                errors.append(result["metrics"]["error"])
            else:
                errors.append(0.0)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(test_ids, errors, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('Test ID')
        plt.ylabel('Error')
        plt.title('Error Trends Across Tests')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plot_path = os.path.join(self.output_dir, "error_trends.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path

    def _create_performance_plot(self) -> str:
        """
        Create performance plot.

        Physical Meaning:
            Creates a plot showing performance metrics
            across different tests.

        Returns:
            Path to generated plot
        """
        if not self.results:
            return None
        
        # Extract performance data
        test_ids = list(self.results.keys())
        execution_times = [r["execution_time"] for r in self.results.values()]
        memory_usage = [r["memory_usage"] for r in self.results.values()]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Execution time plot
        ax1.plot(test_ids, execution_times, 'g-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Test ID')
        ax1.set_ylabel('Execution Time (s)')
        ax1.set_title('Execution Time Trends')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Memory usage plot
        ax2.plot(test_ids, memory_usage, 'r-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Test ID')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage Trends')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, "performance_trends.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path

    def _create_metrics_summary_plot(self) -> str:
        """
        Create metrics summary plot.

        Physical Meaning:
            Creates a plot showing summary metrics
            for all tests.

        Returns:
            Path to generated plot
        """
        if not self.results:
            return None
        
        # Extract summary metrics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r["status"] == "PASS")
        failed_tests = sum(1 for r in self.results.values() if r["status"] == "FAIL")
        
        # Create pie chart
        plt.figure(figsize=(8, 6))
        labels = ['Passed', 'Failed']
        sizes = [passed_tests, failed_tests]
        colors = ['#28a745', '#dc3545']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Test Results Summary')
        
        plot_path = os.path.join(self.output_dir, "metrics_summary.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path

    def generate_summary_report(self) -> str:
        """
        Generate comprehensive summary report.

        Physical Meaning:
            Generates a comprehensive summary report
            including all metrics and visualizations.

        Returns:
            Path to generated summary report
        """
        # Generate HTML report
        html_report_path = self.generate_html_report()
        
        # Create visualizations
        visualization_paths = self.create_visualizations()
        
        # Generate summary text
        summary_path = os.path.join(self.output_dir, "summary_report.txt")
        with open(summary_path, "w") as f:
            f.write("Level A Test Results Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Tests: {len(self.results)}\n")
            f.write(f"Passed: {sum(1 for r in self.results.values() if r['status'] == 'PASS')}\n")
            f.write(f"Failed: {sum(1 for r in self.results.values() if r['status'] == 'FAIL')}\n\n")
            
            f.write("Generated Files:\n")
            f.write(f"- HTML Report: {html_report_path}\n")
            for viz_path in visualization_paths:
                f.write(f"- Visualization: {viz_path}\n")
        
        self.logger.info(f"Generated summary report: {summary_path}")
        return summary_path

    def analyze_trends(self) -> Dict[str, Any]:
        """
        Analyze trends in test results.

        Physical Meaning:
            Analyzes trends in test results to identify
            patterns and potential issues.

        Returns:
            Dictionary containing trend analysis
        """
        if not self.results:
            return {}
        
        # Calculate trends
        execution_times = [r["execution_time"] for r in self.results.values()]
        memory_usage = [r["memory_usage"] for r in self.results.values()]
        
        trends = {
            "execution_time_trend": "increasing" if execution_times[-1] > execution_times[0] else "decreasing",
            "memory_usage_trend": "increasing" if memory_usage[-1] > memory_usage[0] else "decreasing",
            "average_execution_time": np.mean(execution_times),
            "average_memory_usage": np.mean(memory_usage),
            "execution_time_std": np.std(execution_times),
            "memory_usage_std": np.std(memory_usage),
        }
        
        return trends

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.

        Physical Meaning:
            Retrieves comprehensive performance metrics
            for all tests.

        Returns:
            Dictionary containing performance metrics
        """
        if not self.results:
            return {}
        
        execution_times = [r["execution_time"] for r in self.results.values()]
        memory_usage = [r["memory_usage"] for r in self.results.values()]
        
        return {
            "total_execution_time": sum(execution_times),
            "total_memory_usage": sum(memory_usage),
            "average_execution_time": np.mean(execution_times),
            "average_memory_usage": np.mean(memory_usage),
            "max_execution_time": max(execution_times),
            "max_memory_usage": max(memory_usage),
            "min_execution_time": min(execution_times),
            "min_memory_usage": min(memory_usage),
        }


if __name__ == "__main__":
    # Example usage
    reporter = TestReporterAdvanced()
    
    # Record some test results
    reporter.record_test_result(
        test_id="test_001",
        test_name="Basic FFT Test",
        status="PASS",
        metrics={"accuracy": 0.99, "speed": 1.5, "error": 0.01},
        parameters={"N": 64, "L": 1.0},
        execution_time=0.1,
        memory_usage=10.5
    )
    
    # Generate reports and visualizations
    html_report = reporter.generate_html_report()
    visualizations = reporter.create_visualizations()
    summary_report = reporter.generate_summary_report()
    
    print(f"Generated reports: {html_report}, {summary_report}")
    print(f"Generated visualizations: {visualizations}")
    print(f"Trend analysis: {reporter.analyze_trends()}")
    print(f"Performance metrics: {reporter.get_performance_metrics()}")
