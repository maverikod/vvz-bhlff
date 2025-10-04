"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Visualization tools for Level B fundamental properties analysis.

This module provides comprehensive visualization capabilities for Level B
analysis results, including power law fits, node analysis, topological
charge visualization, and zone separation maps.

Theoretical Background:
    Visualization helps understand the fundamental properties of the phase
    field by showing radial profiles, power law behavior, zone structure,
    and topological characteristics in an intuitive way.

Example:
    >>> visualizer = LevelBVisualizer()
    >>> visualizer.create_comprehensive_report(all_results)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Dict, Any, List, Tuple
from pathlib import Path


class LevelBVisualizer:
    """
    Visualization tools for Level B analysis results.
    
    Physical Meaning:
        Creates comprehensive visualizations of Level B analysis results,
        helping to understand the fundamental properties of the phase field
        including power law behavior, zone structure, and topological characteristics.
        
    Mathematical Foundation:
        Visualizations are based on the theoretical predictions of the
        Riesz operator L_β = μ(-Δ)^β + λ and its spectral properties.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize Level B visualizer.
        
        Args:
            style (str): Matplotlib style for plots
        """
        plt.style.use(style)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def create_comprehensive_report(self, results: Dict[str, Any], 
                                  output_dir: str = "level_b_analysis") -> None:
        """
        Create comprehensive visualization report.
        
        Physical Meaning:
            Generates a complete set of visualizations for all Level B
            analysis results, providing a comprehensive view of the
            fundamental properties of the phase field.
            
        Args:
            results (Dict[str, Any]): All Level B analysis results
            output_dir (str): Directory to save visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create individual visualizations
        if 'test_B1_power_law_tail' in results:
            self._visualize_power_law_analysis(
                results['test_B1_power_law_tail'], 
                output_path / "power_law_analysis.png"
            )
        
        if 'test_B2_no_spherical_nodes' in results:
            self._visualize_node_analysis(
                results['test_B2_no_spherical_nodes'],
                output_path / "node_analysis.png"
            )
        
        if 'test_B3_topological_charge' in results:
            self._visualize_topological_analysis(
                results['test_B3_topological_charge'],
                output_path / "topological_analysis.png"
            )
        
        if 'test_B4_zone_separation' in results:
            self._visualize_zone_analysis(
                results['test_B4_zone_separation'],
                output_path / "zone_analysis.png"
            )
        
        # Create summary dashboard
        self._create_summary_dashboard(results, output_path / "summary_dashboard.png")
        
        print(f"Visualizations saved to {output_path}")
    
    def _visualize_power_law_analysis(self, result: Dict[str, Any], output_path: Path) -> None:
        """Visualize power law analysis results."""
        if not result.get('passed', False):
            return
        
        analysis_result = result.get('analysis_result', {})
        if not analysis_result:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Radial profile
        radial_profile = analysis_result.get('radial_profile', {})
        if radial_profile:
            ax1.plot(radial_profile['r'], radial_profile['A'], 'b-', linewidth=2, label='Radial Profile')
            ax1.set_xlabel('Radius r')
            ax1.set_ylabel('Amplitude A(r)')
            ax1.set_title('Radial Profile')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # Plot 2: Log-log fit
        tail_data = analysis_result.get('tail_data', {})
        if tail_data and 'log_r' in tail_data and 'log_A' in tail_data:
            ax2.scatter(tail_data['log_r'], tail_data['log_A'], alpha=0.6, s=20, label='Data')
            
            # Plot fitted line
            if len(tail_data['log_r']) > 1:
                slope = analysis_result.get('slope', 0)
                log_r_fit = np.linspace(tail_data['log_r'].min(), tail_data['log_r'].max(), 100)
                log_A_fit = slope * log_r_fit
                ax2.plot(log_r_fit, log_A_fit, 'r-', linewidth=2, 
                        label=f'Fit: slope={slope:.3f}')
            
            ax2.set_xlabel('log(r)')
            ax2.set_ylabel('log(A)')
            ax2.set_title(f'Power Law Fit (R²={analysis_result.get("r_squared", 0):.3f})')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # Plot 3: Error analysis
        theoretical_slope = analysis_result.get('theoretical_slope', 0)
        slope = analysis_result.get('slope', 0)
        relative_error = analysis_result.get('relative_error', 0)
        
        ax3.bar(['Theoretical', 'Measured'], [theoretical_slope, slope], 
                color=['blue', 'red'], alpha=0.7)
        ax3.set_ylabel('Slope')
        ax3.set_title(f'Slope Comparison (Error: {relative_error:.1%})')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Quality metrics
        metrics = ['R²', 'Log Range', 'Error']
        values = [
            analysis_result.get('r_squared', 0),
            analysis_result.get('log_range', 0),
            1 - analysis_result.get('relative_error', 1)
        ]
        
        ax4.bar(metrics, values, color=['green', 'orange', 'purple'], alpha=0.7)
        ax4.set_ylabel('Value')
        ax4.set_title('Quality Metrics')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_node_analysis(self, result: Dict[str, Any], output_path: Path) -> None:
        """Visualize node analysis results."""
        if not result.get('passed', False):
            return
        
        analysis_result = result.get('analysis_result', {})
        if not analysis_result:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Radial profile
        radial_profile = analysis_result.get('radial_profile', {})
        if radial_profile:
            ax1.plot(radial_profile['r'], radial_profile['A'], 'b-', linewidth=2, label='Radial Profile')
            ax1.set_xlabel('Radius r')
            ax1.set_ylabel('Amplitude A(r)')
            ax1.set_title('Radial Profile')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # Plot 2: Radial derivative
        radial_derivative = analysis_result.get('radial_derivative', [])
        if len(radial_derivative) > 0 and radial_profile:
            ax2.plot(radial_profile['r'], radial_derivative, 'r-', linewidth=2, label='dA/dr')
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Radius r')
            ax2.set_ylabel('dA/dr')
            ax2.set_title('Radial Derivative')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # Plot 3: Sign changes analysis
        sign_changes = analysis_result.get('sign_changes', 0)
        zeros = analysis_result.get('zeros', [])
        
        ax3.bar(['Sign Changes', 'Zeros Found'], [sign_changes, len(zeros)], 
                color=['red', 'blue'], alpha=0.7)
        ax3.set_ylabel('Count')
        ax3.set_title('Node Analysis')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Quality assessment
        is_monotonic = analysis_result.get('is_monotonic', False)
        periodic_zeros = analysis_result.get('periodic_zeros', False)
        
        quality_metrics = ['Monotonic', 'No Periodic Zeros', 'Low Sign Changes']
        quality_values = [is_monotonic, not periodic_zeros, sign_changes <= 1]
        
        colors = ['green' if v else 'red' for v in quality_values]
        ax4.bar(quality_metrics, quality_values, color=colors, alpha=0.7)
        ax4.set_ylabel('Pass/Fail')
        ax4.set_title('Quality Assessment')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_topological_analysis(self, result: Dict[str, Any], output_path: Path) -> None:
        """Visualize topological charge analysis results."""
        if not result.get('passed', False):
            return
        
        charge_result = result.get('charge_result', {})
        if not charge_result:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Charge value
        charge = charge_result.get('charge', 0)
        integer_charge = charge_result.get('integer_charge', 0)
        error = charge_result.get('error', 0)
        
        ax1.bar(['Measured', 'Integer'], [charge, integer_charge], 
                color=['blue', 'red'], alpha=0.7)
        ax1.set_ylabel('Charge Value')
        ax1.set_title(f'Topological Charge (Error: {error:.3f})')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Error analysis
        ax2.bar(['Error'], [error], color='orange', alpha=0.7)
        ax2.axhline(y=0.01, color='red', linestyle='--', label='Threshold (1%)')
        ax2.set_ylabel('Error')
        ax2.set_title('Charge Error Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Contour visualization
        contour_points = charge_result.get('contour_points', [])
        if contour_points:
            contour_array = np.array(contour_points)
            ax3.scatter(contour_array[:, 0], contour_array[:, 1], 
                       c=range(len(contour_array)), cmap='viridis', s=50)
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
            ax3.set_title('Integration Contour')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Quality metrics
        integration_radius = charge_result.get('integration_radius', 0)
        contour_count = len(contour_points)
        
        metrics = ['Integration Radius', 'Contour Points', 'Error < 1%']
        values = [integration_radius, contour_count, error < 0.01]
        
        ax4.bar(metrics, values, color=['blue', 'green', 'red'], alpha=0.7)
        ax4.set_ylabel('Value')
        ax4.set_title('Quality Metrics')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_zone_analysis(self, result: Dict[str, Any], output_path: Path) -> None:
        """Visualize zone separation analysis results."""
        if not result.get('passed', False):
            return
        
        zone_result = result.get('zone_result', {})
        if not zone_result:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Zone radii
        r_core = zone_result.get('r_core', 0)
        r_tail = zone_result.get('r_tail', 0)
        r_transition = zone_result.get('r_transition', 0)
        
        zones = ['Core', 'Transition', 'Tail']
        radii = [r_core, r_transition, r_tail]
        colors = ['red', 'yellow', 'blue']
        
        ax1.bar(zones, radii, color=colors, alpha=0.7)
        ax1.set_ylabel('Radius')
        ax1.set_title('Zone Radii')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Zone fractions
        zone_stats = zone_result.get('zone_stats', {})
        core_fraction = zone_stats.get('core', {}).get('volume_fraction', 0)
        tail_fraction = zone_stats.get('tail', {}).get('volume_fraction', 0)
        transition_fraction = zone_stats.get('transition', {}).get('volume_fraction', 0)
        
        fractions = [core_fraction, transition_fraction, tail_fraction]
        ax2.pie(fractions, labels=zones, colors=colors, autopct='%1.1f%%')
        ax2.set_title('Zone Volume Fractions')
        
        # Plot 3: Zone amplitudes
        core_amplitude = zone_stats.get('core', {}).get('mean_amplitude', 0)
        tail_amplitude = zone_stats.get('tail', {}).get('mean_amplitude', 0)
        transition_amplitude = zone_stats.get('transition', {}).get('mean_amplitude', 0)
        
        amplitudes = [core_amplitude, transition_amplitude, tail_amplitude]
        ax3.bar(zones, amplitudes, color=colors, alpha=0.7)
        ax3.set_ylabel('Mean Amplitude')
        ax3.set_title('Zone Amplitudes')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Quality metrics
        quality_metrics = zone_result.get('quality_metrics', {})
        overall_score = quality_metrics.get('overall_score', 0)
        amplitude_ordering = quality_metrics.get('amplitude_ordering', False)
        zone_balance = quality_metrics.get('zone_balance', False)
        
        metrics = ['Overall Score', 'Amplitude Ordering', 'Zone Balance']
        values = [overall_score, amplitude_ordering, zone_balance]
        
        ax4.bar(metrics, values, color=['green', 'blue', 'orange'], alpha=0.7)
        ax4.set_ylabel('Value')
        ax4.set_title('Quality Metrics')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_summary_dashboard(self, results: Dict[str, Any], output_path: Path) -> None:
        """Create summary dashboard for all results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Test results overview
        test_names = list(results.keys())
        test_passed = [results[name].get('passed', False) for name in test_names]
        
        colors = ['green' if passed else 'red' for passed in test_passed]
        ax1.bar(range(len(test_names)), test_passed, color=colors, alpha=0.7)
        ax1.set_xticks(range(len(test_names)))
        ax1.set_xticklabels([name.replace('test_', '') for name in test_names], rotation=45)
        ax1.set_ylabel('Pass/Fail')
        ax1.set_title('Level B Test Results Overview')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Power law analysis summary
        if 'test_B1_power_law_tail' in results and results['test_B1_power_law_tail'].get('passed'):
            analysis = results['test_B1_power_law_tail'].get('analysis_result', {})
            ax2.bar(['R²', 'Error < 5%', 'Range > 1.5'], 
                   [analysis.get('r_squared', 0), 
                    analysis.get('relative_error', 1) < 0.05,
                    analysis.get('log_range', 0) > 1.5],
                   color=['blue', 'green', 'orange'], alpha=0.7)
            ax2.set_ylabel('Value')
            ax2.set_title('Power Law Analysis')
            ax2.set_ylim(0, 1)
        
        # Plot 3: Node analysis summary
        if 'test_B2_no_spherical_nodes' in results and results['test_B2_no_spherical_nodes'].get('passed'):
            analysis = results['test_B2_no_spherical_nodes'].get('analysis_result', {})
            ax3.bar(['Sign Changes ≤ 1', 'No Periodic Zeros', 'Monotonic'], 
                   [analysis.get('sign_changes', 1) <= 1,
                    not analysis.get('periodic_zeros', True),
                    analysis.get('is_monotonic', False)],
                   color=['red', 'blue', 'green'], alpha=0.7)
            ax3.set_ylabel('Pass/Fail')
            ax3.set_title('Node Analysis')
            ax3.set_ylim(0, 1)
        
        # Plot 4: Overall quality assessment
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result.get('passed', False))
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        ax4.pie([passed_tests, total_tests - passed_tests], 
                labels=['Passed', 'Failed'], 
                colors=['green', 'red'], 
                autopct='%1.1f%%')
        ax4.set_title(f'Overall Success Rate: {success_rate:.1%}')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_3d_visualization(self, field: np.ndarray, center: List[float], 
                              output_path: str = "3d_field_visualization.png") -> None:
        """
        Create 3D visualization of the field.
        
        Physical Meaning:
            Creates 3D visualization of the phase field showing
            the spatial structure and amplitude distribution.
            
        Args:
            field (np.ndarray): 3D field array
            center (List[float]): Center coordinates
            output_path (str): Path to save the plot
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create coordinate grids
        x = np.arange(field.shape[0])
        y = np.arange(field.shape[1])
        z = np.arange(field.shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Get field amplitude
        amplitude = np.abs(field)
        
        # Create 3D scatter plot
        scatter = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), 
                           c=amplitude.flatten(), cmap='viridis', alpha=0.6)
        
        # Mark the center
        ax.scatter(center[0], center[1], center[2], 
                  c='red', s=100, marker='*', label='Center')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('3D Field Visualization')
        
        plt.colorbar(scatter, label='Amplitude')
        plt.legend()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
