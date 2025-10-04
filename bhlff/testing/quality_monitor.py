"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Quality monitoring system for 7D phase field theory experiments.

This module implements comprehensive quality monitoring that tracks both
numerical accuracy and physical validity of experimental results, ensuring
adherence to 7D theory principles and detecting deviations from expected
physical behavior.

Theoretical Background:
    Tracks key physical quantities:
    - Energy conservation: |dE/dt| < ε_energy
    - Virial conditions: |dE/dλ|λ=1| < ε_virial
    - Topological charge: |dB/dt| < ε_topology
    - Passivity: Re Y(ω) ≥ 0 for all ω

Example:
    >>> monitor = QualityMonitor(baseline_metrics, physics_constraints)
    >>> assessment = monitor.check_quality_metrics(test_results)
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
from collections import defaultdict, deque

from .automated_testing import TestResults, TestResult, LevelTestResults


class QualityStatus(Enum):
    """Quality assessment status."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QualityMetrics:
    """Quality metrics for test results."""
    
    # Physics metrics
    energy_conservation: float = 0.0
    virial_conditions: float = 0.0
    topological_charge: float = 0.0
    passivity: float = 0.0
    
    # Numerical metrics
    convergence_rate: float = 0.0
    accuracy: float = 0.0
    stability: float = 0.0
    
    # Spectral metrics
    peak_accuracy: float = 0.0
    quality_factor: float = 0.0
    abcd_accuracy: float = 0.0
    
    # Overall metrics
    overall_score: float = 0.0
    status: QualityStatus = QualityStatus.ACCEPTABLE
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DegradationReport:
    """Report on quality degradation."""
    
    timestamp: datetime = field(default_factory=datetime.now)
    physics_degradation: Dict[str, Any] = field(default_factory=dict)
    numerical_degradation: Dict[str, Any] = field(default_factory=dict)
    spectral_degradation: Dict[str, Any] = field(default_factory=dict)
    convergence_degradation: Dict[str, Any] = field(default_factory=dict)
    overall_severity: AlertSeverity = AlertSeverity.LOW
    recommendations: List[str] = field(default_factory=list)
    
    def add_physics_degradation(self, degradation: Dict[str, Any]) -> None:
        """Add physics degradation analysis."""
        self.physics_degradation.update(degradation)
    
    def add_numerical_degradation(self, degradation: Dict[str, Any]) -> None:
        """Add numerical degradation analysis."""
        self.numerical_degradation.update(degradation)
    
    def add_spectral_degradation(self, degradation: Dict[str, Any]) -> None:
        """Add spectral degradation analysis."""
        self.spectral_degradation.update(degradation)
    
    def add_convergence_degradation(self, degradation: Dict[str, Any]) -> None:
        """Add convergence degradation analysis."""
        self.convergence_degradation.update(degradation)
    
    def set_overall_severity(self, severity: AlertSeverity) -> None:
        """Set overall degradation severity."""
        self.overall_severity = severity


@dataclass
class QualityAlert:
    """Quality degradation alert."""
    
    alert_type: str
    severity: AlertSeverity
    timestamp: datetime
    metric_name: str
    current_value: float
    baseline_value: float
    threshold: float
    physical_interpretation: str
    recommended_actions: List[str]
    theoretical_context: str
    mathematical_expression: str


class PhysicsConstraints:
    """
    Physics constraints for 7D phase field theory validation.
    
    Physical Meaning:
        Defines physical constraints and tolerance values for
        validation of 7D phase field theory principles.
        
    Mathematical Foundation:
        Implements constraints for:
        - Energy conservation: |dE/dt| < ε_energy
        - Virial conditions: |dE/dλ|λ=1| < ε_virial
        - Topological charge: |dB/dt| < ε_topology
        - Passivity: Re Y(ω) ≥ 0 for all ω
    """
    
    def __init__(self, constraint_config: Dict[str, Any]):
        """
        Initialize physics constraints.
        
        Physical Meaning:
            Sets up physical constraint definitions with appropriate
            tolerance values for 7D phase field theory validation.
            
        Args:
            constraint_config (Dict[str, Any]): Constraint configuration.
        """
        self.constraints = constraint_config
        self.energy_tolerance = constraint_config.get('energy_conservation', {}).get('max_relative_error', 1e-6)
        self.virial_tolerance = constraint_config.get('virial_conditions', {}).get('max_relative_error', 1e-6)
        self.topology_tolerance = constraint_config.get('topological_charge', {}).get('max_relative_error', 1e-8)
        self.passivity_tolerance = constraint_config.get('passivity', {}).get('tolerance', 1e-12)
    
    def validate_metrics(self, metrics: Dict[str, Any]) -> bool:
        """
        Validate metrics against physics constraints.
        
        Physical Meaning:
            Validates experimental metrics against fundamental
            physical principles of 7D phase field theory.
            
        Args:
            metrics (Dict[str, Any]): Metrics to validate.
            
        Returns:
            bool: True if all constraints satisfied, False otherwise.
        """
        # Validate energy conservation
        energy_error = metrics.get('energy_conservation', {}).get('relative_error', float('inf'))
        if energy_error > self.energy_tolerance:
            return False
        
        # Validate virial conditions
        virial_error = metrics.get('virial_conditions', {}).get('relative_error', float('inf'))
        if virial_error > self.virial_tolerance:
            return False
        
        # Validate topological charge
        topology_error = metrics.get('topological_charge', {}).get('relative_error', float('inf'))
        if topology_error > self.topology_tolerance:
            return False
        
        # Validate passivity
        min_real_part = metrics.get('passivity', {}).get('min_real_part', float('-inf'))
        if min_real_part < -self.passivity_tolerance:
            return False
        
        return True


class MetricHistory:
    """
    Historical tracking of quality metrics.
    
    Physical Meaning:
        Maintains historical record of quality metrics for trend
        analysis and degradation detection in 7D phase field theory.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metric history.
        
        Physical Meaning:
            Sets up historical tracking with appropriate retention
            period for trend analysis.
            
        Args:
            max_history (int): Maximum number of historical records.
        """
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)
    
    def add_metrics(self, metrics: QualityMetrics) -> None:
        """
        Add metrics to history.
        
        Physical Meaning:
            Records quality metrics for historical analysis
            and trend detection.
            
        Args:
            metrics (QualityMetrics): Quality metrics to record.
        """
        self.metrics_history.append(metrics)
        self.timestamps.append(metrics.timestamp)
    
    def get_recent_metrics(self, days: int = 7) -> List[QualityMetrics]:
        """
        Get recent metrics within specified time window.
        
        Physical Meaning:
            Retrieves recent quality metrics for trend analysis
            and degradation detection.
            
        Args:
            days (int): Number of days to look back.
            
        Returns:
            List[QualityMetrics]: Recent metrics within time window.
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_metrics = []
        
        for i, timestamp in enumerate(self.timestamps):
            if timestamp >= cutoff_time:
                recent_metrics.append(self.metrics_history[i])
        
        return recent_metrics
    
    def get_trend_data(self, metric_name: str, days: int = 30) -> List[float]:
        """
        Get trend data for specific metric.
        
        Physical Meaning:
            Extracts historical values for specific metric to
            analyze trends and detect degradation.
            
        Args:
            metric_name (str): Name of metric to analyze.
            days (int): Number of days to analyze.
            
        Returns:
            List[float]: Historical values for the metric.
        """
        recent_metrics = self.get_recent_metrics(days)
        trend_data = []
        
        for metrics in recent_metrics:
            if hasattr(metrics, metric_name):
                trend_data.append(getattr(metrics, metric_name))
        
        return trend_data


class TrendAnalyzer:
    """
    Trend analysis for quality metrics.
    
    Physical Meaning:
        Analyzes trends in quality metrics to detect degradation
        patterns and predict future quality issues in 7D phase
        field theory experiments.
    """
    
    def __init__(self):
        """Initialize trend analyzer."""
        self.logger = logging.getLogger(__name__)
    
    def analyze_trends(self, historical_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Analyze trends in historical metrics.
        
        Physical Meaning:
            Analyzes trends in physical and numerical metrics
            to detect degradation patterns that could indicate
            quality issues in 7D phase field theory.
            
        Args:
            historical_metrics (List[Dict[str, float]]): Historical metric values.
            
        Returns:
            Dict[str, Any]: Trend analysis results.
        """
        trend_analysis = {
            'overall_trend': 'stable',
            'degrading_metrics': [],
            'improving_metrics': [],
            'stable_metrics': [],
            'trend_scores': {}
        }
        
        if len(historical_metrics) < 2:
            return trend_analysis
        
        # Analyze each metric
        for metric_name in historical_metrics[0].keys():
            values = [m.get(metric_name, 0.0) for m in historical_metrics]
            trend_score = self._calculate_trend_score(values)
            trend_analysis['trend_scores'][metric_name] = trend_score
            
            if trend_score < -0.1:  # Degrading
                trend_analysis['degrading_metrics'].append(metric_name)
            elif trend_score > 0.1:  # Improving
                trend_analysis['improving_metrics'].append(metric_name)
            else:  # Stable
                trend_analysis['stable_metrics'].append(metric_name)
        
        # Determine overall trend
        if len(trend_analysis['degrading_metrics']) > len(trend_analysis['improving_metrics']):
            trend_analysis['overall_trend'] = 'degrading'
        elif len(trend_analysis['improving_metrics']) > len(trend_analysis['degrading_metrics']):
            trend_analysis['overall_trend'] = 'improving'
        else:
            trend_analysis['overall_trend'] = 'stable'
        
        return trend_analysis
    
    def _calculate_trend_score(self, values: List[float]) -> float:
        """
        Calculate trend score for metric values.
        
        Physical Meaning:
            Calculates trend score indicating whether metric
            is improving (positive), degrading (negative), or
            stable (near zero).
            
        Args:
            values (List[float]): Historical values.
            
        Returns:
            float: Trend score (-1 to 1).
        """
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        n = len(values)
        x = np.arange(n)
        y = np.array(values)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize by value range
        value_range = max(values) - min(values)
        if value_range > 0:
            normalized_slope = slope / value_range
        else:
            normalized_slope = 0.0
        
        return np.clip(normalized_slope, -1.0, 1.0)


class AlertSystem:
    """
    Alert system for quality degradation.
    
    Physical Meaning:
        Generates alerts for quality degradation with physics-aware
        interpretation and recommended actions for 7D phase field theory.
    """
    
    def __init__(self, alert_config: Dict[str, Any]):
        """
        Initialize alert system.
        
        Physical Meaning:
            Sets up alert system with physics-aware thresholds
            and notification configuration.
            
        Args:
            alert_config (Dict[str, Any]): Alert configuration.
        """
        self.alert_config = alert_config
        self.logger = logging.getLogger(__name__)
        self.alert_history = []
    
    def generate_alerts(self, degradation_report: DegradationReport) -> List[QualityAlert]:
        """
        Generate alerts for quality degradation.
        
        Physical Meaning:
            Creates alerts for quality degradation with specific
            physical interpretation and recommended actions.
            
        Args:
            degradation_report (DegradationReport): Degradation analysis.
            
        Returns:
            List[QualityAlert]: Generated alerts.
        """
        alerts = []
        
        # Physics-based alerts
        physics_alerts = self._generate_physics_alerts(degradation_report.physics_degradation)
        alerts.extend(physics_alerts)
        
        # Numerical alerts
        numerical_alerts = self._generate_numerical_alerts(degradation_report.numerical_degradation)
        alerts.extend(numerical_alerts)
        
        # Spectral alerts
        spectral_alerts = self._generate_spectral_alerts(degradation_report.spectral_degradation)
        alerts.extend(spectral_alerts)
        
        # Convergence alerts
        convergence_alerts = self._generate_convergence_alerts(degradation_report.convergence_degradation)
        alerts.extend(convergence_alerts)
        
        # Store alerts in history
        self.alert_history.extend(alerts)
        
        return alerts
    
    def _generate_physics_alerts(self, physics_degradation: Dict[str, Any]) -> List[QualityAlert]:
        """Generate physics-specific alerts."""
        alerts = []
        
        # Energy conservation alert
        if physics_degradation.get('energy_conservation', {}).get('severity', 'none') in ['high', 'critical']:
            alert = QualityAlert(
                alert_type="energy_conservation_violation",
                severity=AlertSeverity.CRITICAL,
                timestamp=datetime.now(),
                metric_name="energy_conservation",
                current_value=physics_degradation.get('energy_conservation', {}).get('current_value', 0.0),
                baseline_value=physics_degradation.get('energy_conservation', {}).get('baseline_value', 0.0),
                threshold=1e-6,
                physical_interpretation="Energy conservation violation indicates potential numerical instability or physics violation",
                recommended_actions=[
                    "Check numerical solver stability",
                    "Verify energy conservation implementation",
                    "Review time step and grid resolution"
                ],
                theoretical_context="Energy conservation is fundamental to 7D phase field theory",
                mathematical_expression="|dE/dt| < ε_energy"
            )
            alerts.append(alert)
        
        # Virial condition alert
        if physics_degradation.get('virial_conditions', {}).get('severity', 'none') in ['high', 'critical']:
            alert = QualityAlert(
                alert_type="virial_condition_violation",
                severity=AlertSeverity.HIGH,
                timestamp=datetime.now(),
                metric_name="virial_conditions",
                current_value=physics_degradation.get('virial_conditions', {}).get('current_value', 0.0),
                baseline_value=physics_degradation.get('virial_conditions', {}).get('baseline_value', 0.0),
                threshold=1e-6,
                physical_interpretation="Virial condition violation indicates energy balance issues",
                recommended_actions=[
                    "Check energy balance calculations",
                    "Verify virial condition implementation",
                    "Review boundary conditions"
                ],
                theoretical_context="Virial conditions ensure proper energy distribution in phase fields",
                mathematical_expression="|dE/dλ|λ=1| < ε_virial"
            )
            alerts.append(alert)
        
        return alerts
    
    def _generate_numerical_alerts(self, numerical_degradation: Dict[str, Any]) -> List[QualityAlert]:
        """Generate numerical accuracy alerts."""
        alerts = []
        
        # Convergence alert
        if numerical_degradation.get('convergence_rate', {}).get('severity', 'none') in ['medium', 'high', 'critical']:
            alert = QualityAlert(
                alert_type="convergence_degradation",
                severity=AlertSeverity.MEDIUM,
                timestamp=datetime.now(),
                metric_name="convergence_rate",
                current_value=numerical_degradation.get('convergence_rate', {}).get('current_value', 0.0),
                baseline_value=numerical_degradation.get('convergence_rate', {}).get('baseline_value', 0.0),
                threshold=0.8,
                physical_interpretation="Convergence degradation indicates numerical accuracy issues",
                recommended_actions=[
                    "Check grid resolution",
                    "Review time step size",
                    "Verify numerical scheme stability"
                ],
                theoretical_context="Convergence is essential for accurate physical predictions",
                mathematical_expression="convergence_rate > threshold"
            )
            alerts.append(alert)
        
        return alerts
    
    def _generate_spectral_alerts(self, spectral_degradation: Dict[str, Any]) -> List[QualityAlert]:
        """Generate spectral quality alerts."""
        alerts = []
        
        # Peak accuracy alert
        if spectral_degradation.get('peak_accuracy', {}).get('severity', 'none') in ['medium', 'high', 'critical']:
            alert = QualityAlert(
                alert_type="spectral_peak_degradation",
                severity=AlertSeverity.MEDIUM,
                timestamp=datetime.now(),
                metric_name="peak_accuracy",
                current_value=spectral_degradation.get('peak_accuracy', {}).get('current_value', 0.0),
                baseline_value=spectral_degradation.get('peak_accuracy', {}).get('baseline_value', 0.0),
                threshold=0.95,
                physical_interpretation="Spectral peak degradation indicates resonance analysis issues",
                recommended_actions=[
                    "Check FFT implementation",
                    "Review spectral analysis parameters",
                    "Verify frequency resolution"
                ],
                theoretical_context="Spectral peaks are crucial for resonance analysis in 7D theory",
                mathematical_expression="peak_accuracy > threshold"
            )
            alerts.append(alert)
        
        return alerts
    
    def _generate_convergence_alerts(self, convergence_degradation: Dict[str, Any]) -> List[QualityAlert]:
        """Generate convergence quality alerts."""
        alerts = []
        
        # Overall convergence alert
        if convergence_degradation.get('overall_convergence', {}).get('severity', 'none') in ['high', 'critical']:
            alert = QualityAlert(
                alert_type="overall_convergence_degradation",
                severity=AlertSeverity.HIGH,
                timestamp=datetime.now(),
                metric_name="overall_convergence",
                current_value=convergence_degradation.get('overall_convergence', {}).get('current_value', 0.0),
                baseline_value=convergence_degradation.get('overall_convergence', {}).get('baseline_value', 0.0),
                threshold=0.9,
                physical_interpretation="Overall convergence degradation indicates systematic numerical issues",
                recommended_actions=[
                    "Review numerical scheme",
                    "Check grid and time step",
                    "Verify boundary conditions",
                    "Consider adaptive refinement"
                ],
                theoretical_context="Convergence is fundamental for reliable physical predictions",
                mathematical_expression="overall_convergence > threshold"
            )
            alerts.append(alert)
        
        return alerts


class QualityMonitor:
    """
    Quality monitoring system for 7D phase field theory experiments.
    
    Physical Meaning:
        Monitors both numerical accuracy and physical validity of
        experimental results, ensuring adherence to 7D theory principles
        and detecting deviations from expected physical behavior.
        
    Mathematical Foundation:
        Tracks key physical quantities:
        - Energy conservation: |dE/dt| < ε_energy
        - Virial conditions: |dE/dλ|λ=1| < ε_virial
        - Topological charge: |dB/dt| < ε_topology
        - Passivity: Re Y(ω) ≥ 0 for all ω
    """
    
    def __init__(self, baseline_metrics: Dict[str, Any], physics_constraints: PhysicsConstraints):
        """
        Initialize quality monitor with physics-aware baselines.
        
        Physical Meaning:
            Sets up monitoring with baseline values derived from
            theoretical predictions and validated experimental results.
            
        Args:
            baseline_metrics (Dict[str, Any]): Baseline quality metrics.
            physics_constraints (PhysicsConstraints): Physical constraint definitions.
        """
        self.baseline_metrics = baseline_metrics
        self.physics_constraints = physics_constraints
        self.metric_history = MetricHistory()
        self.alert_system = AlertSystem({})
        self.trend_analyzer = TrendAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    def check_quality_metrics(self, test_results: TestResults) -> QualityMetrics:
        """
        Check quality metrics against physics constraints.
        
        Physical Meaning:
            Validates experimental results against physical principles
            of 7D theory, checking energy conservation, topological
            invariants, and spectral properties.
            
        Args:
            test_results (TestResults): Results from test execution.
            
        Returns:
            QualityMetrics: Comprehensive quality evaluation.
        """
        quality_metrics = QualityMetrics()
        
        # Physics-based quality checks
        physics_quality = self._check_physics_metrics(test_results)
        quality_metrics.energy_conservation = physics_quality.get('energy_conservation', 0.0)
        quality_metrics.virial_conditions = physics_quality.get('virial_conditions', 0.0)
        quality_metrics.topological_charge = physics_quality.get('topological_charge', 0.0)
        quality_metrics.passivity = physics_quality.get('passivity', 0.0)
        
        # Numerical quality checks
        numerical_quality = self._check_numerical_metrics(test_results)
        quality_metrics.convergence_rate = numerical_quality.get('convergence_rate', 0.0)
        quality_metrics.accuracy = numerical_quality.get('accuracy', 0.0)
        quality_metrics.stability = numerical_quality.get('stability', 0.0)
        
        # Spectral quality checks
        spectral_quality = self._check_spectral_metrics(test_results)
        quality_metrics.peak_accuracy = spectral_quality.get('peak_accuracy', 0.0)
        quality_metrics.quality_factor = spectral_quality.get('quality_factor', 0.0)
        quality_metrics.abcd_accuracy = spectral_quality.get('abcd_accuracy', 0.0)
        
        # Overall quality score
        overall_score = self._compute_overall_quality_score(quality_metrics)
        quality_metrics.overall_score = overall_score
        quality_metrics.status = self._determine_quality_status(overall_score)
        
        # Store in history
        self.metric_history.add_metrics(quality_metrics)
        
        return quality_metrics
    
    def detect_quality_degradation(self, current_metrics: Dict[str, float], 
                                 historical_metrics: List[Dict[str, float]]) -> DegradationReport:
        """
        Detect quality degradation with physics-aware analysis.
        
        Physical Meaning:
            Identifies degradation in physical quantities that could
            indicate violations of conservation laws or theoretical principles.
            
        Args:
            current_metrics (Dict[str, float]): Current quality metrics.
            historical_metrics (List[Dict[str, float]]): Historical metric values.
            
        Returns:
            DegradationReport: Analysis of quality degradation.
        """
        report = DegradationReport()
        
        # Physics-based degradation detection
        physics_degradation = self._detect_physics_degradation(current_metrics, historical_metrics)
        report.add_physics_degradation(physics_degradation)
        
        # Numerical degradation detection
        numerical_degradation = self._detect_numerical_degradation(current_metrics, historical_metrics)
        report.add_numerical_degradation(numerical_degradation)
        
        # Spectral degradation detection
        spectral_degradation = self._detect_spectral_degradation(current_metrics, historical_metrics)
        report.add_spectral_degradation(spectral_degradation)
        
        # Convergence degradation detection
        convergence_degradation = self._detect_convergence_degradation(current_metrics, historical_metrics)
        report.add_convergence_degradation(convergence_degradation)
        
        # Overall severity assessment
        severity = self._assess_degradation_severity(report)
        report.set_overall_severity(severity)
        
        return report
    
    def generate_quality_alerts(self, degraded_metrics: DegradationReport) -> List[QualityAlert]:
        """
        Generate quality alerts with physics context.
        
        Physical Meaning:
            Creates alerts for quality degradation with specific
            physical interpretation and recommended actions.
            
        Args:
            degraded_metrics (DegradationReport): Degradation analysis.
            
        Returns:
            List[QualityAlert]: Generated alerts with physics context.
        """
        return self.alert_system.generate_alerts(degraded_metrics)
    
    def update_baseline_metrics(self, new_metrics: Dict[str, Any]) -> None:
        """
        Update baseline metrics with physics validation.
        
        Physical Meaning:
            Updates baseline values only if they maintain physical
            validity and improve upon existing baselines.
            
        Args:
            new_metrics (Dict[str, Any]): New metric values to consider.
        """
        # Validate new metrics against physics constraints
        if self.physics_constraints.validate_metrics(new_metrics):
            # Update baselines if improvement is significant
            if self._is_significant_improvement(new_metrics):
                self.baseline_metrics.update(new_metrics)
                self.logger.info("Baseline metrics updated with physics validation")
        else:
            self.logger.warning("New metrics failed physics validation, not updating baselines")
    
    def _check_physics_metrics(self, test_results: TestResults) -> Dict[str, float]:
        """Check physics-based quality metrics."""
        physics_metrics = {}
        
        # Calculate energy conservation score
        energy_scores = []
        for level_results in test_results.level_results.values():
            for test_result in level_results.test_results:
                energy_validation = test_result.physics_validation.get('energy_conservation', {})
                energy_error = energy_validation.get('relative_error', 1.0)
                energy_score = max(0.0, 1.0 - energy_error / self.physics_constraints.energy_tolerance)
                energy_scores.append(energy_score)
        
        physics_metrics['energy_conservation'] = np.mean(energy_scores) if energy_scores else 0.0
        
        # Calculate virial conditions score
        virial_scores = []
        for level_results in test_results.level_results.values():
            for test_result in level_results.test_results:
                virial_validation = test_result.physics_validation.get('virial_conditions', {})
                virial_error = virial_validation.get('relative_error', 1.0)
                virial_score = max(0.0, 1.0 - virial_error / self.physics_constraints.virial_tolerance)
                virial_scores.append(virial_score)
        
        physics_metrics['virial_conditions'] = np.mean(virial_scores) if virial_scores else 0.0
        
        # Calculate topological charge score
        topology_scores = []
        for level_results in test_results.level_results.values():
            for test_result in level_results.test_results:
                topology_validation = test_result.physics_validation.get('topological_charge', {})
                topology_error = topology_validation.get('relative_error', 1.0)
                topology_score = max(0.0, 1.0 - topology_error / self.physics_constraints.topology_tolerance)
                topology_scores.append(topology_score)
        
        physics_metrics['topological_charge'] = np.mean(topology_scores) if topology_scores else 0.0
        
        # Calculate passivity score
        passivity_scores = []
        for level_results in test_results.level_results.values():
            for test_result in level_results.test_results:
                passivity_validation = test_result.physics_validation.get('passivity', {})
                min_real_part = passivity_validation.get('min_real_part', -1.0)
                passivity_score = max(0.0, min(1.0, (min_real_part + self.physics_constraints.passivity_tolerance) / self.physics_constraints.passivity_tolerance))
                passivity_scores.append(passivity_score)
        
        physics_metrics['passivity'] = np.mean(passivity_scores) if passivity_scores else 0.0
        
        return physics_metrics
    
    def _check_numerical_metrics(self, test_results: TestResults) -> Dict[str, float]:
        """Check numerical quality metrics."""
        numerical_metrics = {}
        
        # Calculate convergence rate
        convergence_rates = []
        for level_results in test_results.level_results.values():
            for test_result in level_results.test_results:
                numerical_metrics_data = test_result.numerical_metrics
                convergence_rate = numerical_metrics_data.get('convergence_rate', 0.0)
                convergence_rates.append(convergence_rate)
        
        numerical_metrics['convergence_rate'] = np.mean(convergence_rates) if convergence_rates else 0.0
        
        # Calculate accuracy
        accuracy_scores = []
        for level_results in test_results.level_results.values():
            for test_result in level_results.test_results:
                numerical_metrics_data = test_result.numerical_metrics
                accuracy = numerical_metrics_data.get('accuracy', 0.0)
                accuracy_scores.append(accuracy)
        
        numerical_metrics['accuracy'] = np.mean(accuracy_scores) if accuracy_scores else 0.0
        
        # Calculate stability
        stability_scores = []
        for level_results in test_results.level_results.values():
            for test_result in level_results.test_results:
                numerical_metrics_data = test_result.numerical_metrics
                stability = numerical_metrics_data.get('stability', 0.0)
                stability_scores.append(stability)
        
        numerical_metrics['stability'] = np.mean(stability_scores) if stability_scores else 0.0
        
        return numerical_metrics
    
    def _check_spectral_metrics(self, test_results: TestResults) -> Dict[str, float]:
        """Check spectral quality metrics."""
        spectral_metrics = {}
        
        # Calculate peak accuracy
        peak_accuracies = []
        for level_results in test_results.level_results.values():
            for test_result in level_results.test_results:
                numerical_metrics_data = test_result.numerical_metrics
                peak_accuracy = numerical_metrics_data.get('peak_accuracy', 0.0)
                peak_accuracies.append(peak_accuracy)
        
        spectral_metrics['peak_accuracy'] = np.mean(peak_accuracies) if peak_accuracies else 0.0
        
        # Calculate quality factor
        quality_factors = []
        for level_results in test_results.level_results.values():
            for test_result in level_results.test_results:
                numerical_metrics_data = test_result.numerical_metrics
                quality_factor = numerical_metrics_data.get('quality_factor', 0.0)
                quality_factors.append(quality_factor)
        
        spectral_metrics['quality_factor'] = np.mean(quality_factors) if quality_factors else 0.0
        
        # Calculate ABCD accuracy
        abcd_accuracies = []
        for level_results in test_results.level_results.values():
            for test_result in level_results.test_results:
                numerical_metrics_data = test_result.numerical_metrics
                abcd_accuracy = numerical_metrics_data.get('abcd_accuracy', 0.0)
                abcd_accuracies.append(abcd_accuracy)
        
        spectral_metrics['abcd_accuracy'] = np.mean(abcd_accuracies) if abcd_accuracies else 0.0
        
        return spectral_metrics
    
    def _compute_overall_quality_score(self, quality_metrics: QualityMetrics) -> float:
        """Compute overall quality score."""
        # Weight different metrics by importance
        weights = {
            'energy_conservation': 0.25,
            'virial_conditions': 0.20,
            'topological_charge': 0.20,
            'passivity': 0.15,
            'convergence_rate': 0.10,
            'accuracy': 0.05,
            'stability': 0.05
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if hasattr(quality_metrics, metric):
                value = getattr(quality_metrics, metric)
                weighted_score += value * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_quality_status(self, overall_score: float) -> QualityStatus:
        """Determine quality status from overall score."""
        if overall_score >= 0.95:
            return QualityStatus.EXCELLENT
        elif overall_score >= 0.85:
            return QualityStatus.GOOD
        elif overall_score >= 0.70:
            return QualityStatus.ACCEPTABLE
        elif overall_score >= 0.50:
            return QualityStatus.DEGRADED
        else:
            return QualityStatus.CRITICAL
    
    def _detect_physics_degradation(self, current_metrics: Dict[str, float], 
                                  historical_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """Detect physics-specific degradation."""
        degradation = {}
        
        # Check energy conservation degradation
        if 'energy_conservation' in current_metrics:
            current_energy = current_metrics['energy_conservation']
            baseline_energy = self.baseline_metrics.get('energy_conservation', 1.0)
            
            if current_energy < baseline_energy * 0.9:  # 10% degradation
                degradation['energy_conservation'] = {
                    'current_value': current_energy,
                    'baseline_value': baseline_energy,
                    'degradation_percent': (baseline_energy - current_energy) / baseline_energy * 100,
                    'severity': 'high' if current_energy < baseline_energy * 0.8 else 'medium'
                }
        
        return degradation
    
    def _detect_numerical_degradation(self, current_metrics: Dict[str, float], 
                                    historical_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """Detect numerical accuracy degradation."""
        degradation = {}
        
        # Check convergence rate degradation
        if 'convergence_rate' in current_metrics:
            current_convergence = current_metrics['convergence_rate']
            baseline_convergence = self.baseline_metrics.get('convergence_rate', 1.0)
            
            if current_convergence < baseline_convergence * 0.9:
                degradation['convergence_rate'] = {
                    'current_value': current_convergence,
                    'baseline_value': baseline_convergence,
                    'degradation_percent': (baseline_convergence - current_convergence) / baseline_convergence * 100,
                    'severity': 'high' if current_convergence < baseline_convergence * 0.8 else 'medium'
                }
        
        return degradation
    
    def _detect_spectral_degradation(self, current_metrics: Dict[str, float], 
                                   historical_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """Detect spectral quality degradation."""
        degradation = {}
        
        # Check peak accuracy degradation
        if 'peak_accuracy' in current_metrics:
            current_peak = current_metrics['peak_accuracy']
            baseline_peak = self.baseline_metrics.get('peak_accuracy', 1.0)
            
            if current_peak < baseline_peak * 0.9:
                degradation['peak_accuracy'] = {
                    'current_value': current_peak,
                    'baseline_value': baseline_peak,
                    'degradation_percent': (baseline_peak - current_peak) / baseline_peak * 100,
                    'severity': 'high' if current_peak < baseline_peak * 0.8 else 'medium'
                }
        
        return degradation
    
    def _detect_convergence_degradation(self, current_metrics: Dict[str, float], 
                                      historical_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """Detect convergence quality degradation."""
        degradation = {}
        
        # Check overall convergence degradation
        convergence_metrics = ['convergence_rate', 'accuracy', 'stability']
        convergence_scores = [current_metrics.get(metric, 0.0) for metric in convergence_metrics]
        overall_convergence = np.mean(convergence_scores)
        
        baseline_convergence = np.mean([self.baseline_metrics.get(metric, 1.0) for metric in convergence_metrics])
        
        if overall_convergence < baseline_convergence * 0.9:
            degradation['overall_convergence'] = {
                'current_value': overall_convergence,
                'baseline_value': baseline_convergence,
                'degradation_percent': (baseline_convergence - overall_convergence) / baseline_convergence * 100,
                'severity': 'high' if overall_convergence < baseline_convergence * 0.8 else 'medium'
            }
        
        return degradation
    
    def _assess_degradation_severity(self, report: DegradationReport) -> AlertSeverity:
        """Assess overall degradation severity."""
        severities = []
        
        # Collect severities from all degradation types
        for degradation_dict in [report.physics_degradation, report.numerical_degradation, 
                               report.spectral_degradation, report.convergence_degradation]:
            for degradation in degradation_dict.values():
                if isinstance(degradation, dict) and 'severity' in degradation:
                    severities.append(degradation['severity'])
        
        if not severities:
            return AlertSeverity.LOW
        
        # Determine overall severity
        if 'critical' in severities:
            return AlertSeverity.CRITICAL
        elif 'high' in severities:
            return AlertSeverity.HIGH
        elif 'medium' in severities:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def _is_significant_improvement(self, new_metrics: Dict[str, Any]) -> bool:
        """Check if new metrics represent significant improvement."""
        improvement_threshold = 0.05  # 5% improvement
        
        for metric_name, new_value in new_metrics.items():
            if metric_name in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric_name]
                improvement = (new_value - baseline_value) / baseline_value
                if improvement > improvement_threshold:
                    return True
        
        return False
