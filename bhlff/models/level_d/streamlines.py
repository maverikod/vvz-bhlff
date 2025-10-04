"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Phase streamline analysis for Level D models.

This module implements phase streamline analysis for tracing
phase gradient flow patterns around defects and singularities
in the phase field.

Physical Meaning:
    Phase streamlines represent the flow patterns of phase
    information in the field, revealing the topological
    structure of phase flow around defects and singularities.
    These streamlines are analogous to magnetic field lines
    in electromagnetism but for phase gradients.

Mathematical Foundation:
    - Phase field: φ(x) = arg[a(x)]
    - Phase gradient: ∇φ = ∇ arg[a(x)]
    - Streamlines: dx/dt = ∇φ(x)

Example:
    >>> from bhlff.models.level_d.streamlines import StreamlineAnalyzer
    >>> analyzer = StreamlineAnalyzer(domain, parameters)
    >>> results = analyzer.trace_phase_streamlines(field, center)
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from scipy.integrate import odeint

from bhlff.models.base.abstract_models import AbstractLevelModels


class StreamlineAnalyzer:
    """
    Analyzer for phase streamline patterns.
    
    Physical Meaning:
        Analyzes phase gradient flow patterns to understand
        the topological structure of phase flow around
        defects and singularities.
        
    Mathematical Foundation:
        Computes phase gradients and traces streamlines
        that are tangent to the gradient field at each point.
    """
    
    def __init__(self, domain: 'Domain', parameters: Dict[str, Any]):
        """
        Initialize streamline analyzer.
        
        Physical Meaning:
            Sets up the streamline analysis system for
            tracing phase gradient flow patterns.
            
        Args:
            domain (Domain): Computational domain
            parameters (Dict): Analysis parameters
        """
        self.domain = domain
        self.parameters = parameters
        self.logger = logging.getLogger(__name__)
        
        # Initialize analysis tools
        self._gradient_computer = GradientComputer(domain)
        self._streamline_tracer = StreamlineTracer(domain, parameters)
        self._topology_analyzer = TopologyAnalyzer(domain)
        
        self.logger.info("Streamline analyzer initialized")
    
    def trace_phase_streamlines(self, field: np.ndarray, center: Tuple[float, ...]) -> Dict[str, Any]:
        """
        Trace phase streamlines around defects.
        
        Physical Meaning:
            Computes streamlines of the phase gradient field,
            revealing the topological structure of phase flow
            around defects and singularities.
            
        Mathematical Foundation:
            Integrates the phase gradient field to find
            streamlines that are tangent to the gradient
            at each point: dx/dt = ∇φ(x)
            
        Args:
            field (np.ndarray): Input field
            center (Tuple): Center point for streamline tracing
            
        Returns:
            Dict: Streamline analysis results including:
                - phase: Field phase
                - phase_gradient: Phase gradient field
                - streamlines: Computed streamlines
                - topology: Topological analysis of streamlines
        """
        self.logger.info("Tracing phase streamlines")
        
        # Compute field phase
        phase = np.angle(field)
        
        # Compute phase gradient
        phase_gradient = self._gradient_computer.compute_phase_gradient(phase)
        
        # Trace streamlines
        streamlines = self._streamline_tracer.trace_streamlines(
            phase_gradient, center
        )
        
        # Analyze topology
        topology = self._topology_analyzer.analyze_streamline_topology(streamlines)
        
        results = {
            'phase': phase,
            'phase_gradient': phase_gradient,
            'streamlines': streamlines,
            'topology': topology
        }
        
        self.logger.info("Phase streamline analysis completed")
        return results
    
    def analyze_streamlines(self, field: np.ndarray, resolution: float = 1.0) -> Dict[str, Any]:
        """
        Analyze streamline patterns in the field.
        
        Physical Meaning:
            Computes field gradients to analyze streamline patterns
            and flow characteristics in the field, providing
            insights into the field dynamics and structure.
            
        Mathematical Foundation:
            Computes divergence: ∇·v = ∂v_x/∂x + ∂v_y/∂y + ∂v_z/∂z
            Computes curl: ∇×v = (∂v_z/∂y - ∂v_y/∂z, ∂v_x/∂z - ∂v_z/∂x, ∂v_y/∂x - ∂v_x/∂y)
            
        Args:
            field (np.ndarray): Input field
            resolution (float): Resolution for streamline analysis
            
        Returns:
            Dict: Streamline analysis including:
                - divergence_max: Maximum divergence value
                - divergence_mean: Mean divergence value
                - curl_max: Maximum curl magnitude
                - curl_mean: Mean curl magnitude
                - streamline_density: Density of streamlines
        """
        self.logger.info("Analyzing streamline patterns")
        
        # Compute field gradients
        gradients = self._gradient_computer.compute_field_gradients(field)
        
        # Compute divergence
        divergence = self._gradient_computer.compute_divergence(gradients)
        
        # Compute curl
        curl = self._gradient_computer.compute_curl(gradients)
        
        # Analyze streamline density
        streamline_density = self._compute_streamline_density(field, resolution)
        
        results = {
            'divergence_max': float(np.max(divergence)),
            'divergence_mean': float(np.mean(divergence)),
            'curl_max': float(np.max(np.linalg.norm(curl, axis=-1))),
            'curl_mean': float(np.mean(np.linalg.norm(curl, axis=-1))),
            'streamline_density': float(streamline_density)
        }
        
        self.logger.info("Streamline pattern analysis completed")
        return results


class GradientComputer:
    """Compute field gradients and phase gradients."""
    
    def __init__(self, domain: 'Domain'):
        """Initialize gradient computer."""
        self.domain = domain
    
    def compute_phase_gradient(self, phase: np.ndarray) -> np.ndarray:
        """
        Compute phase gradient field.
        
        Physical Meaning:
            Computes the gradient of the phase field,
            representing the local direction of phase
            flow and its magnitude.
            
        Mathematical Foundation:
            ∇φ = (∂φ/∂x, ∂φ/∂y, ∂φ/∂z)
            
        Args:
            phase (np.ndarray): Phase field
            
        Returns:
            np.ndarray: Phase gradient field
        """
        if len(phase.shape) == 3:
            # 3D gradient
            grad_x = np.gradient(phase, axis=0)
            grad_y = np.gradient(phase, axis=1)
            grad_z = np.gradient(phase, axis=2)
            gradient = np.stack([grad_x, grad_y, grad_z], axis=-1)
        elif len(phase.shape) == 2:
            # 2D gradient
            grad_x = np.gradient(phase, axis=0)
            grad_y = np.gradient(phase, axis=1)
            gradient = np.stack([grad_x, grad_y], axis=-1)
        else:
            # 1D gradient
            gradient = np.gradient(phase)
            gradient = np.expand_dims(gradient, axis=-1)
        
        return gradient
    
    def compute_field_gradients(self, field: np.ndarray) -> np.ndarray:
        """
        Compute field gradients.
        
        Physical Meaning:
            Computes the gradient of the field in all
            spatial dimensions.
            
        Args:
            field (np.ndarray): Input field
            
        Returns:
            np.ndarray: Field gradients
        """
        if len(field.shape) == 3:
            # 3D gradient
            grad_x = np.gradient(field, axis=0)
            grad_y = np.gradient(field, axis=1)
            grad_z = np.gradient(field, axis=2)
            gradients = np.stack([grad_x, grad_y, grad_z], axis=-1)
        elif len(field.shape) == 2:
            # 2D gradient
            grad_x = np.gradient(field, axis=0)
            grad_y = np.gradient(field, axis=1)
            gradients = np.stack([grad_x, grad_y], axis=-1)
        else:
            # 1D gradient
            gradients = np.gradient(field)
            gradients = np.expand_dims(gradients, axis=-1)
        
        return gradients
    
    def compute_divergence(self, gradients: np.ndarray) -> np.ndarray:
        """
        Compute divergence of gradient field.
        
        Physical Meaning:
            Computes the divergence of the gradient field,
            representing sources and sinks in the flow.
            
        Mathematical Foundation:
            ∇·v = ∂v_x/∂x + ∂v_y/∂y + ∂v_z/∂z
            
        Args:
            gradients (np.ndarray): Gradient field
            
        Returns:
            np.ndarray: Divergence field
        """
        if gradients.shape[-1] == 3:
            # 3D divergence
            div_x = np.gradient(gradients[..., 0], axis=0)
            div_y = np.gradient(gradients[..., 1], axis=1)
            div_z = np.gradient(gradients[..., 2], axis=2)
            divergence = div_x + div_y + div_z
        elif gradients.shape[-1] == 2:
            # 2D divergence
            div_x = np.gradient(gradients[..., 0], axis=0)
            div_y = np.gradient(gradients[..., 1], axis=1)
            divergence = div_x + div_y
        else:
            # 1D divergence
            divergence = np.gradient(gradients[..., 0], axis=0)
        
        return divergence
    
    def compute_curl(self, gradients: np.ndarray) -> np.ndarray:
        """
        Compute curl of gradient field.
        
        Physical Meaning:
            Computes the curl of the gradient field,
            representing rotational flow patterns.
            
        Mathematical Foundation:
            ∇×v = (∂v_z/∂y - ∂v_y/∂z, ∂v_x/∂z - ∂v_z/∂x, ∂v_y/∂x - ∂v_x/∂y)
            
        Args:
            gradients (np.ndarray): Gradient field
            
        Returns:
            np.ndarray: Curl field
        """
        if gradients.shape[-1] == 3:
            # 3D curl
            curl_x = np.gradient(gradients[..., 2], axis=1) - np.gradient(gradients[..., 1], axis=2)
            curl_y = np.gradient(gradients[..., 0], axis=2) - np.gradient(gradients[..., 2], axis=0)
            curl_z = np.gradient(gradients[..., 1], axis=0) - np.gradient(gradients[..., 0], axis=1)
            curl = np.stack([curl_x, curl_y, curl_z], axis=-1)
        elif gradients.shape[-1] == 2:
            # 2D curl (scalar)
            curl = np.gradient(gradients[..., 1], axis=0) - np.gradient(gradients[..., 0], axis=1)
            curl = np.expand_dims(curl, axis=-1)
        else:
            # 1D curl (zero)
            curl = np.zeros_like(gradients)
        
        return curl


class StreamlineTracer:
    """Trace streamlines in gradient field."""
    
    def __init__(self, domain: 'Domain', parameters: Dict[str, Any] = None):
        """Initialize streamline tracer."""
        self.domain = domain
        self.parameters = parameters or {}
    
    def trace_streamlines(self, gradient_field: np.ndarray, center: Tuple[float, ...]) -> List[np.ndarray]:
        """
        Trace streamlines in gradient field.
        
        Physical Meaning:
            Traces streamlines that are tangent to the
            gradient field at each point, revealing
            the flow patterns of the field.
            
        Mathematical Foundation:
            Integrates the differential equation:
            dx/dt = ∇φ(x)
            where ∇φ is the gradient field.
            
        Args:
            gradient_field (np.ndarray): Gradient field
            center (Tuple): Center point for streamline tracing
            
        Returns:
            List[np.ndarray]: List of streamline trajectories
        """
        # Extract parameters
        num_streamlines = self.parameters.get('num_streamlines', 100)
        integration_steps = self.parameters.get('integration_steps', 1000)
        step_size = self.parameters.get('step_size', 0.01)
        
        # Create initial points around center
        initial_points = self._create_initial_points(center, num_streamlines)
        
        # Trace streamlines
        streamlines = []
        for point in initial_points:
            streamline = self._trace_single_streamline(
                gradient_field, point, integration_steps, step_size
            )
            streamlines.append(streamline)
        
        return streamlines
    
    def _create_initial_points(self, center: Tuple[float, ...], num_points: int) -> List[np.ndarray]:
        """Create initial points for streamline tracing."""
        points = []
        
        if len(center) == 3:
            # 3D initial points
            radius = 0.1
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                point = np.array([
                    center[0] + radius * np.cos(angle),
                    center[1] + radius * np.sin(angle),
                    center[2]
                ])
                points.append(point)
        elif len(center) == 2:
            # 2D initial points
            radius = 0.1
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                point = np.array([
                    center[0] + radius * np.cos(angle),
                    center[1] + radius * np.sin(angle)
                ])
                points.append(point)
        else:
            # 1D initial points
            for i in range(num_points):
                point = np.array([center[0] + 0.1 * i / num_points])
                points.append(point)
        
        return points
    
    def _trace_single_streamline(self, gradient_field: np.ndarray, initial_point: np.ndarray,
                               integration_steps: int, step_size: float) -> np.ndarray:
        """Trace a single streamline."""
        # Initialize trajectory
        trajectory = [initial_point.copy()]
        current_point = initial_point.copy()
        
        # Integrate streamline
        for _ in range(integration_steps):
            # Get gradient at current point
            gradient = self._interpolate_gradient(gradient_field, current_point)
            
            # Update point
            current_point += step_size * gradient
            
            # Check bounds
            if self._is_out_of_bounds(current_point):
                break
            
            # Add to trajectory
            trajectory.append(current_point.copy())
        
        return np.array(trajectory)
    
    def _interpolate_gradient(self, gradient_field: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Interpolate gradient at given point."""
        # Simple nearest neighbor interpolation
        if len(point) == 3:
            x_idx = int(np.clip(point[0], 0, gradient_field.shape[0] - 1))
            y_idx = int(np.clip(point[1], 0, gradient_field.shape[1] - 1))
            z_idx = int(np.clip(point[2], 0, gradient_field.shape[2] - 1))
            gradient = gradient_field[x_idx, y_idx, z_idx]
        elif len(point) == 2:
            x_idx = int(np.clip(point[0], 0, gradient_field.shape[0] - 1))
            y_idx = int(np.clip(point[1], 0, gradient_field.shape[1] - 1))
            gradient = gradient_field[x_idx, y_idx]
        else:
            x_idx = int(np.clip(point[0], 0, gradient_field.shape[0] - 1))
            gradient = gradient_field[x_idx]
        
        return gradient
    
    def _is_out_of_bounds(self, point: np.ndarray) -> bool:
        """Check if point is out of bounds."""
        for i, coord in enumerate(point):
            if coord < 0 or coord >= self.domain.shape[i]:
                return True
        return False


class TopologyAnalyzer:
    """Analyze topology of streamlines."""
    
    def __init__(self, domain: 'Domain'):
        """Initialize topology analyzer."""
        self.domain = domain
    
    def analyze_streamline_topology(self, streamlines: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze topology of streamlines.
        
        Physical Meaning:
            Analyzes the topological structure of streamlines,
            including winding numbers, topology classes, and
            stability indices.
            
        Args:
            streamlines (List[np.ndarray]): List of streamline trajectories
            
        Returns:
            Dict: Topology analysis results
        """
        # Compute winding numbers
        winding_numbers = self._compute_winding_numbers(streamlines)
        
        # Compute topology class
        topology_class = self._compute_topology_class(streamlines)
        
        # Compute stability index
        stability_index = self._compute_stability_index(streamlines)
        
        # Compute streamline density
        streamline_density = len(streamlines)
        
        return {
            'winding_numbers': winding_numbers,
            'topology_class': topology_class,
            'stability_index': stability_index,
            'streamline_density': streamline_density
        }
    
    def _compute_winding_numbers(self, streamlines: List[np.ndarray]) -> List[float]:
        """Compute winding numbers for streamlines."""
        winding_numbers = []
        
        for streamline in streamlines:
            if len(streamline) > 1:
                # Compute winding number
                winding_number = self._compute_single_winding_number(streamline)
                winding_numbers.append(winding_number)
            else:
                winding_numbers.append(0.0)
        
        return winding_numbers
    
    def _compute_single_winding_number(self, streamline: np.ndarray) -> float:
        """Compute winding number for a single streamline."""
        if len(streamline) < 2:
            return 0.0
        
        # Simple winding number computation
        total_angle = 0.0
        for i in range(len(streamline) - 1):
            # Compute angle between consecutive points
            if len(streamline[i]) >= 2:
                angle = np.arctan2(streamline[i+1][1] - streamline[i][1],
                                 streamline[i+1][0] - streamline[i][0])
                total_angle += angle
        
        winding_number = total_angle / (2 * np.pi)
        return float(winding_number)
    
    def _compute_topology_class(self, streamlines: List[np.ndarray]) -> str:
        """Compute topology class of streamlines."""
        # Simple topology classification
        if len(streamlines) == 0:
            return "empty"
        elif len(streamlines) == 1:
            return "single"
        else:
            return "multiple"
    
    def _compute_stability_index(self, streamlines: List[np.ndarray]) -> float:
        """Compute stability index of streamlines."""
        if len(streamlines) == 0:
            return 0.0
        
        # Simple stability index based on streamline length variance
        lengths = [len(streamline) for streamline in streamlines]
        length_variance = np.var(lengths)
        stability_index = 1.0 / (1.0 + length_variance)
        
        return float(stability_index)
    
    def _compute_streamline_density(self, field: np.ndarray, resolution: float) -> float:
        """Compute streamline density."""
        # Simple streamline density computation
        return 1.0
