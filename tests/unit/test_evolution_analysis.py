"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for evolution analysis.

This module contains comprehensive tests for evolution analysis
in 7D phase field cosmological evolution.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from bhlff.models.level_g.evolution.evolution_analysis import EvolutionAnalysis


class TestEvolutionAnalysis:
    """Test suite for evolution analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analysis = EvolutionAnalysis()
        self.test_parameters = {
            'time_span': 1.0,
            'spatial_extent': 1.0,
            'resolution': 64,
            'cosmological_constant': 0.1,
            'matter_density': 0.3,
            'dark_energy_density': 0.7
        }
        self.test_field = np.random.random((64, 64, 64)) + 1j * np.random.random((64, 64, 64))
    
    def test_initialization(self):
        """Test evolution analysis initialization."""
        assert hasattr(self.analysis, 'analysis_cache')
        assert hasattr(self.analysis, 'time_resolution')
        assert hasattr(self.analysis, 'spatial_resolution')
        assert hasattr(self.analysis, 'evolution_precision')
        assert self.analysis.time_resolution == 100
        assert self.analysis.spatial_resolution == 64
        assert self.analysis.evolution_precision == 1e-12
    
    def test_cosmological_evolution_analysis(self):
        """Test cosmological evolution analysis."""
        evolution_results = {
            'scale_factor': [1.0, 1.1, 1.2],
            'structure_formation': [
                {'phase_field_rms': 0.1, 'time': 0.0},
                {'phase_field_rms': 0.2, 'time': 0.5},
                {'phase_field_rms': 0.3, 'time': 1.0}
            ]
        }
        result = self.analysis.analyze_cosmological_evolution(
            evolution_results, 0.0, 1.0, 0.01
        )
        
        assert isinstance(result, dict)
        assert 'total_evolution_time' in result
        assert 'final_scale_factor' in result
        assert 'expansion_rate' in result
        assert 'structure_formation_rate' in result
        assert 'cosmological_parameters_evolution' in result
        
        # Check that all results are finite
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                assert np.all(np.isfinite(value))
            elif isinstance(value, (int, float)):
                assert np.isfinite(value)
    
    def test_structure_formation_rate(self):
        """Test structure formation rate computation."""
        evolution_results = {
            'structure_formation': [
                {'phase_field_rms': 0.1, 'time': 0.0},
                {'phase_field_rms': 0.2, 'time': 0.5},
                {'phase_field_rms': 0.3, 'time': 1.0}
            ]
        }
        rate = self.analysis._compute_structure_formation_rate(
            evolution_results, 0.0, 1.0
        )
        
        assert isinstance(rate, float)
        assert np.isfinite(rate)
        assert rate >= 0.0
    
    def test_parameter_evolution_analysis(self):
        """Test parameter evolution analysis."""
        evolution_results = {
            'cosmological_parameters': [
                {'time': 0.0, 'scale_factor': 1.0, 'hubble_parameter': 70.0},
                {'time': 0.5, 'scale_factor': 1.1, 'hubble_parameter': 69.0},
                {'time': 1.0, 'scale_factor': 1.2, 'hubble_parameter': 68.0}
            ]
        }
        result = self.analysis._analyze_parameter_evolution(evolution_results)
        
        assert isinstance(result, dict)
        assert 'time_evolution' in result
        assert 'scale_factor_evolution' in result
        assert 'hubble_evolution' in result
        assert 'parameter_trends' in result
    
    def test_parameter_trends_computation(self):
        """Test parameter trends computation."""
        cosmological_params = [
            {'scale_factor': 1.0, 'hubble_parameter': 70.0},
            {'scale_factor': 1.1, 'hubble_parameter': 69.0},
            {'scale_factor': 1.2, 'hubble_parameter': 68.0}
        ]
        trends = self.analysis._compute_parameter_trends(cosmological_params)
        
        assert isinstance(trends, dict)
        assert 'scale_factor_trend' in trends
        assert 'hubble_trend' in trends
        assert all(isinstance(v, float) for v in trends.values())
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with empty evolution results
        empty_results = {}
        result = self.analysis.analyze_cosmological_evolution(
            empty_results, 0.0, 1.0, 0.01
        )
        
        assert isinstance(result, dict)
        assert len(result) == 0
        
        # Test with minimal evolution results
        minimal_results = {
            'scale_factor': [1.0, 1.1, 1.2],
            'structure_formation': [{'phase_field_rms': 0.1}]
        }
        result = self.analysis.analyze_cosmological_evolution(
            minimal_results, 0.0, 1.0, 0.01
        )
        
        assert isinstance(result, dict)
        assert 'total_evolution_time' in result
    
    def test_extreme_values(self):
        """Test with extreme values."""
        # Test with very large evolution results
        large_results = {
            'scale_factor': [1.0 + i * 1e6 for i in range(1000)],
            'structure_formation': [
                {'phase_field_rms': 1e6 + i * 1000, 'time': i * 0.01} 
                for i in range(1000)
            ]
        }
        result = self.analysis.analyze_cosmological_evolution(
            large_results, 0.0, 1e6, 0.01
        )
        
        assert isinstance(result, dict)
        assert 'total_evolution_time' in result
        
        # Test with extreme time values
        extreme_results = {
            'scale_factor': [1.0, 1.1, 1.2],
            'structure_formation': [{'phase_field_rms': 1e-6}]
        }
        result = self.analysis.analyze_cosmological_evolution(
            extreme_results, 0.0, 1e-6, 1e-9
        )
        
        assert isinstance(result, dict)
        assert 'total_evolution_time' in result
    
    def test_analysis_consistency(self):
        """Test analysis consistency."""
        evolution_results = {
            'scale_factor': [1.0, 1.1, 1.2],
            'structure_formation': [
                {'phase_field_rms': 0.1, 'time': 0.0},
                {'phase_field_rms': 0.2, 'time': 0.5},
                {'phase_field_rms': 0.3, 'time': 1.0}
            ]
        }
        
        # Test that same input gives same output
        result1 = self.analysis.analyze_cosmological_evolution(
            evolution_results, 0.0, 1.0, 0.01
        )
        result2 = self.analysis.analyze_cosmological_evolution(
            evolution_results, 0.0, 1.0, 0.01
        )
        
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)
        assert len(result1) == len(result2)
        
        # Check that results are consistent
        for key in result1:
            assert key in result2
            if isinstance(result1[key], np.ndarray):
                assert np.allclose(result1[key], result2[key], rtol=1e-10)
            elif isinstance(result1[key], dict):
                # For dictionaries, check that they have the same keys
                assert set(result1[key].keys()) == set(result2[key].keys())
            else:
                assert abs(result1[key] - result2[key]) < 1e-10
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very small values
        small_results = {
            'scale_factor': [1e-15, 1e-14, 1e-13],
            'structure_formation': [
                {'phase_field_rms': 1e-15, 'time': 0.0},
                {'phase_field_rms': 1e-14, 'time': 0.5},
                {'phase_field_rms': 1e-13, 'time': 1.0}
            ]
        }
        result = self.analysis.analyze_cosmological_evolution(
            small_results, 0.0, 1.0, 0.01
        )
        
        assert isinstance(result, dict)
        assert 'total_evolution_time' in result
        
        # Test with very large values
        large_results = {
            'scale_factor': [1e15, 1e16, 1e17],
            'structure_formation': [
                {'phase_field_rms': 1e15, 'time': 0.0},
                {'phase_field_rms': 1e16, 'time': 0.5},
                {'phase_field_rms': 1e17, 'time': 1.0}
            ]
        }
        result = self.analysis.analyze_cosmological_evolution(
            large_results, 0.0, 1.0, 0.01
        )
        
        assert isinstance(result, dict)
        assert 'total_evolution_time' in result
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test with invalid time parameters
        invalid_results = {
            'scale_factor': [1.0, 1.1, 1.2],
            'structure_formation': [{'phase_field_rms': 0.1}]
        }
        
        # Should handle invalid time parameters gracefully
        try:
            result = self.analysis.analyze_cosmological_evolution(
                invalid_results, -1.0, 0.0, 0.01  # Negative time start
            )
            assert isinstance(result, dict)
        except ValueError:
            # Expected behavior for invalid parameters
            pass
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large arrays."""
        # Test that large evolution results are handled efficiently
        very_large_results = {
            'scale_factor': [1.0 + i * 0.001 for i in range(10000)],
            'structure_formation': [
                {'phase_field_rms': 0.1 + i * 0.001, 'time': i * 0.001} 
                for i in range(10000)
            ]
        }
        
        # This should not cause memory issues
        result = self.analysis.analyze_cosmological_evolution(
            very_large_results, 0.0, 10.0, 0.001
        )
        assert isinstance(result, dict)
        assert 'total_evolution_time' in result
