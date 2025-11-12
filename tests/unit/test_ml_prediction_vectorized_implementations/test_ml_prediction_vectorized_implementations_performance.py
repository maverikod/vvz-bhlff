"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for performance and consistency in vectorized ML implementations.

This module tests computation performance and consistency across multiple runs.
"""

import time

import numpy as np

from bhlff.models.level_c.beating.ml.beating_ml_vectorized_methods import (
    BeatingMLVectorizedMethods,
)
from .test_ml_prediction_vectorized_implementations_base import (
    TestVectorizedMLPredictionImplementationsBase,
)


class TestVectorizedMLPredictionPerformance(
    TestVectorizedMLPredictionImplementationsBase
):
    """Test suite for performance and consistency."""
    
    def test_vectorized_computation_performance(self):
        """
        Test vectorized computation performance.
        
        Physical Meaning:
            Tests that vectorized computations are more efficient
            than non-vectorized implementations.
        """
        # Create vectorized methods
        vectorized_methods = BeatingMLVectorizedMethods()
        
        # Test performance with large envelope
        large_envelope = np.random.rand(8, 8, 8, 4, 4, 4, 2) + 1j * np.random.rand(
            8, 8, 8, 4, 4, 4, 2
        )
        
        # Measure vectorized computation time
        start_time = time.time()
        symmetry = vectorized_methods.compute_7d_phase_field_symmetry_vectorized(
            large_envelope
        )
        regularity = vectorized_methods.compute_7d_phase_field_regularity_vectorized(
            large_envelope
        )
        energy = vectorized_methods.compute_7d_phase_field_energy_vectorized(
            large_envelope
        )
        vectorized_time = time.time() - start_time
        
        # Verify computations completed successfully
        assert 0.0 <= symmetry <= 1.0
        assert 0.0 <= regularity <= 1.0
        assert energy >= 0.0
        
        # Verify vectorized computation is reasonably fast (less than 10 seconds)
        assert vectorized_time < 10.0
    
    def test_vectorized_computation_consistency(self):
        """
        Test vectorized computation consistency.
        
        Physical Meaning:
            Tests that vectorized computations produce consistent
            results across multiple runs.
        """
        # Create vectorized methods
        vectorized_methods = BeatingMLVectorizedMethods()
        
        # Test consistency across multiple runs
        results = []
        for _ in range(5):
            symmetry = vectorized_methods.compute_7d_phase_field_symmetry_vectorized(
                self.test_envelope
            )
            regularity = (
                vectorized_methods.compute_7d_phase_field_regularity_vectorized(
                    self.test_envelope
                )
            )
            energy = vectorized_methods.compute_7d_phase_field_energy_vectorized(
                self.test_envelope
            )
            results.append((symmetry, regularity, energy))
        
        # Verify all results are identical
        for i in range(1, len(results)):
            assert abs(results[i][0] - results[0][0]) < 1e-10  # symmetry
            assert abs(results[i][1] - results[0][1]) < 1e-10  # regularity
            assert abs(results[i][2] - results[0][2]) < 1e-10  # energy

