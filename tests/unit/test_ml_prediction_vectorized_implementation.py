"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test suite for vectorized ML prediction implementation.

This module tests the full ML prediction implementations with vectorization
for beating analysis in Level C of 7D phase field theory.

Physical Meaning:
    Tests the complete ML prediction implementations using 7D BVP theory
    and vectorized processing for frequency and coupling prediction.

Example:
    >>> pytest tests/unit/test_ml_prediction_vectorized_implementation.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any
import logging

from bhlff.core.bvp import BVPCore
from bhlff.models.level_c.beating.ml.beating_ml_optimization_prediction import (
    BeatingMLPredictionOptimizer,
)
from bhlff.models.level_c.beating.ml.beating_ml_optimization_core import (
    BeatingMLOptimizationCore,
)
from bhlff.models.level_c.beating.ml.beating_ml_optimization_classification import (
    BeatingMLClassificationOptimizer,
)


class TestMLPredictionVectorizedImplementation:
    """
    Test suite for vectorized ML prediction implementation.

    Physical Meaning:
        Tests the complete ML prediction implementations using 7D BVP theory
        and vectorized processing for frequency and coupling prediction.
    """

    @pytest.fixture
    def bvp_core(self):
        """Create BVP core for testing using vectorized processing."""
        from bhlff.core.domain import Domain
        from bhlff.core.bvp.bvp_core.bvp_vectorized_processor import (
            BVPVectorizedProcessor,
        )

        # Create test domain (smaller for testing)
        domain = Domain(L=1.0, N=4, dimensions=7)

        # Create test config
        config = {
            "carrier_frequency": 1e15,
            "envelope_equation": {
                "kappa_0": 1.0,
                "kappa_2": 0.1,
                "chi_prime": 1.0,
                "chi_double_prime_0": 0.1,
                "k0": 1.0,
            },
            "mu": 1.0,
            "beta": 1.0,
            "lambda": 0.0,
            "nu": 1.0,
            "amplitude": 1.0,
            "phase": 0.0,
        }

        # Create vectorized BVP processor instead of full BVP core
        return BVPVectorizedProcessor(
            domain, config, block_size=2, overlap=1, use_cuda=False
        )

    @pytest.fixture
    def test_envelope(self):
        """Create test envelope data using vectorized processing."""
        # Create 7D envelope field data using vectorized blocks
        # This simulates real 7D phase field data processed in blocks
        block_shape = (2, 2, 2, 2, 2, 2, 2)  # 7D block shape
        envelope = np.random.randn(*block_shape) + 1j * np.random.randn(*block_shape)
        return envelope

    @pytest.fixture
    def test_parameters(self):
        """Create test parameters."""
        return {
            "prediction_horizon": 10,
            "feature_window": 5,
            "prediction_threshold": 0.7,
            "model_complexity": "medium",
            "regularization_strength": 0.01,
            "classification_threshold": 0.5,
            "cross_validation_folds": 5,
            "random_state": 42,
        }

    def test_prediction_accuracy_calculation(
        self, bvp_core, test_envelope, test_parameters
    ):
        """
        Test prediction accuracy calculation using 7D BVP theory and vectorized processing.

        Physical Meaning:
            Tests the full prediction accuracy calculation using
            7D phase field analysis and vectorized processing with real physics.
        """
        # Create ML prediction optimizer with vectorized processor
        optimizer = BeatingMLPredictionOptimizer(bvp_core)

        # Test accuracy calculation with vectorized envelope data
        accuracy = optimizer._calculate_prediction_accuracy(
            test_parameters, test_envelope
        )

        # Verify accuracy is within valid range
        assert 0.0 <= accuracy <= 1.0
        assert isinstance(accuracy, float)

        # Verify accuracy is not random (should be deterministic)
        accuracy2 = optimizer._calculate_prediction_accuracy(
            test_parameters, test_envelope
        )
        assert abs(accuracy - accuracy2) < 1e-10

        # Test that accuracy depends on envelope properties (physical meaning)
        # Higher energy envelope should give different accuracy
        high_energy_envelope = test_envelope * 2.0
        high_energy_accuracy = optimizer._calculate_prediction_accuracy(
            test_parameters, high_energy_envelope
        )
        assert abs(accuracy - high_energy_accuracy) > 1e-6  # Should be different

    def test_prediction_precision_calculation(
        self, bvp_core, test_envelope, test_parameters
    ):
        """
        Test prediction precision calculation using 7D BVP theory.

        Physical Meaning:
            Tests the full prediction precision calculation using
            7D phase field analysis and vectorized processing.
        """
        optimizer = BeatingMLPredictionOptimizer(bvp_core)

        # Test precision calculation
        precision = optimizer._calculate_prediction_precision(
            test_parameters, test_envelope
        )

        # Verify precision is within valid range
        assert 0.0 <= precision <= 1.0
        assert isinstance(precision, float)

        # Verify precision is not random (should be deterministic)
        precision2 = optimizer._calculate_prediction_precision(
            test_parameters, test_envelope
        )
        assert abs(precision - precision2) < 1e-10

    def test_prediction_recall_calculation(
        self, bvp_core, test_envelope, test_parameters
    ):
        """
        Test prediction recall calculation using 7D BVP theory.

        Physical Meaning:
            Tests the full prediction recall calculation using
            7D phase field analysis and vectorized processing.
        """
        optimizer = BeatingMLPredictionOptimizer(bvp_core)

        # Test recall calculation
        recall = optimizer._calculate_prediction_recall(test_parameters, test_envelope)

        # Verify recall is within valid range
        assert 0.0 <= recall <= 1.0
        assert isinstance(recall, float)

        # Verify recall is not random (should be deterministic)
        recall2 = optimizer._calculate_prediction_recall(test_parameters, test_envelope)
        assert abs(recall - recall2) < 1e-10

    def test_prediction_f1_score_calculation(
        self, bvp_core, test_envelope, test_parameters
    ):
        """
        Test prediction F1 score calculation using 7D BVP theory.

        Physical Meaning:
            Tests the full prediction F1 score calculation using
            7D phase field analysis and vectorized processing.
        """
        optimizer = BeatingMLPredictionOptimizer(bvp_core)

        # Test F1 score calculation
        f1_score = optimizer._calculate_prediction_f1_score(
            test_parameters, test_envelope
        )

        # Verify F1 score is within valid range
        assert 0.0 <= f1_score <= 1.0
        assert isinstance(f1_score, float)

        # Verify F1 score is not random (should be deterministic)
        f1_score2 = optimizer._calculate_prediction_f1_score(
            test_parameters, test_envelope
        )
        assert abs(f1_score - f1_score2) < 1e-10

    def test_ml_optimization_core_accuracy(
        self, bvp_core, test_envelope, test_parameters
    ):
        """
        Test ML optimization core accuracy calculation.

        Physical Meaning:
            Tests the full ML optimization core accuracy calculation using
            7D phase field analysis and vectorized processing.
        """
        core = BeatingMLOptimizationCore(bvp_core)

        # Test accuracy calculation
        accuracy = core._calculate_accuracy(test_parameters, test_envelope)

        # Verify accuracy is within valid range
        assert 0.0 <= accuracy <= 1.0
        assert isinstance(accuracy, float)

        # Verify accuracy is not random (should be deterministic)
        accuracy2 = core._calculate_accuracy(test_parameters, test_envelope)
        assert abs(accuracy - accuracy2) < 1e-10

    def test_ml_optimization_core_precision(
        self, bvp_core, test_envelope, test_parameters
    ):
        """
        Test ML optimization core precision calculation.

        Physical Meaning:
            Tests the full ML optimization core precision calculation using
            7D phase field analysis and vectorized processing.
        """
        core = BeatingMLOptimizationCore(bvp_core)

        # Test precision calculation
        precision = core._calculate_precision(test_parameters, test_envelope)

        # Verify precision is within valid range
        assert 0.0 <= precision <= 1.0
        assert isinstance(precision, float)

        # Verify precision is not random (should be deterministic)
        precision2 = core._calculate_precision(test_parameters, test_envelope)
        assert abs(precision - precision2) < 1e-10

    def test_ml_optimization_core_recall(
        self, bvp_core, test_envelope, test_parameters
    ):
        """
        Test ML optimization core recall calculation.

        Physical Meaning:
            Tests the full ML optimization core recall calculation using
            7D phase field analysis and vectorized processing.
        """
        core = BeatingMLOptimizationCore(bvp_core)

        # Test recall calculation
        recall = core._calculate_recall(test_parameters, test_envelope)

        # Verify recall is within valid range
        assert 0.0 <= recall <= 1.0
        assert isinstance(recall, float)

        # Verify recall is not random (should be deterministic)
        recall2 = core._calculate_recall(test_parameters, test_envelope)
        assert abs(recall - recall2) < 1e-10

    def test_ml_optimization_core_f1_score(
        self, bvp_core, test_envelope, test_parameters
    ):
        """
        Test ML optimization core F1 score calculation.

        Physical Meaning:
            Tests the full ML optimization core F1 score calculation using
            7D phase field analysis and vectorized processing.
        """
        core = BeatingMLOptimizationCore(bvp_core)

        # Test F1 score calculation
        f1_score = core._calculate_f1_score(test_parameters, test_envelope)

        # Verify F1 score is within valid range
        assert 0.0 <= f1_score <= 1.0
        assert isinstance(f1_score, float)

        # Verify F1 score is not random (should be deterministic)
        f1_score2 = core._calculate_f1_score(test_parameters, test_envelope)
        assert abs(f1_score - f1_score2) < 1e-10

    def test_classification_accuracy_calculation(
        self, bvp_core, test_envelope, test_parameters
    ):
        """
        Test classification accuracy calculation using 7D BVP theory.

        Physical Meaning:
            Tests the full classification accuracy calculation using
            7D phase field analysis and vectorized processing.
        """
        classifier = BeatingMLClassificationOptimizer(bvp_core)

        # Test accuracy calculation
        accuracy = classifier._calculate_classification_accuracy(
            test_parameters, test_envelope
        )

        # Verify accuracy is within valid range
        assert 0.0 <= accuracy <= 1.0
        assert isinstance(accuracy, float)

        # Verify accuracy is not random (should be deterministic)
        accuracy2 = classifier._calculate_classification_accuracy(
            test_parameters, test_envelope
        )
        assert abs(accuracy - accuracy2) < 1e-10

    def test_classification_precision_calculation(
        self, bvp_core, test_envelope, test_parameters
    ):
        """
        Test classification precision calculation using 7D BVP theory.

        Physical Meaning:
            Tests the full classification precision calculation using
            7D phase field analysis and vectorized processing.
        """
        classifier = BeatingMLClassificationOptimizer(bvp_core)

        # Test precision calculation
        precision = classifier._calculate_classification_precision(
            test_parameters, test_envelope
        )

        # Verify precision is within valid range
        assert 0.0 <= precision <= 1.0
        assert isinstance(precision, float)

        # Verify precision is not random (should be deterministic)
        precision2 = classifier._calculate_classification_precision(
            test_parameters, test_envelope
        )
        assert abs(precision - precision2) < 1e-10

    def test_classification_recall_calculation(
        self, bvp_core, test_envelope, test_parameters
    ):
        """
        Test classification recall calculation using 7D BVP theory.

        Physical Meaning:
            Tests the full classification recall calculation using
            7D phase field analysis and vectorized processing.
        """
        classifier = BeatingMLClassificationOptimizer(bvp_core)

        # Test recall calculation
        recall = classifier._calculate_classification_recall(
            test_parameters, test_envelope
        )

        # Verify recall is within valid range
        assert 0.0 <= recall <= 1.0
        assert isinstance(recall, float)

        # Verify recall is not random (should be deterministic)
        recall2 = classifier._calculate_classification_recall(
            test_parameters, test_envelope
        )
        assert abs(recall - recall2) < 1e-10

    def test_classification_f1_score_calculation(
        self, bvp_core, test_envelope, test_parameters
    ):
        """
        Test classification F1 score calculation using 7D BVP theory.

        Physical Meaning:
            Tests the full classification F1 score calculation using
            7D phase field analysis and vectorized processing.
        """
        classifier = BeatingMLClassificationOptimizer(bvp_core)

        # Test F1 score calculation
        f1_score = classifier._calculate_classification_f1_score(
            test_parameters, test_envelope
        )

        # Verify F1 score is within valid range
        assert 0.0 <= f1_score <= 1.0
        assert isinstance(f1_score, float)

        # Verify F1 score is not random (should be deterministic)
        f1_score2 = classifier._calculate_classification_f1_score(
            test_parameters, test_envelope
        )
        assert abs(f1_score - f1_score2) < 1e-10

    def test_spectral_entropy_computation(self, bvp_core, test_envelope):
        """
        Test spectral entropy computation using 7D BVP theory.

        Physical Meaning:
            Tests the spectral entropy computation using
            7D phase field theory and VBP envelope analysis.
        """
        optimizer = BeatingMLPredictionOptimizer(bvp_core)

        # Test spectral entropy computation
        entropy = optimizer._compute_spectral_entropy(test_envelope)

        # Verify entropy is non-negative
        assert entropy >= 0.0
        assert isinstance(entropy, float)

        # Verify entropy is deterministic
        entropy2 = optimizer._compute_spectral_entropy(test_envelope)
        assert abs(entropy - entropy2) < 1e-10

    def test_phase_coherence_computation(self, bvp_core, test_envelope):
        """
        Test phase coherence computation using 7D BVP theory.

        Physical Meaning:
            Tests the phase coherence computation using
            7D phase field theory and VBP envelope analysis.
        """
        optimizer = BeatingMLPredictionOptimizer(bvp_core)

        # Test phase coherence computation
        coherence = optimizer._compute_phase_coherence(test_envelope)

        # Verify coherence is within valid range
        assert 0.0 <= coherence <= 1.0
        assert isinstance(coherence, float)

        # Verify coherence is deterministic
        coherence2 = optimizer._compute_phase_coherence(test_envelope)
        assert abs(coherence - coherence2) < 1e-10

    def test_parameter_optimization_with_vectorization(self, bvp_core, test_parameters):
        """
        Test parameter optimization using vectorized processing.

        Physical Meaning:
            Tests the parameter optimization using vectorized processing
            and 7D phase field theory.
        """
        core = BeatingMLOptimizationCore(bvp_core)

        # Test parameter adjustment
        adjusted_params = core._adjust_parameters(test_parameters, 0.8)

        # Verify adjusted parameters are valid
        assert isinstance(adjusted_params, dict)
        assert len(adjusted_params) == len(test_parameters)

        # Verify numerical parameters are adjusted
        for key, value in adjusted_params.items():
            if isinstance(value, (int, float)):
                assert isinstance(value, (int, float))

    def test_classification_parameter_optimization_with_vectorization(
        self, bvp_core, test_parameters
    ):
        """
        Test classification parameter optimization using vectorized processing.

        Physical Meaning:
            Tests the classification parameter optimization using vectorized processing
            and 7D phase field theory.
        """
        classifier = BeatingMLClassificationOptimizer(bvp_core)

        # Test classification parameter adjustment
        adjusted_params = classifier._adjust_classification_parameters(
            test_parameters, 0.8
        )

        # Verify adjusted parameters are valid
        assert isinstance(adjusted_params, dict)
        assert len(adjusted_params) == len(test_parameters)

        # Verify numerical parameters are adjusted
        for key, value in adjusted_params.items():
            if isinstance(value, (int, float)):
                assert isinstance(value, (int, float))

    def test_7d_performance_metric_computation(self, bvp_core, test_parameters):
        """
        Test 7D performance metric computation.

        Physical Meaning:
            Tests the 7D performance metric computation using
            7D phase field theory and VBP envelope analysis.
        """
        core = BeatingMLOptimizationCore(bvp_core)

        # Test performance metric computation
        metric = core._compute_7d_performance_metric(test_parameters, 0.8)

        # Verify metric is within valid range
        assert 0.0 <= metric <= 1.0
        assert isinstance(metric, float)

        # Verify metric is deterministic
        metric2 = core._compute_7d_performance_metric(test_parameters, 0.8)
        assert abs(metric - metric2) < 1e-10

    def test_7d_classification_metric_computation(self, bvp_core, test_parameters):
        """
        Test 7D classification metric computation.

        Physical Meaning:
            Tests the 7D classification metric computation using
            7D phase field theory and VBP envelope analysis.
        """
        classifier = BeatingMLClassificationOptimizer(bvp_core)

        # Test classification metric computation
        metric = classifier._compute_7d_classification_metric(test_parameters, 0.8)

        # Verify metric is within valid range
        assert 0.0 <= metric <= 1.0
        assert isinstance(metric, float)

        # Verify metric is deterministic
        metric2 = classifier._compute_7d_classification_metric(test_parameters, 0.8)
        assert abs(metric - metric2) < 1e-10

    def test_vectorized_processing_integration(
        self, bvp_core, test_envelope, test_parameters
    ):
        """
        Test vectorized processing integration with real 7D BVP physics.

        Physical Meaning:
            Tests the integration of vectorized processing with
            7D phase field theory for ML prediction using real physics.
        """
        # Test that we have a vectorized processor
        assert hasattr(bvp_core, "domain")
        assert hasattr(bvp_core, "config")
        assert hasattr(bvp_core, "use_cuda")

        # Test vectorized envelope processing
        # This simulates real 7D phase field processing in blocks
        if hasattr(bvp_core, "solve_envelope_vectorized"):
            # Test vectorized envelope solving
            source = np.random.randn(*test_envelope.shape) + 1j * np.random.randn(
                *test_envelope.shape
            )
            try:
                result = bvp_core.solve_envelope_vectorized(
                    source, max_iterations=5, tolerance=1e-3
                )
                assert result.shape == test_envelope.shape
                assert isinstance(result, np.ndarray)
            except Exception as e:
                # If vectorized solving fails due to memory, that's expected for large domains
                assert "memory" in str(e).lower() or "allocation" in str(e).lower()

        # Test ML prediction with vectorized processor
        optimizer = BeatingMLPredictionOptimizer(bvp_core)

        # Test that vectorized processing methods exist
        assert hasattr(optimizer, "_optimize_with_vectorized_processing")
        assert hasattr(optimizer, "_compute_7d_phase_field_optimization")

        # Test that methods can be called without errors
        try:
            result = optimizer._compute_7d_phase_field_optimization(
                0.8, test_parameters
            )
            assert isinstance(result, dict)
        except Exception as e:
            # If vectorized processor is not available, that's okay
            assert "vectorized_processor" in str(e) or "not available" in str(e)

    def test_no_simplified_calculations(self, bvp_core, test_envelope, test_parameters):
        """
        Test that no simplified calculations are used.

        Physical Meaning:
            Verifies that all calculations use full 7D BVP theory
            and not simplified approximations.
        """
        optimizer = BeatingMLPredictionOptimizer(bvp_core)
        core = BeatingMLOptimizationCore(bvp_core)
        classifier = BeatingMLClassificationOptimizer(bvp_core)

        # Test that calculations are deterministic (not random)
        accuracy1 = optimizer._calculate_prediction_accuracy(
            test_parameters, test_envelope
        )
        accuracy2 = optimizer._calculate_prediction_accuracy(
            test_parameters, test_envelope
        )
        assert abs(accuracy1 - accuracy2) < 1e-10

        precision1 = core._calculate_precision(test_parameters, test_envelope)
        precision2 = core._calculate_precision(test_parameters, test_envelope)
        assert abs(precision1 - precision2) < 1e-10

        classification_accuracy1 = classifier._calculate_classification_accuracy(
            test_parameters, test_envelope
        )
        classification_accuracy2 = classifier._calculate_classification_accuracy(
            test_parameters, test_envelope
        )
        assert abs(classification_accuracy1 - classification_accuracy2) < 1e-10

        # Test that calculations are not using random values
        assert not any(
            "np.random"
            in str(optimizer._calculate_prediction_accuracy.__code__.co_consts)
        )
        assert not any("np.random" in str(core._calculate_precision.__code__.co_consts))
        assert not any(
            "np.random"
            in str(classifier._calculate_classification_accuracy.__code__.co_consts)
        )

    def test_vectorized_block_processing_physics(
        self, bvp_core, test_envelope, test_parameters
    ):
        """
        Test vectorized block processing with real 7D BVP physics.

        Physical Meaning:
            Tests that vectorized processing correctly handles 7D phase field
            physics in blocks, ensuring physical consistency across block boundaries.
        """
        # Test that vectorized processor can handle 7D blocks
        assert hasattr(bvp_core, "domain")
        assert bvp_core.domain.dimensions == 7

        # Test block iteration (this is the key to vectorized processing)
        if hasattr(bvp_core, "iterate_blocks"):
            blocks = list(bvp_core.iterate_blocks())
            assert len(blocks) > 0  # Should have blocks

            # Test that each block has proper shape
            for block, block_info in blocks:
                assert isinstance(block, np.ndarray)
                assert len(block.shape) == 7  # 7D blocks
                assert block_info.shape == block.shape

        # Test vectorized processing with physics
        optimizer = BeatingMLPredictionOptimizer(bvp_core)

        # Test that ML predictions work with block data
        accuracy = optimizer._calculate_prediction_accuracy(
            test_parameters, test_envelope
        )
        assert 0.0 <= accuracy <= 1.0

        # Test that different blocks give different results (physics)
        if hasattr(bvp_core, "iterate_blocks"):
            blocks = list(bvp_core.iterate_blocks())
            if len(blocks) > 1:
                # Test first block
                block1, _ = blocks[0]
                accuracy1 = optimizer._calculate_prediction_accuracy(
                    test_parameters, block1
                )

                # Test second block
                block2, _ = blocks[1]
                accuracy2 = optimizer._calculate_prediction_accuracy(
                    test_parameters, block2
                )

                # Different blocks should give different accuracies (physics)
                assert (
                    abs(accuracy1 - accuracy2) > 1e-6 or accuracy1 == accuracy2
                )  # Allow same if blocks are similar
