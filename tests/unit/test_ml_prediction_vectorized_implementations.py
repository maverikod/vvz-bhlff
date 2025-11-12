"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test facade for vectorized ML prediction implementations.

This module provides a unified interface for all vectorized ML prediction tests.
"""

from .test_ml_prediction_vectorized_implementations.test_ml_prediction_vectorized_implementations_predictions import (
    TestVectorizedMLPredictionPredictions,
)
from .test_ml_prediction_vectorized_implementations.test_ml_prediction_vectorized_implementations_7d import (
    TestVectorizedMLPrediction7DComputations,
)
from .test_ml_prediction_vectorized_implementations.test_ml_prediction_vectorized_implementations_performance import (
    TestVectorizedMLPredictionPerformance,
)
from .test_ml_prediction_vectorized_implementations.test_ml_prediction_vectorized_implementations_components import (
    TestVectorizedMLPredictionComponents,
)

# Export all test classes for pytest discovery
__all__ = [
    "TestVectorizedMLPredictionPredictions",
    "TestVectorizedMLPrediction7DComputations",
    "TestVectorizedMLPredictionPerformance",
    "TestVectorizedMLPredictionComponents",
]
