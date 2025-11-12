"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Data structures for Level C acceptance criteria validation results.

Physical Meaning:
    Defines data structures for storing validation results
    for each Level C test component.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class C1AcceptanceResults:
    """C1 test acceptance criteria validation results."""

    no_peaks_at_zero_contrast: bool
    resonance_birth_threshold: float
    localization_correct: bool
    passivity_check: bool
    convergence_omega: bool
    convergence_q: bool
    all_passed: bool
    failures: List[str]


@dataclass
class C2AcceptanceResults:
    """C2 test acceptance criteria validation results."""

    minimum_peaks_count: int
    abcd_errors_overall: float
    abcd_errors_at_peaks: float
    frequency_errors: List[float]
    quality_factor_errors: List[float]
    passivity_check: bool
    all_passed: bool
    failures: List[str]


@dataclass
class C3AcceptanceResults:
    """C3 test acceptance criteria validation results."""

    drift_velocity_at_zero_memory: float
    drift_velocity_error_at_zero: float
    freezing_threshold_gamma_star: float
    drift_velocity_at_threshold: float
    jaccard_index: float
    all_passed: bool
    failures: List[str]


@dataclass
class C4AcceptanceResults:
    """C4 test acceptance criteria validation results."""

    beating_error_without_pinning: float
    suppression_factor_with_pinning: float
    all_passed: bool
    failures: List[str]

