"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Core beating analysis module.

This module implements core beating analysis functionality
for analyzing mode beating in the 7D phase field according to the
theoretical framework.

Physical Meaning:
    Implements core beating analysis including interference patterns,
    mode coupling, and phase coherence analysis.

Example:
    >>> analyzer = CoreBeatingAnalyzer(bvp_core)
    >>> results = analyzer.analyze_beating_comprehensive(envelope)
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from scipy.fft import fftn, ifftn, fftfreq
from scipy.signal import find_peaks, welch
from scipy.optimize import minimize

from bhlff.core.bvp import BVPCore
from .core_analysis_basic import BasicBeatingAnalyzer
from .core_analysis_interference import InterferencePatternAnalyzer
from .core_analysis_coupling import ModeCouplingAnalyzer
from .core_analysis_phase import PhaseCoherenceAnalyzer
from .core_analysis_frequency import BeatingFrequencyAnalyzer


class CoreBeatingAnalyzer:
    """
    Core beating analysis for Level C.

    Physical Meaning:
        Performs core beating analysis according to the 7D phase field
        theory, including interference patterns, mode coupling, and
        phase coherence analysis.

    Mathematical Foundation:
        Analyzes beating through mode interference:
        I(t) = |A₁e^(iω₁t) + A₂e^(iω₂t)|²
        where A₁, A₂ are mode amplitudes and ω₁, ω₂ are frequencies.
    """

    def __init__(self, bvp_core: BVPCore):
        """
        Initialize core beating analyzer.

        Physical Meaning:
            Sets up the core beating analysis system with
            theoretical parameters and analysis modules.

        Args:
            bvp_core (BVPCore): BVP core instance for field access.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)

        # Theoretical analysis parameters
        self.interference_threshold = 1e-12  # Minimum interference strength
        self.coupling_threshold = 1e-10  # Minimum coupling strength
        self.phase_coherence_threshold = 0.01  # Minimum phase coherence
        self.statistical_significance = 0.05
        self.optimization_tolerance = 1e-8

        # Initialize analysis components
        self._basic_analyzer = BasicBeatingAnalyzer()
        self._interference_analyzer = InterferencePatternAnalyzer(self.interference_threshold)
        self._coupling_analyzer = ModeCouplingAnalyzer(self.coupling_threshold)
        self._phase_analyzer = PhaseCoherenceAnalyzer(self.phase_coherence_threshold)
        self._frequency_analyzer = BeatingFrequencyAnalyzer()

    def analyze_beating_comprehensive(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive beating analysis according to theoretical framework.

        Physical Meaning:
            Performs full theoretical analysis of mode beating
            according to the 7D phase field theory, including
            interference patterns, mode coupling, and phase coherence.

        Mathematical Foundation:
            Analyzes beating through mode interference:
            I(t) = |A₁e^(iω₁t) + A₂e^(iω₂t)|²
            where A₁, A₂ are mode amplitudes and ω₁, ω₂ are frequencies.

        Args:
            envelope (np.ndarray): 7D envelope field data.

        Returns:
            Dict[str, Any]: Comprehensive analysis results including:
                - interference_patterns: Detected interference patterns
                - mode_coupling: Mode coupling analysis
                - phase_coherence: Phase coherence analysis
                - beating_frequencies: Theoretical beating frequencies
        """
        self.logger.info("Starting comprehensive beating analysis")

        # Basic analysis
        basic_results = self._basic_analyzer.analyze_beating_basic(envelope)

        # Interference pattern analysis
        interference_results = self._interference_analyzer.analyze_interference_patterns(envelope)

        # Mode coupling analysis
        coupling_results = self._coupling_analyzer.analyze_mode_coupling(envelope)

        # Phase coherence analysis
        phase_results = self._phase_analyzer.analyze_phase_coherence(envelope)

        # Beating frequency analysis
        frequency_results = self._frequency_analyzer.analyze_beating_frequencies(envelope)

        # Combine all results
        comprehensive_results = {
            "basic_analysis": basic_results,
            "interference_patterns": interference_results,
            "mode_coupling": coupling_results,
            "phase_coherence": phase_results,
            "beating_frequencies": frequency_results,
            "analysis_complete": True,
        }

        self.logger.info("Comprehensive beating analysis completed")
        return comprehensive_results
