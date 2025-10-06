"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical validation tests for BVP postulates integration.

This module provides comprehensive physical validation tests for the
integration of all 9 BVP postulates, ensuring they work together to
provide complete physical consistency of the BVP theory.

Physical Meaning:
    Tests validate that all 9 BVP postulates work together to ensure
    complete physical consistency of the BVP theory.

Mathematical Foundation:
    Tests that all postulates are satisfied simultaneously,
    ensuring the complete BVP framework is physically consistent.

Example:
    >>> pytest tests/unit/test_core/test_bvp_postulates_integration_physics.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any, List

from bhlff.core.domain import Domain
from bhlff.core.bvp.constants.bvp_constants_advanced import BVPConstantsAdvanced
from bhlff.core.bvp.postulates.carrier_primacy_postulate import (
    BVPPostulate1_CarrierPrimacy,
)
from bhlff.core.bvp.postulates.scale_separation_postulate import (
    BVPPostulate2_ScaleSeparation,
)
from bhlff.core.bvp.postulates.bvp_rigidity_postulate import BVPPostulate3_BVPRigidity
from bhlff.core.bvp.postulates.u1_phase_structure_postulate import (
    BVPPostulate4_U1PhaseStructure,
)
from bhlff.core.bvp.postulates.quenches_postulate import BVPPostulate5_Quenches
from bhlff.core.bvp.postulates.tail_resonatorness_postulate import (
    BVPPostulate6_TailResonatorness,
)
from bhlff.core.bvp.postulates.transition_zone_postulate import (
    BVPPostulate7_TransitionZone,
)
from bhlff.core.bvp.postulates.core_renormalization_postulate import (
    BVPPostulate8_CoreRenormalization,
)
from bhlff.core.bvp.postulates.power_balance.power_balance_postulate import (
    PowerBalancePostulate,
)


class TestBVPPostulatesIntegrationPhysics:
    """Physical validation tests for BVP postulates integration."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for postulate testing."""
        return Domain(L=1.0, N=8, dimensions=3, N_phi=4, N_t=8, T=1.0)

    @pytest.fixture
    def bvp_constants(self):
        """Create BVP constants for postulate testing."""
        config = {
            "envelope_equation": {
                "kappa_0": 1.0,
                "kappa_2": 0.1,
                "chi_prime": 1.0,
                "chi_double_prime_0": 0.01,
                "k0_squared": 4.0,
            },
            "basic_material": {
                "mu": 1.0,
                "beta": 1.5,
                "lambda_param": 0.1,
            },
        }
        return BVPConstantsAdvanced(config)

    @pytest.fixture
    def test_envelope(self, domain_7d):
        """Create test envelope for postulate validation."""
        envelope = np.zeros(domain_7d.shape)

        # Create envelope with known properties
        center = domain_7d.N // 2
        envelope[
            center - 4 : center + 5,
            center - 4 : center + 5,
            center - 4 : center + 5,
            :,
            :,
            :,
            :,
        ] = 1.0

        # Add phase structure
        phi1 = np.linspace(0, 2 * np.pi, domain_7d.N_phi)
        phi2 = np.linspace(0, 2 * np.pi, domain_7d.N_phi)
        phi3 = np.linspace(0, 2 * np.pi, domain_7d.N_phi)

        PHI1, PHI2, PHI3 = np.meshgrid(phi1, phi2, phi3, indexing="ij")
        phase_factor = np.exp(1j * (PHI1 + PHI2 + PHI3))

        envelope = envelope * phase_factor

        return envelope

    def test_all_postulates_integration_physics(
        self, domain_7d, bvp_constants, test_envelope
    ):
        """
        Test integration of all BVP postulates.

        Physical Meaning:
            Validates that all 9 BVP postulates work together to ensure
            complete physical consistency of the BVP theory.

        Mathematical Foundation:
            Tests that all postulates are satisfied simultaneously,
            ensuring the complete BVP framework is physically consistent.
        """
        # Create all postulates
        postulates = [
            BVPPostulate1_CarrierPrimacy(domain_7d, bvp_constants),
            BVPPostulate2_ScaleSeparation(domain_7d, bvp_constants),
            BVPPostulate3_BVPRigidity(domain_7d, bvp_constants),
            BVPPostulate4_U1PhaseStructure(domain_7d, bvp_constants),
            BVPPostulate5_Quenches(domain_7d, bvp_constants),
            BVPPostulate6_TailResonatorness(domain_7d, bvp_constants),
            BVPPostulate7_TransitionZone(domain_7d, bvp_constants),
            BVPPostulate8_CoreRenormalization(domain_7d, bvp_constants),
            PowerBalancePostulate(domain_7d, bvp_constants),
        ]

        # Apply all postulates
        results = []
        for postulate in postulates:
            result = postulate.apply(test_envelope)
            results.append(result)

        # Physical validation 1: All postulates should be satisfied
        satisfied_count = sum(1 for result in results if result["satisfied"])
        assert satisfied_count == len(
            postulates
        ), f"Only {satisfied_count}/{len(postulates)} postulates satisfied"

        # Physical validation 2: Overall consistency should be high
        overall_consistency = satisfied_count / len(postulates)
        assert (
            overall_consistency > 0.8
        ), f"Low overall consistency: {overall_consistency}"

        # Physical validation 3: No contradictory results
        for i, result1 in enumerate(results):
            for j, result2 in enumerate(results):
                if i != j:
                    # Check for contradictions in key parameters
                    if "energy" in result1 and "energy" in result2:
                        energy_ratio = result1["energy"] / result2["energy"]
                        assert (
                            0.1 < energy_ratio < 10.0
                        ), f"Contradictory energy values: {result1['energy']} vs {result2['energy']}"

        # Physical validation 4: Physical parameters should be consistent
        self._validate_physical_consistency(results)

    def _validate_physical_consistency(self, results: List[Dict[str, Any]]) -> None:
        """Validate physical consistency across all postulate results."""
        # Extract key physical parameters
        energies = [r.get("energy", 0) for r in results if "energy" in r]
        frequencies = [r.get("frequency", 0) for r in results if "frequency" in r]
        coherence_values = [r.get("coherence", 0) for r in results if "coherence" in r]

        # Check energy consistency
        if energies:
            energy_variance = np.var(energies) / np.mean(energies) ** 2
            assert energy_variance < 0.1, f"High energy variance: {energy_variance}"

        # Check frequency consistency
        if frequencies:
            frequency_variance = np.var(frequencies) / np.mean(frequencies) ** 2
            assert (
                frequency_variance < 0.1
            ), f"High frequency variance: {frequency_variance}"

        # Check coherence consistency
        if coherence_values:
            coherence_variance = np.var(coherence_values)
            assert (
                coherence_variance < 0.1
            ), f"High coherence variance: {coherence_variance}"
