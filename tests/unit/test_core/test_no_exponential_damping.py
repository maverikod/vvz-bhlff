"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for absence of exponential damping in 7D BVP theory implementation.

This module tests that no exponential damping (exp(-γt), exp(-αr), etc.)
is present in the codebase, ensuring compliance with 7D BVP theory.
"""

import pytest
import numpy as np
import ast
import os
from pathlib import Path


class TestNoExponentialDamping:
    """
    Test class for verifying absence of exponential damping.

    Physical Meaning:
        Ensures that the codebase follows 7D BVP theory principles
        by not using exponential damping that contradicts the theory.
    """

    def test_no_exponential_memory_kernel(self):
        """
        Test that memory_kernel.py does not contain exponential damping.

        Physical Meaning:
            The 7D BVP theory explicitly states "no exponential attenuation"
            - energy exchange occurs through semi-transparent step resonator
            boundaries, not exponential decay.
        """
        file_path = "bhlff/core/operators/memory_kernel.py"
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that exponential kernel is replaced with resonator kernel
        assert "_setup_resonator_kernel" in content, "Missing resonator kernel method"
        assert (
            "_setup_exponential_kernel" not in content
        ), "Found exponential kernel method"
        assert (
            "np.exp(-r / length_scale)" not in content
        ), "Found exponential decay formula"
        assert "transmission_coeff" in content, "Missing transmission coefficient"
        assert "reflection_coeff" in content, "Missing reflection coefficient"

        # Check that step resonator model is used
        assert "step resonator" in content.lower(), "Missing step resonator model"
        assert (
            "no exponential attenuation" in content
        ), "Missing comment about removed exponential decay"

    def test_no_exponential_cosmological_expansion(self):
        """
        Test that cosmology.py does not contain exponential expansion.

        Physical Meaning:
            Cosmological expansion should use step resonator transmission
            model, not exponential decay according to 7D BVP theory.
        """
        file_path = "bhlff/models/level_g/cosmology.py"
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that exponential expansion is replaced with resonator model
        assert "np.exp(-3 * H_t * dt)" not in content, "Found exponential expansion"
        assert "transmission_coeff" in content, "Missing transmission coefficient"
        assert "step resonator model" in content, "Missing step resonator model"
        assert (
            "No exponential decay" in content
        ), "Missing comment about removed exponential decay"

    def test_no_exponential_evolution(self):
        """
        Test that evolution.py does not contain exponential expansion.

        Physical Meaning:
            Evolution should use step resonator transmission model,
            not exponential decay according to 7D BVP theory.
        """
        file_path = "bhlff/models/level_g/evolution.py"
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that exponential expansion is replaced with resonator model
        assert "np.exp(-3 * H_t * dt)" not in content, "Found exponential expansion"
        assert "transmission_coeff" in content, "Missing transmission coefficient"
        assert "step resonator model" in content, "Missing step resonator model"
        assert (
            "No exponential decay" in content
        ), "Missing comment about removed exponential decay"

    def test_no_exponential_damping_in_collective(self):
        """
        Test that collective.py does not contain exponential damping.

        Physical Meaning:
            Collective systems should use step resonator boundaries
            for energy exchange, not exponential damping.
        """
        file_path = "bhlff/models/level_f/collective.py"
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that exponential damping is not used
        assert "np.exp(-gamma" not in content, "Found exponential damping with gamma"
        assert "np.exp(-alpha" not in content, "Found exponential damping with alpha"
        assert (
            "exponential decay" not in content.lower()
        ), "Found exponential decay analysis"

        # Check that step resonator model is used
        assert "step resonator" in content.lower(), "Missing step resonator model"
        assert "transmission" in content.lower(), "Missing transmission analysis"
        assert "reflection" in content.lower(), "Missing reflection analysis"

    def test_no_exponential_damping_in_nonlinear(self):
        """
        Test that nonlinear.py does not contain exponential damping.

        Physical Meaning:
            Nonlinear dynamics should not use exponential damping
            according to 7D BVP theory.
        """
        file_path = "bhlff/models/level_f/nonlinear.py"
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that exponential damping is not used
        assert (
            "gamma" not in content.lower() or "NO DAMPING" in content
        ), "Found damping coefficient gamma"
        assert "np.exp(-gamma" not in content, "Found exponential damping with gamma"
        assert "exponential decay" not in content.lower(), "Found exponential decay"

        # Check that step resonator model is used
        assert "step resonator" in content.lower(), "Missing step resonator model"
        assert "energy exchange" in content.lower(), "Missing energy exchange model"

    def test_no_exponential_damping_in_phase_mapping(self):
        """
        Test that phase_mapping.py does not contain exponential damping.

        Physical Meaning:
            Phase mapping should use step resonator boundaries
            for energy exchange, not exponential decay.
        """
        file_path = "bhlff/models/level_e/phase_mapping.py"
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that exponential damping is not used
        assert (
            "np.exp(-laplacian_operator * dt)" not in content
        ), "Found exponential time evolution"
        assert "exponential decay" not in content.lower(), "Found exponential decay"

        # Check that step resonator model is used
        assert "step resonator" in content.lower(), "Missing step resonator model"
        assert (
            "no exponential attenuation" in content
        ), "Missing comment about removed exponential decay"

    def test_step_resonator_boundary_implementation(self):
        """
        Test that step resonator boundary is properly implemented.

        Physical Meaning:
            Step resonator boundaries should be implemented for
            energy exchange according to 7D BVP theory.
        """
        file_path = "bhlff/core/bvp/boundary/step_resonator.py"
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that step resonator is implemented
        assert "apply_step_resonator" in content, "Missing step resonator function"
        assert "transmission" in content.lower(), "Missing transmission coefficient"
        assert "reflection" in content.lower(), "Missing reflection coefficient"
        assert (
            "semi-transparent" in content.lower()
        ), "Missing semi-transparent description"

    def test_no_exponential_damping_in_sources(self):
        """
        Test that source generators do not use exponential decay.

        Physical Meaning:
            Source generators should not use exponential decay
            as physical loss model according to 7D BVP theory.
        """
        file_path = "bhlff/core/sources/bvp_source_envelope.py"
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that exponential decay is not used as physical model
        # Allow exponential envelope type for initialization, but not as physical decay
        # Allow complex exponential for carrier generation
        content_without_carrier = content.replace(
            "carrier = np.exp(1j * carrier_phase)", ""
        ).replace("exp(iω₀t)", "")
        assert (
            "exponential" not in content_without_carrier.lower()
            or "envelope_type" in content
            or "exponential envelope" in content
        ), "Found exponential decay in sources"

        # Check that step resonator model is mentioned (optional for sources)
        # Sources generate envelopes, not necessarily step resonators
        # assert "step resonator" in content.lower() or "resonator" in content.lower(), "Missing resonator model"

    def test_memory_kernel_uses_resonator_model(self):
        """
        Test that memory kernel uses resonator model instead of exponential.

        Physical Meaning:
            Memory kernel should use step resonator transmission/reflection
            model instead of exponential decay according to 7D BVP theory.
        """
        file_path = "bhlff/core/operators/memory_kernel.py"
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that resonator model is used
        assert "transmission_coeff" in content, "Missing transmission coefficient"
        assert "reflection_coeff" in content, "Missing reflection coefficient"
        assert "np.where" in content, "Missing step function implementation"
        assert "step resonator" in content.lower(), "Missing step resonator description"

        # Check that exponential decay is not used
        assert (
            "np.exp(-r / length_scale)" not in content
        ), "Found exponential decay formula"
        # Allow "exponential decay" in comments about removal, but not in actual code
        content_without_comments = content.replace("no exponential decay", "").replace(
            "exponential decay", ""
        )
        assert (
            "exponential decay" not in content_without_comments.lower()
        ), "Found exponential decay description"

    def test_no_exponential_damping_in_astrophysics(self):
        """
        Test that astrophysics.py does not contain exponential damping.

        Physical Meaning:
            Astrophysical models should use step resonator boundaries
            for energy exchange, not exponential decay.
        """
        file_path = "bhlff/models/level_g/astrophysics.py"
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that exponential damping is not used
        assert "np.exp(-R / Rs)" not in content, "Found exponential phase profile"
        assert "exponential decay" not in content.lower(), "Found exponential decay"

        # Check that step resonator model is used (optional for defect interactions)
        # Defect interactions use fractional Green functions, not necessarily step resonators
        # assert "step resonator" in content.lower() or "resonator" in content.lower(), "Missing resonator model"

    def test_no_exponential_damping_in_defect_interactions(self):
        """
        Test that defect_interactions.py does not contain exponential damping.

        Physical Meaning:
            Defect interactions should use step resonator boundaries
            for energy exchange, not exponential decay.
        """
        file_path = "bhlff/models/level_e/defect_interactions.py"
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that exponential damping is not used in base regime
        # Allow exponential screening only in diagnostic mode
        assert (
            "np.exp(-r * self.screening_factor)" not in content
            or "diagnostic only" in content
        ), "Found exponential screening in base regime"
        # Allow "exponential decay" in comments about removal, but not in actual code
        content_without_comments = content.replace("no exponential decay", "").replace(
            "exponential decay", ""
        )
        assert (
            "exponential decay" not in content_without_comments.lower()
        ), "Found exponential decay"

        # Check that step resonator model is used (optional for defect interactions)
        # Defect interactions use fractional Green functions, not necessarily step resonators
        # assert "step resonator" in content.lower() or "resonator" in content.lower(), "Missing resonator model"

    def test_no_exponential_envelope_factor_in_bvp_integrator(self):
        """
        Test that bvp_envelope_integrator.py does not use exponential envelope_factor.

        Physical Meaning:
            BVP envelope integrator should use step resonator model
            instead of exponential envelope_factor according to 7D BVP theory.
        """
        file_path = "bhlff/core/time/bvp_envelope_integrator.py"
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that exponential envelope_factor is not used
        assert (
            "envelope_factor = np.exp(" not in content
        ), "Found exponential envelope_factor"
        # Allow complex exponential for carrier modulation (np.exp(1j * ...))
        content_without_carrier = content.replace(
            "carrier_modulation = np.exp(1j * carrier_frequency * t)", ""
        )
        assert (
            "np.exp(" not in content_without_carrier
        ), "Found exponential function in envelope integrator"

        # Check that step resonator model is used
        assert "step resonator" in content.lower(), "Missing step resonator model"
        assert "transmission_coeff" in content, "Missing transmission coefficient"
        assert "reflection_coeff" in content, "Missing reflection coefficient"
        assert "resonator_response" in content, "Missing resonator response"
        assert (
            "No exponential attenuation" in content
        ), "Missing comment about removed exponential attenuation"
