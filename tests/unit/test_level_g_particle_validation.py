"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for level G particle validation models.

This module tests the particle validation for 7D phase field theory,
including validation against experimental data and physical constraints.

Physical Meaning:
    Tests the validation of inverted parameters against
    experimental data and physical constraints.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from bhlff.models.level_g.validation import ParticleValidation


class TestParticleValidation:
    """
    Test particle validation.

    Physical Meaning:
        Tests the validation of inverted parameters against
        experimental data and physical constraints.
    """

    def test_validation_initialization(self):
        """Test validation initialization."""
        inversion_results = {
            "optimized_parameters": {
                "beta": 1.0,
                "layers_count": 2,
                "eta": 0.2,
                "gamma": 0.5,
                "tau": 1.0,
                "q": 1,
            },
            "final_loss": 0.1,
            "parameter_uncertainties": {
                "beta": 0.05,
                "layers_count": 0.1,
                "eta": 0.02,
                "gamma": 0.05,
                "tau": 0.1,
                "q": 0.1,
            },
        }

        validation_criteria = {
            "chi_squared_threshold": 0.05,
            "confidence_level": 0.95,
            "parameter_tolerance": 0.01,
        }

        experimental_data = {
            "mass": 9.10938356e-31,
            "charge": -1.602176634e-19,
            "magnetic_moment": -9.2847647043e-24,
        }

        validation = ParticleValidation(
            inversion_results, validation_criteria, experimental_data
        )

        assert validation.inversion_results == inversion_results
        assert validation.validation_criteria == validation_criteria
        assert validation.experimental_data == experimental_data

    def test_parameter_validation(self):
        """Test parameter validation."""
        inversion_results = {
            "optimized_parameters": {
                "beta": 1.0,
                "layers_count": 2,
                "eta": 0.2,
                "gamma": 0.5,
                "tau": 1.0,
                "q": 1,
            },
            "parameter_uncertainties": {
                "beta": 0.05,
                "layers_count": 0.1,
                "eta": 0.02,
                "gamma": 0.05,
                "tau": 0.1,
                "q": 0.1,
            },
        }

        validation_criteria = {
            "chi_squared_threshold": 0.05,
            "confidence_level": 0.95,
            "parameter_tolerance": 0.01,
        }

        validation = ParticleValidation(inversion_results, validation_criteria)
        parameter_validation = validation._validate_parameters()

        assert "beta" in parameter_validation
        assert "layers_count" in parameter_validation
        assert "eta" in parameter_validation
        assert "gamma" in parameter_validation
        assert "tau" in parameter_validation
        assert "q" in parameter_validation

        # Check that validation results are boolean
        for key, value in parameter_validation.items():
            assert isinstance(value, bool)

    def test_energy_balance_validation(self):
        """Test energy balance validation."""
        inversion_results = {
            "optimized_parameters": {
                "beta": 1.0,
                "layers_count": 2,
                "eta": 0.2,
                "gamma": 0.5,
                "tau": 1.0,
                "q": 1,
            }
        }

        validation_criteria = {"energy_balance_tolerance": 0.03}

        validation = ParticleValidation(inversion_results, validation_criteria)
        energy_validation = validation._validate_energy_balance()

        assert "total_energy_conserved" in energy_validation
        assert "kinetic_energy_positive" in energy_validation
        assert "potential_energy_positive" in energy_validation
        assert "energy_balance_residual" in energy_validation

        # Check that validation results have expected types
        assert isinstance(energy_validation["total_energy_conserved"], bool)
        assert isinstance(energy_validation["kinetic_energy_positive"], bool)
        assert isinstance(energy_validation["potential_energy_positive"], bool)
        assert isinstance(energy_validation["energy_balance_residual"], (int, float))

    def test_physical_constraint_validation(self):
        """Test physical constraint validation."""
        inversion_results = {
            "optimized_parameters": {
                "beta": 1.0,
                "layers_count": 2,
                "eta": 0.2,
                "gamma": 0.5,
                "tau": 1.0,
                "q": 1,
            }
        }

        validation_criteria = {"passivity_threshold": 0.0}

        validation = ParticleValidation(inversion_results, validation_criteria)
        constraint_validation = validation._validate_physical_constraints()

        assert "passivity_constraint" in constraint_validation
        assert "causality_constraint" in constraint_validation
        assert "unitarity_constraint" in constraint_validation
        assert "gauge_invariance" in constraint_validation

        # Check that validation results are boolean
        for key, value in constraint_validation.items():
            assert isinstance(value, bool)

    def test_experimental_validation(self):
        """Test experimental validation."""
        inversion_results = {
            "optimized_parameters": {
                "beta": 1.0,
                "layers_count": 2,
                "eta": 0.2,
                "gamma": 0.5,
                "tau": 1.0,
                "q": 1,
            }
        }

        validation_criteria = {"chi_squared_threshold": 0.05}

        experimental_data = {
            "mass": 9.10938356e-31,
            "charge": -1.602176634e-19,
            "magnetic_moment": -9.2847647043e-24,
        }

        validation = ParticleValidation(
            inversion_results, validation_criteria, experimental_data
        )
        experimental_validation = validation._validate_experimental_data()

        assert "mass_spectrum_agreement" in experimental_validation
        assert "charge_spectrum_agreement" in experimental_validation
        assert "magnetic_moment_agreement" in experimental_validation
        assert "lifetime_agreement" in experimental_validation

        # Check that validation results are boolean
        for key, value in experimental_validation.items():
            assert isinstance(value, bool)

    def test_overall_validation(self):
        """Test overall validation."""
        inversion_results = {
            "optimized_parameters": {
                "beta": 1.0,
                "layers_count": 2,
                "eta": 0.2,
                "gamma": 0.5,
                "tau": 1.0,
                "q": 1,
            }
        }

        validation_criteria = {
            "chi_squared_threshold": 0.05,
            "confidence_level": 0.95,
            "parameter_tolerance": 0.01,
        }

        validation = ParticleValidation(inversion_results, validation_criteria)
        overall_validation = validation._compute_overall_validation()

        assert "all_tests_passed" in overall_validation
        assert "validation_score" in overall_validation
        assert "critical_failures" in overall_validation
        assert "warnings" in overall_validation
        assert "recommendations" in overall_validation

        # Check that validation results have expected types
        assert isinstance(overall_validation["all_tests_passed"], bool)
        assert isinstance(overall_validation["validation_score"], (int, float))
        assert isinstance(overall_validation["critical_failures"], list)
        assert isinstance(overall_validation["warnings"], list)
        assert isinstance(overall_validation["recommendations"], list)

    def test_full_validation_process(self):
        """Test full validation process."""
        inversion_results = {
            "optimized_parameters": {
                "beta": 1.0,
                "layers_count": 2,
                "eta": 0.2,
                "gamma": 0.5,
                "tau": 1.0,
                "q": 1,
            },
            "parameter_uncertainties": {
                "beta": 0.05,
                "layers_count": 0.1,
                "eta": 0.02,
                "gamma": 0.05,
                "tau": 0.1,
                "q": 0.1,
            },
        }

        validation_criteria = {
            "chi_squared_threshold": 0.05,
            "confidence_level": 0.95,
            "parameter_tolerance": 0.01,
            "energy_balance_tolerance": 0.03,
            "passivity_threshold": 0.0,
            "em_tolerance": 0.1,
        }

        experimental_data = {
            "mass": 9.10938356e-31,
            "charge": -1.602176634e-19,
            "magnetic_moment": -9.2847647043e-24,
        }

        validation = ParticleValidation(
            inversion_results, validation_criteria, experimental_data
        )
        results = validation.validate_parameters()

        assert "parameter_validation" in results
        assert "energy_balance_validation" in results
        assert "physical_constraint_validation" in results
        assert "experimental_validation" in results
        assert "overall_validation" in results

        # Check that all validation results are present
        for key, value in results.items():
            assert isinstance(value, dict)
            assert len(value) > 0
