"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for level G particle validation models.

This module tests the particle inversion and validation for 7D phase field theory,
including parameter inversion and validation against experimental data.

Physical Meaning:
    Tests the inversion of model parameters from observable particle properties
    and validation of the results against experimental data.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from bhlff.models.level_g.validation import ParticleInversion, ParticleValidation


class TestParticleInversion:
    """
    Test particle parameter inversion.
    
    Physical Meaning:
        Tests the inversion of fundamental model parameters from
        observable particle properties.
    """
    
    def test_inversion_initialization(self):
        """Test inversion initialization."""
        observables = {
            'tail': 1.0,
            'jr': 0.5,
            'Achi': 0.3,
            'peaks': 2,
            'mobility': 0.8,
            'Meff': 1.2
        }
        
        priors = {
            'beta': [0.6, 1.4],
            'layers_count': [1, 3],
            'eta': [0.1, 0.3],
            'gamma': [0.1, 0.8],
            'tau': [0.5, 2.0],
            'q': [-2, 2]
        }
        
        loss_weights = {
            'tail': 1.0,
            'jr': 1.0,
            'Achi': 0.5,
            'peaks': 0.5,
            'mobility': 0.5,
            'Meff': 1.0
        }
        
        optimization_params = {
            'max_iterations': 100,
            'tolerance': 1e-6,
            'learning_rate': 0.01
        }
        
        inversion = ParticleInversion(
            observables, priors, loss_weights, optimization_params
        )
        
        assert inversion.observables == observables
        assert inversion.priors == priors
        assert inversion.loss_weights == loss_weights
        assert inversion.optimization_params == optimization_params
    
    def test_parameter_initialization(self):
        """Test parameter initialization."""
        observables = {
            'tail': 1.0,
            'jr': 0.5,
            'Achi': 0.3,
            'peaks': 2,
            'mobility': 0.8,
            'Meff': 1.2
        }
        
        priors = {
            'beta': [0.6, 1.4],
            'layers_count': [1, 3],
            'eta': [0.1, 0.3],
            'gamma': [0.1, 0.8],
            'tau': [0.5, 2.0],
            'q': [-2, 2]
        }
        
        inversion = ParticleInversion(observables, priors)
        initial_params = inversion._initialize_parameters()
        
        assert 'beta' in initial_params
        assert 'layers_count' in initial_params
        assert 'eta' in initial_params
        assert 'gamma' in initial_params
        assert 'tau' in initial_params
        assert 'q' in initial_params
        
        # Check that parameters are within prior ranges
        assert 0.6 <= initial_params['beta'] <= 1.4
        assert 1 <= initial_params['layers_count'] <= 3
        assert 0.1 <= initial_params['eta'] <= 0.3
        assert 0.1 <= initial_params['gamma'] <= 0.8
        assert 0.5 <= initial_params['tau'] <= 2.0
        assert -2 <= initial_params['q'] <= 2
    
    def test_loss_function_computation(self):
        """Test loss function computation."""
        observables = {
            'tail': 1.0,
            'jr': 0.5,
            'Achi': 0.3,
            'peaks': 2,
            'mobility': 0.8,
            'Meff': 1.2
        }
        
        priors = {
            'beta': [0.6, 1.4],
            'layers_count': [1, 3],
            'eta': [0.1, 0.3],
            'gamma': [0.1, 0.8],
            'tau': [0.5, 2.0],
            'q': [-2, 2]
        }
        
        inversion = ParticleInversion(observables, priors)
        
        # Test with some parameters
        params = {
            'beta': 1.0,
            'layers_count': 2,
            'eta': 0.2,
            'gamma': 0.5,
            'tau': 1.0,
            'q': 1
        }
        
        loss = inversion._compute_loss(params)
        
        assert loss >= 0
        assert np.isfinite(loss)
    
    def test_model_predictions_computation(self):
        """Test model predictions computation."""
        observables = {
            'tail': 1.0,
            'jr': 0.5,
            'Achi': 0.3,
            'peaks': 2,
            'mobility': 0.8,
            'Meff': 1.2
        }
        
        priors = {
            'beta': [0.6, 1.4],
            'layers_count': [1, 3],
            'eta': [0.1, 0.3],
            'gamma': [0.1, 0.8],
            'tau': [0.5, 2.0],
            'q': [-2, 2]
        }
        
        inversion = ParticleInversion(observables, priors)
        
        params = {
            'beta': 1.0,
            'layers_count': 2,
            'eta': 0.2,
            'gamma': 0.5,
            'tau': 1.0,
            'q': 1
        }
        
        predictions = inversion._compute_model_predictions(params)
        
        assert 'tail' in predictions
        assert 'jr' in predictions
        assert 'Achi' in predictions
        assert 'peaks' in predictions
        assert 'mobility' in predictions
        assert 'Meff' in predictions
        
        # Check that predictions are finite
        for key, value in predictions.items():
            assert np.isfinite(value)
    
    def test_distance_metric_computation(self):
        """Test distance metric computation."""
        observables = {
            'tail': 1.0,
            'jr': 0.5,
            'Achi': 0.3,
            'peaks': 2,
            'mobility': 0.8,
            'Meff': 1.2
        }
        
        priors = {
            'beta': [0.6, 1.4],
            'layers_count': [1, 3],
            'eta': [0.1, 0.3],
            'gamma': [0.1, 0.8],
            'tau': [0.5, 2.0],
            'q': [-2, 2]
        }
        
        inversion = ParticleInversion(observables, priors)
        
        # Test distance metrics
        distance1 = inversion._compute_distance_metric(1.0, 1.0, 'tail')
        distance2 = inversion._compute_distance_metric(1.0, 0.5, 'tail')
        distance3 = inversion._compute_distance_metric(2, 2, 'peaks')
        distance4 = inversion._compute_distance_metric(2, 3, 'peaks')
        
        assert distance1 == 0.0  # Perfect match
        assert distance2 > 0.0   # Mismatch
        assert distance3 == 0.0  # Perfect match for integer metric
        assert distance4 > 0.0   # Mismatch for integer metric
    
    def test_regularization_computation(self):
        """Test regularization computation."""
        observables = {
            'tail': 1.0,
            'jr': 0.5,
            'Achi': 0.3,
            'peaks': 2,
            'mobility': 0.8,
            'Meff': 1.2
        }
        
        priors = {
            'beta': [0.6, 1.4],
            'layers_count': [1, 3],
            'eta': [0.1, 0.3],
            'gamma': [0.1, 0.8],
            'tau': [0.5, 2.0],
            'q': [-2, 2]
        }
        
        inversion = ParticleInversion(observables, priors)
        
        # Test with parameters within bounds
        params_within_bounds = {
            'beta': 1.0,
            'layers_count': 2,
            'eta': 0.2,
            'gamma': 0.5,
            'tau': 1.0,
            'q': 1
        }
        
        regularization1 = inversion._compute_regularization(params_within_bounds)
        assert regularization1 >= 0
        
        # Test with parameters outside bounds
        params_outside_bounds = {
            'beta': 2.0,  # Outside bounds
            'layers_count': 2,
            'eta': 0.2,
            'gamma': 0.5,
            'tau': 1.0,
            'q': 1
        }
        
        regularization2 = inversion._compute_regularization(params_outside_bounds)
        assert regularization2 > regularization1  # Should be higher for out-of-bounds
    
    def test_gradients_computation(self):
        """Test gradients computation."""
        observables = {
            'tail': 1.0,
            'jr': 0.5,
            'Achi': 0.3,
            'peaks': 2,
            'mobility': 0.8,
            'Meff': 1.2
        }
        
        priors = {
            'beta': [0.6, 1.4],
            'layers_count': [1, 3],
            'eta': [0.1, 0.3],
            'gamma': [0.1, 0.8],
            'tau': [0.5, 2.0],
            'q': [-2, 2]
        }
        
        inversion = ParticleInversion(observables, priors)
        
        params = {
            'beta': 1.0,
            'layers_count': 2,
            'eta': 0.2,
            'gamma': 0.5,
            'tau': 1.0,
            'q': 1
        }
        
        gradients = inversion._compute_gradients(params)
        
        assert 'beta' in gradients
        assert 'layers_count' in gradients
        assert 'eta' in gradients
        assert 'gamma' in gradients
        assert 'tau' in gradients
        assert 'q' in gradients
        
        # Check that gradients are finite
        for key, value in gradients.items():
            assert np.isfinite(value)
    
    def test_parameter_uncertainties_computation(self):
        """Test parameter uncertainties computation."""
        observables = {
            'tail': 1.0,
            'jr': 0.5,
            'Achi': 0.3,
            'peaks': 2,
            'mobility': 0.8,
            'Meff': 1.2
        }
        
        priors = {
            'beta': [0.6, 1.4],
            'layers_count': [1, 3],
            'eta': [0.1, 0.3],
            'gamma': [0.1, 0.8],
            'tau': [0.5, 2.0],
            'q': [-2, 2]
        }
        
        inversion = ParticleInversion(observables, priors)
        
        params = {
            'beta': 1.0,
            'layers_count': 2,
            'eta': 0.2,
            'gamma': 0.5,
            'tau': 1.0,
            'q': 1
        }
        
        uncertainties = inversion._compute_parameter_uncertainties(params)
        
        assert 'beta' in uncertainties
        assert 'layers_count' in uncertainties
        assert 'eta' in uncertainties
        assert 'gamma' in uncertainties
        assert 'tau' in uncertainties
        assert 'q' in uncertainties
        
        # Check that uncertainties are positive
        for key, value in uncertainties.items():
            assert value > 0
            assert np.isfinite(value)
    
    def test_inversion_optimization(self):
        """Test inversion optimization."""
        observables = {
            'tail': 1.0,
            'jr': 0.5,
            'Achi': 0.3,
            'peaks': 2,
            'mobility': 0.8,
            'Meff': 1.2
        }
        
        priors = {
            'beta': [0.6, 1.4],
            'layers_count': [1, 3],
            'eta': [0.1, 0.3],
            'gamma': [0.1, 0.8],
            'tau': [0.5, 2.0],
            'q': [-2, 2]
        }
        
        optimization_params = {
            'max_iterations': 10,  # Small number for testing
            'tolerance': 1e-6,
            'learning_rate': 0.01
        }
        
        inversion = ParticleInversion(observables, priors, optimization_params=optimization_params)
        results = inversion.invert_parameters()
        
        assert 'optimized_parameters' in results
        assert 'final_loss' in results
        assert 'convergence_info' in results
        assert 'parameter_uncertainties' in results
        
        # Check that optimized parameters are within bounds
        optimized_params = results['optimized_parameters']
        for param_name, param_value in optimized_params.items():
            if param_name in priors:
                prior_range = priors[param_name]
                if isinstance(prior_range, list) and len(prior_range) == 2:
                    min_val, max_val = prior_range
                    assert min_val <= param_value <= max_val
        
        # Check that final loss is finite
        assert np.isfinite(results['final_loss'])
        assert results['final_loss'] >= 0


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
            'optimized_parameters': {
                'beta': 1.0,
                'layers_count': 2,
                'eta': 0.2,
                'gamma': 0.5,
                'tau': 1.0,
                'q': 1
            },
            'final_loss': 0.1,
            'parameter_uncertainties': {
                'beta': 0.05,
                'layers_count': 0.1,
                'eta': 0.02,
                'gamma': 0.05,
                'tau': 0.1,
                'q': 0.1
            }
        }
        
        validation_criteria = {
            'chi_squared_threshold': 0.05,
            'confidence_level': 0.95,
            'parameter_tolerance': 0.01
        }
        
        experimental_data = {
            'mass': 9.10938356e-31,
            'charge': -1.602176634e-19,
            'magnetic_moment': -9.2847647043e-24
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
            'optimized_parameters': {
                'beta': 1.0,
                'layers_count': 2,
                'eta': 0.2,
                'gamma': 0.5,
                'tau': 1.0,
                'q': 1
            },
            'parameter_uncertainties': {
                'beta': 0.05,
                'layers_count': 0.1,
                'eta': 0.02,
                'gamma': 0.05,
                'tau': 0.1,
                'q': 0.1
            }
        }
        
        validation_criteria = {
            'chi_squared_threshold': 0.05,
            'confidence_level': 0.95,
            'parameter_tolerance': 0.01
        }
        
        validation = ParticleValidation(inversion_results, validation_criteria)
        parameter_validation = validation._validate_parameters()
        
        assert 'beta' in parameter_validation
        assert 'layers_count' in parameter_validation
        assert 'eta' in parameter_validation
        assert 'gamma' in parameter_validation
        assert 'tau' in parameter_validation
        assert 'q' in parameter_validation
        
        # Check that validation results are boolean
        for key, value in parameter_validation.items():
            assert isinstance(value, bool)
    
    def test_energy_balance_validation(self):
        """Test energy balance validation."""
        inversion_results = {
            'optimized_parameters': {
                'beta': 1.0,
                'layers_count': 2,
                'eta': 0.2,
                'gamma': 0.5,
                'tau': 1.0,
                'q': 1
            }
        }
        
        validation_criteria = {
            'energy_balance_tolerance': 0.03
        }
        
        validation = ParticleValidation(inversion_results, validation_criteria)
        energy_validation = validation._validate_energy_balance()
        
        assert 'total_energy_conserved' in energy_validation
        assert 'kinetic_energy_positive' in energy_validation
        assert 'potential_energy_positive' in energy_validation
        assert 'energy_balance_residual' in energy_validation
        
        # Check that validation results are boolean
        for key, value in energy_validation.items():
            if key != 'energy_balance_residual':
                assert isinstance(value, bool)
            else:
                assert isinstance(value, (int, float))
    
    def test_physical_constraint_validation(self):
        """Test physical constraint validation."""
        inversion_results = {
            'optimized_parameters': {
                'beta': 1.0,
                'layers_count': 2,
                'eta': 0.2,
                'gamma': 0.5,
                'tau': 1.0,
                'q': 1
            }
        }
        
        validation_criteria = {
            'passivity_threshold': 0.0
        }
        
        validation = ParticleValidation(inversion_results, validation_criteria)
        constraint_validation = validation._validate_physical_constraints()
        
        assert 'passivity_constraint' in constraint_validation
        assert 'causality_constraint' in constraint_validation
        assert 'unitarity_constraint' in constraint_validation
        assert 'gauge_invariance' in constraint_validation
        
        # Check that validation results are boolean
        for key, value in constraint_validation.items():
            assert isinstance(value, bool)
    
    def test_experimental_validation(self):
        """Test experimental validation."""
        inversion_results = {
            'optimized_parameters': {
                'beta': 1.0,
                'layers_count': 2,
                'eta': 0.2,
                'gamma': 0.5,
                'tau': 1.0,
                'q': 1
            }
        }
        
        validation_criteria = {
            'chi_squared_threshold': 0.05
        }
        
        experimental_data = {
            'mass': 9.10938356e-31,
            'charge': -1.602176634e-19,
            'magnetic_moment': -9.2847647043e-24
        }
        
        validation = ParticleValidation(inversion_results, validation_criteria, experimental_data)
        experimental_validation = validation._validate_experimental_data()
        
        assert 'mass_spectrum_agreement' in experimental_validation
        assert 'charge_spectrum_agreement' in experimental_validation
        assert 'magnetic_moment_agreement' in experimental_validation
        assert 'lifetime_agreement' in experimental_validation
        
        # Check that validation results are boolean
        for key, value in experimental_validation.items():
            assert isinstance(value, bool)
    
    def test_overall_validation(self):
        """Test overall validation."""
        inversion_results = {
            'optimized_parameters': {
                'beta': 1.0,
                'layers_count': 2,
                'eta': 0.2,
                'gamma': 0.5,
                'tau': 1.0,
                'q': 1
            }
        }
        
        validation_criteria = {
            'chi_squared_threshold': 0.05,
            'confidence_level': 0.95,
            'parameter_tolerance': 0.01
        }
        
        validation = ParticleValidation(inversion_results, validation_criteria)
        overall_validation = validation._compute_overall_validation()
        
        assert 'all_tests_passed' in overall_validation
        assert 'validation_score' in overall_validation
        assert 'critical_failures' in overall_validation
        assert 'warnings' in overall_validation
        assert 'recommendations' in overall_validation
        
        # Check that validation results have expected types
        assert isinstance(overall_validation['all_tests_passed'], bool)
        assert isinstance(overall_validation['validation_score'], (int, float))
        assert isinstance(overall_validation['critical_failures'], list)
        assert isinstance(overall_validation['warnings'], list)
        assert isinstance(overall_validation['recommendations'], list)
    
    def test_full_validation_process(self):
        """Test full validation process."""
        inversion_results = {
            'optimized_parameters': {
                'beta': 1.0,
                'layers_count': 2,
                'eta': 0.2,
                'gamma': 0.5,
                'tau': 1.0,
                'q': 1
            },
            'parameter_uncertainties': {
                'beta': 0.05,
                'layers_count': 0.1,
                'eta': 0.02,
                'gamma': 0.05,
                'tau': 0.1,
                'q': 0.1
            }
        }
        
        validation_criteria = {
            'chi_squared_threshold': 0.05,
            'confidence_level': 0.95,
            'parameter_tolerance': 0.01,
            'energy_balance_tolerance': 0.03,
            'passivity_threshold': 0.0,
            'em_tolerance': 0.1
        }
        
        experimental_data = {
            'mass': 9.10938356e-31,
            'charge': -1.602176634e-19,
            'magnetic_moment': -9.2847647043e-24
        }
        
        validation = ParticleValidation(inversion_results, validation_criteria, experimental_data)
        results = validation.validate_parameters()
        
        assert 'parameter_validation' in results
        assert 'energy_balance_validation' in results
        assert 'physical_constraint_validation' in results
        assert 'experimental_validation' in results
        assert 'overall_validation' in results
        
        # Check that all validation results are present
        for key, value in results.items():
            assert isinstance(value, dict)
            assert len(value) > 0
