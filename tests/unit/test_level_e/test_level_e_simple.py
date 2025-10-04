"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Simple tests for Level E experiments.

This module implements basic tests for Level E experiments including
sensitivity analysis, robustness testing, and soliton/defect models.

Physical Meaning:
    Tests the fundamental functionality of Level E experiments,
    ensuring that sensitivity analysis, robustness testing, and
    soliton/defect models work correctly.

Example:
    >>> pytest tests/unit/test_level_e/test_level_e_simple.py
"""

import pytest
import numpy as np
import json
import tempfile
import os
from typing import Dict, Any

from bhlff.models.level_e import (
    SensitivityAnalyzer,
    RobustnessTester,
    DiscretizationAnalyzer,
    FailureDetector,
    PhaseMapper,
    PerformanceAnalyzer,
    SolitonModel,
    BaryonSoliton,
    SkyrmionSoliton,
    DefectModel,
    VortexDefect,
    MultiDefectSystem,
    LevelEExperiments
)


class TestSensitivityAnalyzer:
    """Test sensitivity analysis functionality."""
    
    def test_initialization(self):
        """Test SensitivityAnalyzer initialization."""
        parameter_ranges = {
            'beta': (0.6, 1.4),
            'mu': (0.5, 1.5),
            'eta': (0.0, 0.3)
        }
        
        analyzer = SensitivityAnalyzer(parameter_ranges)
        
        assert analyzer.param_ranges == parameter_ranges
        assert analyzer.n_params == 3
        assert 'beta' in analyzer.param_names
        assert 'mu' in analyzer.param_names
        assert 'eta' in analyzer.param_names
    
    def test_lhs_sampling(self):
        """Test Latin Hypercube sampling."""
        parameter_ranges = {
            'beta': (0.6, 1.4),
            'mu': (0.5, 1.5)
        }
        
        analyzer = SensitivityAnalyzer(parameter_ranges)
        samples = analyzer.generate_lhs_samples(100)
        
        assert samples.shape == (100, 2)
        assert np.all(samples[:, 0] >= 0.6)  # beta range
        assert np.all(samples[:, 0] <= 1.4)
        assert np.all(samples[:, 1] >= 0.5)  # mu range
        assert np.all(samples[:, 1] <= 1.5)
    
    def test_sobol_indices(self):
        """Test Sobol index computation."""
        parameter_ranges = {
            'beta': (0.6, 1.4),
            'mu': (0.5, 1.5)
        }
        
        analyzer = SensitivityAnalyzer(parameter_ranges)
        
        # Generate test data
        samples = analyzer.generate_lhs_samples(50)
        outputs = np.random.normal(0, 1, 50)
        
        sobol_indices = analyzer.compute_sobol_indices(samples, outputs)
        
        assert 'beta' in sobol_indices
        assert 'mu' in sobol_indices
        assert 'first_order' in sobol_indices['beta']
        assert 'total_order' in sobol_indices['beta']
        assert 'interaction' in sobol_indices['beta']
    
    def test_parameter_sensitivity_analysis(self):
        """Test complete parameter sensitivity analysis."""
        parameter_ranges = {
            'beta': (0.6, 1.4),
            'mu': (0.5, 1.5)
        }
        
        analyzer = SensitivityAnalyzer(parameter_ranges)
        results = analyzer.analyze_parameter_sensitivity(n_samples=100)
        
        assert 'samples' in results
        assert 'outputs' in results
        assert 'sobol_indices' in results
        assert 'parameter_ranking' in results
        assert 'stability_metrics' in results
        
        # Check that results are reasonable
        assert len(results['samples']) == 100
        assert len(results['outputs']) == 100
        assert len(results['parameter_ranking']) == 2


class TestRobustnessTester:
    """Test robustness testing functionality."""
    
    def test_initialization(self):
        """Test RobustnessTester initialization."""
        base_config = {
            'beta': 1.0,
            'mu': 1.0,
            'eta': 0.1
        }
        
        tester = RobustnessTester(base_config)
        
        assert tester.base_config == base_config
    
    def test_noise_robustness(self):
        """Test noise robustness testing."""
        base_config = {
            'beta': 1.0,
            'mu': 1.0,
            'eta': 0.1
        }
        
        tester = RobustnessTester(base_config)
        noise_levels = [0.0, 0.05, 0.1]
        
        results = tester.test_noise_robustness(noise_levels)
        
        assert len(results) == 3
        assert 0.0 in results
        assert 0.05 in results
        assert 0.1 in results
        
        # Check result structure
        for noise_level, result in results.items():
            assert 'degradation' in result
            assert 'passive_violations' in result
            assert 'topological_stability' in result
    
    def test_parameter_uncertainty(self):
        """Test parameter uncertainty testing."""
        base_config = {
            'beta': 1.0,
            'mu': 1.0,
            'eta': 0.1
        }
        
        tester = RobustnessTester(base_config)
        uncertainty_ranges = {
            'beta': 0.05,
            'mu': 0.1
        }
        
        results = tester.test_parameter_uncertainty(uncertainty_ranges)
        
        assert 'beta' in results
        assert 'mu' in results
        
        # Check result structure
        for param_name, result in results.items():
            assert 'uncertainty' in result
            assert 'variations' in result
            assert 'outputs' in result
            assert 'sensitivity' in result
    
    def test_geometry_perturbations(self):
        """Test geometry perturbation testing."""
        base_config = {
            'beta': 1.0,
            'mu': 1.0,
            'eta': 0.1
        }
        
        tester = RobustnessTester(base_config)
        perturbation_types = ['boundary_jitter', 'domain_deformation']
        
        results = tester.test_geometry_perturbations(perturbation_types)
        
        assert len(results) == 2
        assert 'boundary_jitter' in results
        assert 'domain_deformation' in results
        
        # Check result structure
        for perturbation_type, result in results.items():
            assert 'perturbed_configs' in result
            assert 'outputs' in result
            assert 'sensitivity' in result


class TestDiscretizationAnalyzer:
    """Test discretization effects analysis functionality."""
    
    def test_initialization(self):
        """Test DiscretizationAnalyzer initialization."""
        reference_config = {
            'L': 20.0,
            'N': 256,
            'beta': 1.0,
            'mu': 1.0
        }
        
        analyzer = DiscretizationAnalyzer(reference_config)
        
        assert analyzer.reference_config == reference_config
    
    def test_grid_convergence(self):
        """Test grid convergence analysis."""
        reference_config = {
            'L': 20.0,
            'N': 256,
            'beta': 1.0,
            'mu': 1.0
        }
        
        analyzer = DiscretizationAnalyzer(reference_config)
        grid_sizes = [64, 128, 256]
        
        results = analyzer.analyze_grid_convergence(grid_sizes)
        
        assert 'grid_results' in results
        assert 'convergence_analysis' in results
        assert 'recommended_grid_size' in results
        
        # Check that all grid sizes are tested
        assert len(results['grid_results']) == 3
        assert 64 in results['grid_results']
        assert 128 in results['grid_results']
        assert 256 in results['grid_results']
    
    def test_domain_size_effects(self):
        """Test domain size effects analysis."""
        reference_config = {
            'L': 20.0,
            'N': 256,
            'beta': 1.0,
            'mu': 1.0
        }
        
        analyzer = DiscretizationAnalyzer(reference_config)
        domain_sizes = [10.0, 15.0, 20.0]
        
        results = analyzer.analyze_domain_size_effects(domain_sizes)
        
        assert 'domain_results' in results
        assert 'domain_analysis' in results
        
        # Check that all domain sizes are tested
        assert len(results['domain_results']) == 3
        assert 10.0 in results['domain_results']
        assert 15.0 in results['domain_results']
        assert 20.0 in results['domain_results']
    
    def test_time_step_stability(self):
        """Test time step stability analysis."""
        reference_config = {
            'L': 20.0,
            'N': 256,
            'beta': 1.0,
            'mu': 1.0
        }
        
        analyzer = DiscretizationAnalyzer(reference_config)
        time_steps = [0.001, 0.005, 0.01]
        
        results = analyzer.analyze_time_step_stability(time_steps)
        
        assert 'time_step_results' in results
        assert 'stability_analysis' in results
        
        # Check that all time steps are tested
        assert len(results['time_step_results']) == 3
        assert 0.001 in results['time_step_results']
        assert 0.005 in results['time_step_results']
        assert 0.01 in results['time_step_results']


class TestFailureDetector:
    """Test failure detection functionality."""
    
    def test_initialization(self):
        """Test FailureDetector initialization."""
        config = {
            'beta': 1.0,
            'mu': 1.0,
            'eta': 0.1
        }
        
        detector = FailureDetector(config)
        
        assert detector.config == config
    
    def test_failure_detection(self):
        """Test failure detection."""
        config = {
            'beta': 1.0,
            'mu': 1.0,
            'eta': 0.1
        }
        
        detector = FailureDetector(config)
        failures = detector.detect_failures()
        
        assert 'passivity_violation' in failures
        assert 'singular_mode' in failures
        assert 'energy_conservation' in failures
        assert 'topological_charge' in failures
        assert 'numerical_stability' in failures
        assert 'overall_assessment' in failures
        
        # Check result structure
        for failure_type, result in failures.items():
            if failure_type != 'overall_assessment':
                assert 'detected' in result
    
    def test_failure_boundaries(self):
        """Test failure boundary analysis."""
        config = {
            'beta': 1.0,
            'mu': 1.0,
            'eta': 0.1
        }
        
        detector = FailureDetector(config)
        parameter_ranges = {
            'beta': (0.6, 1.4),
            'mu': (0.5, 1.5)
        }
        
        boundaries = detector.analyze_failure_boundaries(parameter_ranges)
        
        assert 'beta' in boundaries
        assert 'mu' in boundaries
        
        # Check result structure
        for param_name, boundary in boundaries.items():
            assert 'failure_points' in boundary
            assert 'min_failure' in boundary
            assert 'max_failure' in boundary
            assert 'failure_range' in boundary


class TestPhaseMapper:
    """Test phase mapping functionality."""
    
    def test_initialization(self):
        """Test PhaseMapper initialization."""
        config = {
            'eta_range': [0.0, 0.3],
            'chi_double_prime_range': [0.0, 0.8],
            'beta_range': [0.6, 1.4]
        }
        
        mapper = PhaseMapper(config)
        
        assert mapper.config == config
    
    def test_phase_mapping(self):
        """Test phase mapping."""
        config = {
            'eta_range': [0.0, 0.3],
            'chi_double_prime_range': [0.0, 0.8],
            'beta_range': [0.6, 1.4]
        }
        
        mapper = PhaseMapper(config)
        phase_map = mapper.map_phases()
        
        assert 'parameter_grid' in phase_map
        assert 'classifications' in phase_map
        assert 'boundaries' in phase_map
        assert 'statistics' in phase_map
        assert 'phase_diagram' in phase_map
    
    def test_resonance_classification(self):
        """Test resonance classification."""
        config = {
            'eta_range': [0.0, 0.3],
            'chi_double_prime_range': [0.0, 0.8],
            'beta_range': [0.6, 1.4]
        }
        
        mapper = PhaseMapper(config)
        
        # Create test resonance data
        resonance_data = [
            {
                'frequencies': [1.0, 1.1, 1.2],
                'q_factors': [10.0, 11.0, 12.0],
                'widths': [0.1, 0.11, 0.12],
                'shapes': [0.8, 0.81, 0.82],
                'diversity': 0.7,
                'consistency': 0.8
            },
            {
                'frequencies': [2.0, 2.1, 2.2],
                'q_factors': [20.0, 21.0, 22.0],
                'widths': [0.05, 0.051, 0.052],
                'shapes': [0.9, 0.91, 0.92],
                'diversity': 0.9,
                'consistency': 0.9
            }
        ]
        
        classifications = mapper.classify_resonances(resonance_data)
        
        assert 'classifications' in classifications
        assert 'summary' in classifications
        
        # Check that all resonances are classified
        assert len(classifications['classifications']) == 2
        
        # Check summary
        summary = classifications['summary']
        assert 'total_resonances' in summary
        assert 'fundamental_count' in summary
        assert 'emergent_count' in summary
        assert 'unclear_count' in summary


class TestPerformanceAnalyzer:
    """Test performance analysis functionality."""
    
    def test_initialization(self):
        """Test PerformanceAnalyzer initialization."""
        config = {
            'N': 256,
            'L': 20.0,
            'dt': 0.01
        }
        
        analyzer = PerformanceAnalyzer(config)
        
        assert analyzer.config == config
    
    def test_performance_analysis(self):
        """Test performance analysis."""
        config = {
            'N': 256,
            'L': 20.0,
            'dt': 0.01
        }
        
        analyzer = PerformanceAnalyzer(config)
        results = analyzer.analyze_performance()
        
        assert 'scaling_analysis' in results
        assert 'accuracy_cost_analysis' in results
        assert 'benchmark_results' in results
        assert 'memory_analysis' in results
        assert 'optimization_results' in results
    
    def test_scaling_analysis(self):
        """Test scaling analysis."""
        config = {
            'N': 256,
            'L': 20.0,
            'dt': 0.01
        }
        
        analyzer = PerformanceAnalyzer(config)
        scaling_results = analyzer._analyze_scaling_behavior()
        
        assert 'grid_scaling' in scaling_results
        assert 'domain_scaling' in scaling_results
        assert 'time_scaling' in scaling_results
        assert 'overall_scaling' in scaling_results


class TestSolitonModels:
    """Test soliton model functionality."""
    
    def test_soliton_model_initialization(self):
        """Test SolitonModel initialization."""
        # Mock domain object
        class MockDomain:
            def __init__(self):
                self.N = 256
                self.L = 20.0
        
        domain = MockDomain()
        physics_params = {
            'mu': 1.0,
            'beta': 1.0,
            'lambda': 0.0,
            'S4': 0.1,
            'S6': 0.01
        }
        
        soliton = SolitonModel(domain, physics_params)
        
        assert soliton.domain == domain
        assert soliton.params == physics_params
    
    def test_baryon_soliton_initialization(self):
        """Test BaryonSoliton initialization."""
        # Mock domain object
        class MockDomain:
            def __init__(self):
                self.N = 256
                self.L = 20.0
        
        domain = MockDomain()
        physics_params = {
            'mu': 1.0,
            'beta': 1.0,
            'lambda': 0.0,
            'S4': 0.1,
            'S6': 0.01
        }
        
        baryon = BaryonSoliton(domain, physics_params)
        
        assert baryon.baryon_number == 1
        assert isinstance(baryon, SolitonModel)
    
    def test_skyrmion_soliton_initialization(self):
        """Test SkyrmionSoliton initialization."""
        # Mock domain object
        class MockDomain:
            def __init__(self):
                self.N = 256
                self.L = 20.0
        
        domain = MockDomain()
        physics_params = {
            'mu': 1.0,
            'beta': 1.0,
            'lambda': 0.0,
            'S4': 0.1,
            'S6': 0.01
        }
        charge = 2
        
        skyrmion = SkyrmionSoliton(domain, physics_params, charge)
        
        assert skyrmion.charge == 2
        assert isinstance(skyrmion, SolitonModel)


class TestDefectModels:
    """Test defect model functionality."""
    
    def test_defect_model_initialization(self):
        """Test DefectModel initialization."""
        # Mock domain object
        class MockDomain:
            def __init__(self):
                self.N = 256
                self.L = 20.0
        
        domain = MockDomain()
        physics_params = {
            'mu': 1.0,
            'beta': 1.0,
            'lambda': 0.0
        }
        
        defect = DefectModel(domain, physics_params)
        
        assert defect.domain == domain
        assert defect.params == physics_params
    
    def test_vortex_defect_initialization(self):
        """Test VortexDefect initialization."""
        # Mock domain object
        class MockDomain:
            def __init__(self):
                self.N = 256
                self.L = 20.0
        
        domain = MockDomain()
        physics_params = {
            'mu': 1.0,
            'beta': 1.0,
            'lambda': 0.0
        }
        
        vortex = VortexDefect(domain, physics_params)
        
        assert vortex.charge == 1
        assert isinstance(vortex, DefectModel)
    
    def test_multi_defect_system_initialization(self):
        """Test MultiDefectSystem initialization."""
        # Mock domain object
        class MockDomain:
            def __init__(self):
                self.N = 256
                self.L = 20.0
        
        domain = MockDomain()
        physics_params = {
            'mu': 1.0,
            'beta': 1.0,
            'lambda': 0.0
        }
        defect_list = [
            {'position': [5.0, 5.0, 5.0], 'charge': 1},
            {'position': [15.0, 5.0, 5.0], 'charge': -1}
        ]
        
        multi_defect = MultiDefectSystem(domain, physics_params, defect_list)
        
        assert multi_defect.defects == defect_list
        assert isinstance(multi_defect, DefectModel)


class TestLevelEExperiments:
    """Test Level E experiments functionality."""
    
    def test_initialization(self):
        """Test LevelEExperiments initialization."""
        config = {
            'parameter_ranges': {
                'beta': (0.6, 1.4),
                'mu': (0.5, 1.5)
            },
            'base_config': {
                'beta': 1.0,
                'mu': 1.0,
                'eta': 0.1
            },
            'reference_config': {
                'L': 20.0,
                'N': 256,
                'beta': 1.0,
                'mu': 1.0
            }
        }
        
        experiments = LevelEExperiments(config)
        
        assert experiments.config == config
        assert experiments.sensitivity_analyzer is not None
        assert experiments.robustness_tester is not None
        assert experiments.discretization_analyzer is not None
        assert experiments.failure_detector is not None
        assert experiments.phase_mapper is not None
        assert experiments.performance_analyzer is not None
    
    def test_full_analysis(self):
        """Test full analysis execution."""
        config = {
            'parameter_ranges': {
                'beta': (0.6, 1.4),
                'mu': (0.5, 1.5)
            },
            'base_config': {
                'beta': 1.0,
                'mu': 1.0,
                'eta': 0.1
            },
            'reference_config': {
                'L': 20.0,
                'N': 256,
                'beta': 1.0,
                'mu': 1.0
            },
            'E1_sensitivity': {
                'lhs_samples': 100
            },
            'E2_robustness': {
                'noise_levels': [0.0, 0.1]
            },
            'E3_discretization': {
                'grid_sizes': [64, 128, 256]
            },
            'E4_failures': {},
            'E5_phase_mapping': {
                'eta_range': [0.0, 0.3],
                'chi_double_prime_range': [0.0, 0.8],
                'beta_range': [0.6, 1.4]
            },
            'E6_performance': {}
        }
        
        experiments = LevelEExperiments(config)
        results = experiments.run_full_analysis()
        
        assert 'E1_sensitivity' in results
        assert 'E2_robustness' in results
        assert 'E3_discretization' in results
        assert 'E4_failures' in results
        assert 'E5_phase_mapping' in results
        assert 'E6_performance' in results
        assert 'overall_assessment' in results
    
    def test_soliton_experiments(self):
        """Test soliton experiments."""
        config = {
            'parameter_ranges': {
                'beta': (0.6, 1.4),
                'mu': (0.5, 1.5)
            },
            'base_config': {
                'beta': 1.0,
                'mu': 1.0,
                'eta': 0.1
            }
        }
        
        experiments = LevelEExperiments(config)
        results = experiments.run_soliton_experiments()
        
        assert 'baryon_solitons' in results
        assert 'skyrmion_solitons' in results
        assert 'soliton_interactions' in results
    
    def test_defect_experiments(self):
        """Test defect experiments."""
        config = {
            'parameter_ranges': {
                'beta': (0.6, 1.4),
                'mu': (0.5, 1.5)
            },
            'base_config': {
                'beta': 1.0,
                'mu': 1.0,
                'eta': 0.1
            }
        }
        
        experiments = LevelEExperiments(config)
        results = experiments.run_defect_experiments()
        
        assert 'single_defects' in results
        assert 'defect_pairs' in results
        assert 'multi_defect_systems' in results


class TestIntegration:
    """Test integration between components."""
    
    def test_save_and_load_results(self):
        """Test saving and loading results."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filename = f.name
        
        try:
            # Create test results
            results = {
                'test_data': [1, 2, 3],
                'nested': {
                    'value': 42,
                    'array': np.array([1, 2, 3])
                }
            }
            
            # Test saving
            experiments = LevelEExperiments({})
            experiments.save_results(results, temp_filename)
            
            # Test loading
            with open(temp_filename, 'r') as f:
                loaded_results = json.load(f)
            
            assert loaded_results['test_data'] == [1, 2, 3]
            assert loaded_results['nested']['value'] == 42
            assert loaded_results['nested']['array'] == [1, 2, 3]
            
        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
    
    def test_configuration_loading(self):
        """Test loading configuration from file."""
        # Create temporary config file
        config_data = {
            'parameter_ranges': {
                'beta': (0.6, 1.4),
                'mu': (0.5, 1.5)
            },
            'base_config': {
                'beta': 1.0,
                'mu': 1.0,
                'eta': 0.1
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_filename = f.name
        
        try:
            # Load configuration
            with open(temp_filename, 'r') as f:
                loaded_config = json.load(f)
            
            # Create experiments with loaded config
            experiments = LevelEExperiments(loaded_config)
            
            assert experiments.config == loaded_config
            
        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)


if __name__ == '__main__':
    pytest.main([__file__])
