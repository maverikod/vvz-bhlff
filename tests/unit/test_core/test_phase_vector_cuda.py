"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

CUDA optimization tests for U(1)³ Phase Vector structure.

This module provides comprehensive tests for CUDA optimization
in the U(1)³ Phase Vector implementation, ensuring GPU utilization.

Physical Meaning:
    Tests validate CUDA optimization:
    - GPU memory utilization
    - CUDA computation performance
    - Correctness of CUDA vs CPU results
    - GPU memory management

Mathematical Foundation:
    Tests CUDA-accelerated computations for:
    - Phase coherence calculations
    - Topological charge computation
    - Electroweak current generation
    - 7D gradient operations

Example:
    >>> pytest tests/unit/test_core/test_phase_vector_cuda.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any
import time

from bhlff.core.domain import Domain
from bhlff.core.bvp.phase_vector.phase_vector import PhaseVector
from bhlff.core.bvp.phase_vector.phase_components import PhaseComponents
from bhlff.core.bvp.phase_vector.electroweak_coupling import ElectroweakCoupling

# CUDA availability check
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


class TestPhaseVectorCUDA:
    """CUDA optimization tests for PhaseVector."""
    
    @pytest.fixture
    def domain_7d_small(self):
        """Create small 7D domain for CUDA testing."""
        return Domain(L=1.0, N=4, dimensions=7)
    
    @pytest.fixture
    def phase_vector_config_cuda(self):
        """Create phase vector configuration with CUDA enabled."""
        return {
            "use_cuda": True,
            "phase_components": {
                "amplitude_1": 1.0,
                "amplitude_2": 0.8,
                "amplitude_3": 0.6,
                "frequency_1": 1.0,
                "frequency_2": 1.5,
                "frequency_3": 2.0,
            },
            "su2_coupling": {
                "coupling_strength": 0.1
            },
            "electroweak": {
                "em_coupling": 1.0,
                "weak_coupling": 0.1,
                "mixing_angle": 0.23,
                "gauge_coupling": 0.65
            }
        }
    
    @pytest.fixture
    def phase_vector_config_cpu(self):
        """Create phase vector configuration with CUDA disabled."""
        return {
            "use_cuda": False,
            "phase_components": {
                "amplitude_1": 1.0,
                "amplitude_2": 0.8,
                "amplitude_3": 0.6,
                "frequency_1": 1.0,
                "frequency_2": 1.5,
                "frequency_3": 2.0,
            },
            "su2_coupling": {
                "coupling_strength": 0.1
            },
            "electroweak": {
                "em_coupling": 1.0,
                "weak_coupling": 0.1,
                "mixing_angle": 0.23,
                "gauge_coupling": 0.65
            }
        }
    
    @pytest.fixture
    def phase_vector_cuda(self, domain_7d_small, phase_vector_config_cuda):
        """Create PhaseVector instance with CUDA enabled."""
        return PhaseVector(domain_7d_small, phase_vector_config_cuda)
    
    @pytest.fixture
    def phase_vector_cpu(self, domain_7d_small, phase_vector_config_cpu):
        """Create PhaseVector instance with CUDA disabled."""
        return PhaseVector(domain_7d_small, phase_vector_config_cpu)
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_cuda_initialization(self, phase_vector_cuda):
        """
        Test CUDA initialization.
        
        Physical Meaning:
            Validates that PhaseVector is properly initialized
            with CUDA support enabled.
        """
        assert phase_vector_cuda.use_cuda == True
        assert phase_vector_cuda.cuda_available == True
        assert "CUDA" in repr(phase_vector_cuda)
    
    def test_cpu_initialization(self, phase_vector_cpu):
        """
        Test CPU initialization.
        
        Physical Meaning:
            Validates that PhaseVector is properly initialized
            with CPU computation.
        """
        assert phase_vector_cpu.use_cuda == False
        assert "CPU" in repr(phase_vector_cpu)
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_cuda_vs_cpu_correctness(self, phase_vector_cuda, phase_vector_cpu):
        """
        Test CUDA vs CPU correctness.
        
        Physical Meaning:
            Validates that CUDA computations produce the same
            results as CPU computations.
        """
        # Create test envelope
        envelope = np.ones(phase_vector_cuda.domain.shape, dtype=complex)
        envelope *= np.exp(1j * np.pi / 4)
        
        # Test phase coherence
        coherence_cuda = phase_vector_cuda.compute_phase_coherence(envelope)
        coherence_cpu = phase_vector_cpu.compute_phase_coherence(envelope)
        
        assert np.isclose(coherence_cuda, coherence_cpu, atol=1e-10)
        
        # Test topological charge
        charge_cuda = phase_vector_cuda.compute_topological_charge(envelope)
        charge_cpu = phase_vector_cpu.compute_topological_charge(envelope)
        
        assert np.isclose(charge_cuda, charge_cpu, atol=1e-10)
        
        # Test electroweak currents
        currents_cuda = phase_vector_cuda.compute_electroweak_currents(envelope)
        currents_cpu = phase_vector_cpu.compute_electroweak_currents(envelope)
        
        for key in currents_cuda:
            assert np.allclose(currents_cuda[key], currents_cpu[key], atol=1e-10)
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_cuda_performance(self, phase_vector_cuda, phase_vector_cpu):
        """
        Test CUDA performance improvement.
        
        Physical Meaning:
            Validates that CUDA computations are faster than
            CPU computations for large arrays.
        """
        # Create larger test envelope for performance testing
        domain_large = Domain(L=2.0, N=8, dimensions=7)
        config_large = phase_vector_cuda.config.copy()
        config_large["use_cuda"] = True
        
        phase_vector_cuda_large = PhaseVector(domain_large, config_large)
        config_large["use_cuda"] = False
        phase_vector_cpu_large = PhaseVector(domain_large, config_large)
        
        envelope = np.ones(domain_large.shape, dtype=complex)
        envelope *= np.exp(1j * np.pi / 4)
        
        # Measure CUDA performance
        start_time = time.time()
        for _ in range(10):
            coherence_cuda = phase_vector_cuda_large.compute_phase_coherence(envelope)
            charge_cuda = phase_vector_cuda_large.compute_topological_charge(envelope)
            currents_cuda = phase_vector_cuda_large.compute_electroweak_currents(envelope)
        cuda_time = time.time() - start_time
        
        # Measure CPU performance
        start_time = time.time()
        for _ in range(10):
            coherence_cpu = phase_vector_cpu_large.compute_phase_coherence(envelope)
            charge_cpu = phase_vector_cpu_large.compute_topological_charge(envelope)
            currents_cpu = phase_vector_cpu_large.compute_electroweak_currents(envelope)
        cpu_time = time.time() - start_time
        
        # CUDA should be faster (or at least not significantly slower)
        # Allow for some overhead in small arrays
        assert cuda_time <= cpu_time * 2.0, f"CUDA time {cuda_time:.3f}s > CPU time {cpu_time:.3f}s * 2"
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_gpu_memory_management(self, phase_vector_cuda):
        """
        Test GPU memory management.
        
        Physical Meaning:
            Validates that GPU memory is properly managed
            and arrays are correctly transferred.
        """
        # Test GPU array creation
        test_array = np.ones((4, 4, 4, 4, 4, 4, 4), dtype=complex)
        gpu_array = phase_vector_cuda._to_gpu(test_array)
        
        # Check that it's a GPU array
        assert hasattr(gpu_array, 'get')
        
        # Test CPU array conversion
        cpu_array = phase_vector_cuda._to_cpu(gpu_array)
        assert isinstance(cpu_array, np.ndarray)
        assert np.allclose(cpu_array, test_array)
        
        # Test CUDA operations
        result = phase_vector_cuda._cuda_abs(gpu_array)
        assert hasattr(result, 'get')
        
        # Test conversion back to CPU
        cpu_result = phase_vector_cuda._to_cpu(result)
        assert isinstance(cpu_result, np.ndarray)
        assert np.allclose(cpu_result, np.abs(test_array))
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_cuda_gradient_operations(self, phase_vector_cuda):
        """
        Test CUDA gradient operations.
        
        Physical Meaning:
            Validates that CUDA gradient computations work
            correctly for 7D arrays.
        """
        # Create test array
        test_array = np.ones((4, 4, 4, 4, 4, 4, 4), dtype=complex)
        test_array *= np.exp(1j * np.pi / 4)
        
        # Test gradient computation
        gradient = phase_vector_cuda._cuda_gradient(test_array, axis=0)
        
        # Check that gradient is computed
        assert gradient.shape == test_array.shape
        assert hasattr(gradient, 'get')  # Should be GPU array
        
        # Convert to CPU and check
        gradient_cpu = phase_vector_cuda._to_cpu(gradient)
        assert isinstance(gradient_cpu, np.ndarray)
        assert np.all(np.isfinite(gradient_cpu))
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_cuda_mathematical_operations(self, phase_vector_cuda):
        """
        Test CUDA mathematical operations.
        
        Physical Meaning:
            Validates that CUDA mathematical operations work
            correctly for complex arrays.
        """
        # Create test array
        test_array = np.ones((4, 4, 4, 4, 4, 4, 4), dtype=complex)
        test_array *= np.exp(1j * np.pi / 4)
        
        # Test various CUDA operations
        gpu_array = phase_vector_cuda._to_gpu(test_array)
        
        # Test abs
        abs_result = phase_vector_cuda._cuda_abs(gpu_array)
        abs_cpu = phase_vector_cuda._to_cpu(abs_result)
        assert np.allclose(abs_cpu, np.abs(test_array))
        
        # Test angle
        angle_result = phase_vector_cuda._cuda_angle(gpu_array)
        angle_cpu = phase_vector_cuda._to_cpu(angle_result)
        assert np.allclose(angle_cpu, np.angle(test_array))
        
        # Test exp
        exp_result = phase_vector_cuda._cuda_exp(gpu_array)
        exp_cpu = phase_vector_cuda._to_cpu(exp_result)
        assert np.allclose(exp_cpu, np.exp(test_array))
        
        # Test sqrt
        sqrt_result = phase_vector_cuda._cuda_sqrt(gpu_array)
        sqrt_cpu = phase_vector_cuda._to_cpu(sqrt_result)
        assert np.allclose(sqrt_cpu, np.sqrt(test_array))
    
    def test_cuda_fallback_to_cpu(self):
        """
        Test CUDA fallback to CPU when CUDA is not available.
        
        Physical Meaning:
            Validates that the system gracefully falls back
            to CPU computation when CUDA is not available.
        """
        # This test should work even without CUDA
        domain = Domain(L=1.0, N=2, dimensions=7)
        config = {
            "use_cuda": False,  # Force CPU mode
            "phase_components": {
                "amplitude_1": 1.0,
                "amplitude_2": 0.8,
                "amplitude_3": 0.6,
                "frequency_1": 1.0,
                "frequency_2": 1.5,
                "frequency_3": 2.0,
            },
            "su2_coupling": {
                "coupling_strength": 0.1
            },
            "electroweak": {
                "em_coupling": 1.0,
                "weak_coupling": 0.1,
                "mixing_angle": 0.23,
                "gauge_coupling": 0.65
            }
        }
        
        phase_vector = PhaseVector(domain, config)
        
        # Should work in CPU mode
        assert phase_vector.use_cuda == False
        assert "CPU" in repr(phase_vector)
        
        # Test basic functionality
        envelope = np.ones(domain.shape, dtype=complex)
        coherence = phase_vector.compute_phase_coherence(envelope)
        assert isinstance(coherence, float)
        assert 0 <= coherence <= 1


class TestPhaseComponentsCUDA:
    """CUDA optimization tests for PhaseComponents."""
    
    @pytest.fixture
    def domain_7d_small(self):
        """Create small 7D domain for CUDA testing."""
        return Domain(L=1.0, N=4, dimensions=7)
    
    @pytest.fixture
    def phase_components_config_cuda(self):
        """Create phase components configuration with CUDA enabled."""
        return {
            "use_cuda": True,
            "phase_components": {
                "amplitude_1": 1.0,
                "amplitude_2": 0.8,
                "amplitude_3": 0.6,
                "frequency_1": 1.0,
                "frequency_2": 1.5,
                "frequency_3": 2.0,
            }
        }
    
    @pytest.fixture
    def phase_components_cuda(self, domain_7d_small, phase_components_config_cuda):
        """Create PhaseComponents instance with CUDA enabled."""
        return PhaseComponents(domain_7d_small, phase_components_config_cuda)
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_phase_components_cuda_initialization(self, phase_components_cuda):
        """
        Test PhaseComponents CUDA initialization.
        
        Physical Meaning:
            Validates that PhaseComponents is properly initialized
            with CUDA support enabled.
        """
        assert phase_components_cuda.use_cuda == True
        assert phase_components_cuda.cuda_available == True
        assert "CUDA" in repr(phase_components_cuda)
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_phase_components_cuda_coherence(self, phase_components_cuda):
        """
        Test PhaseComponents CUDA coherence computation.
        
        Physical Meaning:
            Validates that phase coherence is correctly computed
            using CUDA operations.
        """
        coherence = phase_components_cuda.compute_phase_coherence()
        
        assert isinstance(coherence, np.ndarray)
        assert coherence.shape == phase_components_cuda.domain.shape
        assert np.all(coherence >= 0)
        assert np.all(coherence <= 1)


class TestElectroweakCouplingCUDA:
    """CUDA optimization tests for ElectroweakCoupling."""
    
    @pytest.fixture
    def electroweak_config_cuda(self):
        """Create electroweak configuration with CUDA enabled."""
        return {
            "use_cuda": True,
            "electroweak": {
                "em_coupling": 1.0,
                "weak_coupling": 0.1,
                "mixing_angle": 0.23,
                "gauge_coupling": 0.65
            }
        }
    
    @pytest.fixture
    def electroweak_coupling_cuda(self, electroweak_config_cuda):
        """Create ElectroweakCoupling instance with CUDA enabled."""
        return ElectroweakCoupling(electroweak_config_cuda)
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_electroweak_coupling_cuda_initialization(self, electroweak_coupling_cuda):
        """
        Test ElectroweakCoupling CUDA initialization.
        
        Physical Meaning:
            Validates that ElectroweakCoupling is properly initialized
            with CUDA support enabled.
        """
        assert electroweak_coupling_cuda.use_cuda == True
        assert electroweak_coupling_cuda.cuda_available == True
        assert "CUDA" in repr(electroweak_coupling_cuda)
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_electroweak_coupling_cuda_currents(self, electroweak_coupling_cuda):
        """
        Test ElectroweakCoupling CUDA current computation.
        
        Physical Meaning:
            Validates that electroweak currents are correctly computed
            using CUDA operations.
        """
        domain = Domain(L=1.0, N=4, dimensions=7)
        envelope = np.ones(domain.shape, dtype=complex)
        envelope *= np.exp(1j * np.pi / 4)
        
        # Create phase components
        phase_components = []
        for a in range(3):
            component = np.ones(domain.shape, dtype=complex)
            component *= np.exp(1j * 2 * np.pi * a / 3)
            phase_components.append(component)
        
        currents = electroweak_coupling_cuda.compute_electroweak_currents(
            envelope, phase_components, domain
        )
        
        # Check current structure
        assert isinstance(currents, dict)
        assert "em_current" in currents
        assert "weak_current" in currents
        assert "mixed_current" in currents
        
        # Check current properties
        for current_name, current in currents.items():
            assert isinstance(current, np.ndarray)
            assert current.shape == domain.shape
            assert np.all(np.isfinite(current))
