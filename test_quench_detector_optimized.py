#!/usr/bin/env python3
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Optimized QuenchDetector test script with GPU acceleration and detailed logging.

This script tests the QuenchDetector with optimized GPU usage and parallel processing
for large 7D arrays, using 80% of GPU memory efficiently.
"""

import numpy as np
import time
import logging
import sys
from typing import Dict, Any, Tuple
import gc

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('quench_detector_test.log')
    ]
)
logger = logging.getLogger(__name__)

try:
    import cupy as cp
    CUDA_AVAILABLE = True
    logger.info("CUDA available - using GPU acceleration")
except ImportError:
    CUDA_AVAILABLE = False
    cp = None
    logger.warning("CUDA not available - using CPU processing")

def get_gpu_memory_info():
    """Get GPU memory information."""
    if not CUDA_AVAILABLE:
        return {"total": 0, "free": 0, "used": 0}
    
    try:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        
        # Get memory info
        total_memory = cp.cuda.runtime.memGetInfo()[1]
        free_memory = cp.cuda.runtime.memGetInfo()[0]
        used_memory = total_memory - free_memory
        
        return {
            "total": total_memory / 1024**3,  # GB
            "free": free_memory / 1024**3,    # GB
            "used": used_memory / 1024**3,    # GB
            "mempool_used": mempool.used_bytes() / 1024**3,  # GB
            "pinned_used": pinned_mempool.n_free_blocks()
        }
    except Exception as e:
        logger.error(f"Error getting GPU memory info: {e}")
        return {"total": 0, "free": 0, "used": 0}

def optimize_gpu_memory():
    """Optimize GPU memory usage."""
    if not CUDA_AVAILABLE:
        return
    
    try:
        # Clear memory pools
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        
        # Set memory pool size to use 80% of GPU memory
        mem_info = get_gpu_memory_info()
        if mem_info["total"] > 0:
            target_memory = int(mem_info["total"] * 0.8 * 1024**3)  # 80% in bytes
            cp.get_default_memory_pool().set_limit(size=target_memory)
            logger.info(f"Set GPU memory limit to {target_memory / 1024**3:.2f} GB (80% of total)")
        
    except Exception as e:
        logger.error(f"Error optimizing GPU memory: {e}")

def create_optimized_domain():
    """Create optimized domain for testing."""
    logger.info("Creating optimized 7D domain...")
    
    try:
        logger.info("Importing domain configuration classes...")
        from bhlff.core.domain.config import SpatialConfig, PhaseConfig, TemporalConfig
        from bhlff.core.domain.domain_7d import Domain7D
        logger.info("Domain classes imported successfully")
        
        # Optimized domain size for GPU processing
        logger.info("Creating spatial configuration...")
        spatial_config = SpatialConfig(
            L_x=1.0, L_y=1.0, L_z=1.0,  # Smaller spatial domain
            N_x=32, N_y=32, N_z=32      # Reduced grid size
        )
        logger.info(f"Spatial config: {spatial_config}")
        
        logger.info("Creating phase configuration...")
        phase_config = PhaseConfig(
            N_phi_1=16, N_phi_2=16, N_phi_3=16  # Reduced phase grid
        )
        logger.info(f"Phase config: {phase_config}")
        
        logger.info("Creating temporal configuration...")
        temporal_config = TemporalConfig(
            T_max=0.5, N_t=20, dt=0.025  # Shorter time evolution
        )
        logger.info(f"Temporal config: {temporal_config}")
        
        logger.info("Creating Domain7D instance...")
        domain_7d = Domain7D(spatial_config, phase_config, temporal_config)
        logger.info(f"Domain created successfully: {domain_7d}")
        
        return domain_7d
        
    except Exception as e:
        logger.error(f"Error creating domain: {e}")
        raise

def create_test_envelope(domain_7d, use_gpu=False):
    """Create optimized test envelope with quench events."""
    logger.info("Creating test envelope...")
    
    try:
        # Get domain shape
        logger.info("Calculating domain shape...")
        shape = (
            domain_7d.spatial_config.N_x,
            domain_7d.spatial_config.N_y,
            domain_7d.spatial_config.N_z,
            domain_7d.phase_config.N_phi_1,
            domain_7d.phase_config.N_phi_2,
            domain_7d.phase_config.N_phi_3,
            domain_7d.temporal_config.N_t
        )
        
        logger.info(f"Envelope shape: {shape}")
        total_elements = np.prod(shape)
        logger.info(f"Total elements: {total_elements:,}")
        logger.info(f"Memory estimate: {total_elements * 4 / 1024**3:.2f} GB (float32)")
        
        # Create envelope array
        logger.info("Creating envelope array...")
        start_time = time.time()
        
        if use_gpu and CUDA_AVAILABLE:
            logger.info("Creating envelope on GPU...")
            envelope = cp.random.random(shape, dtype=cp.float32) * 0.3
            logger.info("GPU envelope created successfully")
        else:
            logger.info("Creating envelope on CPU...")
            envelope = np.random.random(shape).astype(np.float32) * 0.3
            logger.info("CPU envelope created successfully")
        
        creation_time = time.time() - start_time
        logger.info(f"Envelope creation time: {creation_time:.3f}s")
        
        # Add quench events
        logger.info("Adding quench events...")
        start_time = time.time()
        
        # High amplitude quench
        if use_gpu and CUDA_AVAILABLE:
            logger.info("Adding GPU quench events...")
            envelope[16, 16, 16, 8, 8, 8, 10] = 1.0
            envelope[8, 8, 8, 4, 4, 4, 5] = 0.9
            envelope[24, 24, 24, 12, 12, 12, 15] = 0.8
            logger.info("GPU quench events added")
        else:
            logger.info("Adding CPU quench events...")
            envelope[16, 16, 16, 8, 8, 8, 10] = 1.0
            envelope[8, 8, 8, 4, 4, 4, 5] = 0.9
            envelope[24, 24, 24, 12, 12, 12, 15] = 0.8
            logger.info("CPU quench events added")
        
        quench_time = time.time() - start_time
        logger.info(f"Quench events addition time: {quench_time:.3f}s")
        
        # Log envelope statistics
        if use_gpu and CUDA_AVAILABLE:
            logger.info(f"GPU envelope stats: min={cp.min(envelope):.3f}, max={cp.max(envelope):.3f}, mean={cp.mean(envelope):.3f}")
        else:
            logger.info(f"CPU envelope stats: min={np.min(envelope):.3f}, max={np.max(envelope):.3f}, mean={np.mean(envelope):.3f}")
        
        logger.info("Test envelope created successfully")
        return envelope
        
    except Exception as e:
        logger.error(f"Error creating test envelope: {e}")
        raise

def test_quench_detector_cpu(domain_7d, envelope):
    """Test QuenchDetector on CPU."""
    logger.info("Testing QuenchDetector on CPU...")
    
    try:
        logger.info("Importing QuenchDetector...")
        from bhlff.core.bvp.quench_detector import QuenchDetector
        logger.info("QuenchDetector imported successfully")
        
        # CPU configuration
        logger.info("Setting up CPU configuration...")
        config = {
            'amplitude_threshold': 0.5,
            'detuning_threshold': 0.1,
            'gradient_threshold': 0.3,
            'carrier_frequency': 1.0,
            'use_cuda': False,
            'min_quench_size': 2
        }
        logger.info(f"CPU config: {config}")
        
        # Create detector
        logger.info("Creating QuenchDetector instance...")
        start_time = time.time()
        detector = QuenchDetector(domain_7d, config)
        init_time = time.time() - start_time
        logger.info(f"Detector initialization time: {init_time:.3f}s")
        logger.info(f"Detector thresholds: amp={detector.amplitude_threshold}, det={detector.detuning_threshold}, grad={detector.gradient_threshold}")
        
        # Detect quenches
        logger.info("Starting quench detection...")
        start_time = time.time()
        results = detector.detect_quenches(envelope)
        detection_time = time.time() - start_time
        logger.info(f"Quench detection completed in: {detection_time:.3f}s")
        
        # Log results
        logger.info("CPU Detection Results:")
        logger.info(f"  Quenches detected: {results['quenches_detected']}")
        logger.info(f"  Total quenches: {results['total_quenches']}")
        logger.info(f"  Amplitude quenches: {len(results['amplitude_quenches'])}")
        logger.info(f"  Detuning quenches: {len(results['detuning_quenches'])}")
        logger.info(f"  Gradient quenches: {len(results['gradient_quenches'])}")
        
        if results['quench_locations']:
            logger.info(f"  First quench location: {results['quench_locations'][0]}")
            logger.info(f"  First quench type: {results['quench_types'][0]}")
            logger.info(f"  First quench strength: {results['quench_strengths'][0]:.3f}")
        else:
            logger.info("  No quenches detected")
        
        return results, detection_time
        
    except Exception as e:
        logger.error(f"Error in CPU quench detection: {e}")
        raise

def test_quench_detector_gpu(domain_7d, envelope_gpu):
    """Test QuenchDetector on GPU."""
    if not CUDA_AVAILABLE:
        logger.warning("CUDA not available - skipping GPU test")
        return None, 0
    
    logger.info("Testing QuenchDetector on GPU...")
    
    from bhlff.core.bvp.quench_detector import QuenchDetector
    
    # GPU configuration
    config = {
        'amplitude_threshold': 0.5,
        'detuning_threshold': 0.1,
        'gradient_threshold': 0.3,
        'carrier_frequency': 1.0,
        'use_cuda': True,
        'min_quench_size': 2
    }
    
    # Create detector
    start_time = time.time()
    detector = QuenchDetector(domain_7d, config)
    init_time = time.time() - start_time
    logger.info(f"GPU Detector initialization time: {init_time:.3f}s")
    
    # Detect quenches
    start_time = time.time()
    results = detector.detect_quenches(envelope_gpu)
    detection_time = time.time() - start_time
    logger.info(f"GPU Quench detection time: {detection_time:.3f}s")
    
    # Log results
    logger.info("GPU Detection Results:")
    logger.info(f"  Quenches detected: {results['quenches_detected']}")
    logger.info(f"  Total quenches: {results['total_quenches']}")
    logger.info(f"  Amplitude quenches: {len(results['amplitude_quenches'])}")
    logger.info(f"  Detuning quenches: {len(results['detuning_quenches'])}")
    logger.info(f"  Gradient quenches: {len(results['gradient_quenches'])}")
    
    if results['quench_locations']:
        logger.info(f"  First quench location: {results['quench_locations'][0]}")
        logger.info(f"  First quench type: {results['quench_types'][0]}")
        logger.info(f"  First quench strength: {results['quench_strengths'][0]:.3f}")
    
    return results, detection_time

def benchmark_detection(domain_7d, num_runs=3):
    """Benchmark quench detection performance."""
    logger.info(f"Benchmarking quench detection with {num_runs} runs...")
    
    cpu_times = []
    gpu_times = []
    
    for run in range(num_runs):
        logger.info(f"Run {run + 1}/{num_runs}")
        
        # Create test envelope
        envelope_cpu = create_test_envelope(domain_7d, use_gpu=False)
        
        # CPU test
        logger.info("CPU test...")
        _, cpu_time = test_quench_detector_cpu(domain_7d, envelope_cpu)
        cpu_times.append(cpu_time)
        
        # GPU test
        if CUDA_AVAILABLE:
            logger.info("GPU test...")
            envelope_gpu = create_test_envelope(domain_7d, use_gpu=True)
            _, gpu_time = test_quench_detector_gpu(domain_7d, envelope_gpu)
            gpu_times.append(gpu_time)
            
            # Cleanup GPU memory
            del envelope_gpu
            cp.get_default_memory_pool().free_all_blocks()
        
        # Cleanup CPU memory
        del envelope_cpu
        gc.collect()
    
    # Calculate statistics
    cpu_mean = np.mean(cpu_times)
    cpu_std = np.std(cpu_times)
    
    logger.info(f"CPU Performance:")
    logger.info(f"  Mean time: {cpu_mean:.3f}s ± {cpu_std:.3f}s")
    logger.info(f"  Times: {cpu_times}")
    
    if gpu_times:
        gpu_mean = np.mean(gpu_times)
        gpu_std = np.std(gpu_times)
        speedup = cpu_mean / gpu_mean
        
        logger.info(f"GPU Performance:")
        logger.info(f"  Mean time: {gpu_mean:.3f}s ± {gpu_std:.3f}s")
        logger.info(f"  Times: {gpu_times}")
        logger.info(f"  Speedup: {speedup:.2f}x")
    
    return cpu_times, gpu_times

def main():
    """Main test function."""
    logger.info("=" * 80)
    logger.info("Starting QuenchDetector optimization test...")
    logger.info("=" * 80)
    
    try:
        # Check GPU memory
        logger.info("Step 1: Checking GPU memory...")
        if CUDA_AVAILABLE:
            mem_info = get_gpu_memory_info()
            logger.info(f"GPU Memory: {mem_info['used']:.2f}GB used / {mem_info['total']:.2f}GB total")
            optimize_gpu_memory()
        else:
            logger.info("CUDA not available - using CPU only")
        
        # Create optimized domain
        logger.info("Step 2: Creating optimized domain...")
        domain_7d = create_optimized_domain()
        
        # Test single detection
        logger.info("Step 3: Testing single quench detection...")
        logger.info("Creating CPU envelope...")
        envelope_cpu = create_test_envelope(domain_7d, use_gpu=False)
        logger.info("Testing CPU quench detection...")
        results_cpu, time_cpu = test_quench_detector_cpu(domain_7d, envelope_cpu)
        
        if CUDA_AVAILABLE:
            logger.info("Creating GPU envelope...")
            envelope_gpu = create_test_envelope(domain_7d, use_gpu=True)
            logger.info("Testing GPU quench detection...")
            results_gpu, time_gpu = test_quench_detector_gpu(domain_7d, envelope_gpu)
            
            # Compare results
            logger.info("Step 4: Comparing CPU vs GPU results...")
            logger.info(f"CPU detected {results_cpu['total_quenches']} quenches in {time_cpu:.3f}s")
            logger.info(f"GPU detected {results_gpu['total_quenches']} quenches in {time_gpu:.3f}s")
            
            if time_gpu > 0:
                speedup = time_cpu / time_gpu
                logger.info(f"GPU speedup: {speedup:.2f}x")
        else:
            logger.info("Skipping GPU test - CUDA not available")
        
        # Benchmark performance
        logger.info("Step 5: Running performance benchmark...")
        cpu_times, gpu_times = benchmark_detection(domain_7d, num_runs=3)
        
        # Final GPU memory check
        logger.info("Step 6: Final memory check...")
        if CUDA_AVAILABLE:
            mem_info = get_gpu_memory_info()
            logger.info(f"Final GPU Memory: {mem_info['used']:.2f}GB used / {mem_info['total']:.2f}GB total")
        
        logger.info("=" * 80)
        logger.info("QuenchDetector optimization test completed successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error in main test: {e}")
        logger.error("Test failed!")
        raise

if __name__ == "__main__":
    main()
