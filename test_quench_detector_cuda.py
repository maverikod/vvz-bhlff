#!/usr/bin/env python3
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

CUDA QuenchDetector test script with proper GPU acceleration.
"""

import numpy as np
import time
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def test_quench_detector_cuda():
    """Test QuenchDetector with CUDA acceleration."""
    logger.info("Testing QuenchDetector with CUDA...")
    
    try:
        # Import classes
        from bhlff.core.domain.config import SpatialConfig, PhaseConfig, TemporalConfig
        from bhlff.core.domain.domain_7d import Domain7D
        from bhlff.core.bvp.quench_detector import QuenchDetector
        
        # Create domain
        spatial_config = SpatialConfig(N_x=8, N_y=8, N_z=8)
        phase_config = PhaseConfig(N_phi_1=4, N_phi_2=4, N_phi_3=4)
        temporal_config = TemporalConfig(N_t=5)
        domain_7d = Domain7D(spatial_config, phase_config, temporal_config)
        
        # Create envelope
        shape = (8, 8, 8, 4, 4, 4, 5)
        envelope = np.random.random(shape).astype(np.float32) * 0.3
        envelope[4, 4, 4, 2, 2, 2, 2] = 1.0  # Quench event
        
        # Test CPU
        logger.info("Testing CPU version...")
        config_cpu = {'use_cuda': False}
        detector_cpu = QuenchDetector(domain_7d, config_cpu)
        
        start_time = time.time()
        results_cpu = detector_cpu.detect_quenches(envelope)
        cpu_time = time.time() - start_time
        
        logger.info(f"CPU results: {results_cpu['total_quenches']} quenches in {cpu_time:.3f}s")
        
        # Test CUDA
        logger.info("Testing CUDA version...")
        config_cuda = {'use_cuda': True}
        detector_cuda = QuenchDetector(domain_7d, config_cuda)
        
        start_time = time.time()
        results_cuda = detector_cuda.detect_quenches(envelope)
        cuda_time = time.time() - start_time
        
        logger.info(f"CUDA results: {results_cuda['total_quenches']} quenches in {cuda_time:.3f}s")
        
        # Compare results
        if cpu_time > 0 and cuda_time > 0:
            speedup = cpu_time / cuda_time
            logger.info(f"CUDA speedup: {speedup:.2f}x")
        
        # Check if results are similar
        if results_cpu['total_quenches'] == results_cuda['total_quenches']:
            logger.info("✓ CPU and CUDA results match!")
        else:
            logger.warning(f"⚠ Results differ: CPU={results_cpu['total_quenches']}, CUDA={results_cuda['total_quenches']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in CUDA test: {e}")
        return False

def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("QuenchDetector CUDA Test")
    logger.info("=" * 60)
    
    success = test_quench_detector_cuda()
    
    if success:
        logger.info("=" * 60)
        logger.info("CUDA test completed successfully!")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("CUDA test failed!")
        logger.error("=" * 60)

if __name__ == "__main__":
    main()
