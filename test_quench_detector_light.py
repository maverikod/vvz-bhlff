#!/usr/bin/env python3
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Lightweight QuenchDetector test script with minimal resource usage.

This script tests the QuenchDetector with very small arrays to avoid
system overload and hard resets.
"""

import numpy as np
import time
import logging
import sys
from typing import Dict, Any, Tuple
import gc

# Setup minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def test_quench_detector_minimal():
    """Test QuenchDetector with minimal resources."""
    logger.info("Starting minimal QuenchDetector test...")
    
    try:
        # Step 1: Import classes
        logger.info("Step 1: Importing classes...")
        from bhlff.core.domain.config import SpatialConfig, PhaseConfig, TemporalConfig
        from bhlff.core.domain.domain_7d import Domain7D
        from bhlff.core.bvp.quench_detector import QuenchDetector
        logger.info("Classes imported successfully")
        
        # Step 2: Create minimal domain
        logger.info("Step 2: Creating minimal domain...")
        spatial_config = SpatialConfig(
            L_x=1.0, L_y=1.0, L_z=1.0,
            N_x=8, N_y=8, N_z=8  # Very small grid
        )
        phase_config = PhaseConfig(
            N_phi_1=4, N_phi_2=4, N_phi_3=4  # Very small phase grid
        )
        temporal_config = TemporalConfig(
            T_max=0.1, N_t=5, dt=0.02  # Very short time
        )
        
        domain_7d = Domain7D(spatial_config, phase_config, temporal_config)
        logger.info(f"Domain created: {domain_7d}")
        
        # Step 3: Create minimal envelope
        logger.info("Step 3: Creating minimal envelope...")
        shape = (8, 8, 8, 4, 4, 4, 5)
        total_elements = np.prod(shape)
        logger.info(f"Envelope shape: {shape}")
        logger.info(f"Total elements: {total_elements:,}")
        logger.info(f"Memory estimate: {total_elements * 4 / 1024**2:.2f} MB")
        
        # Create small envelope
        envelope = np.random.random(shape).astype(np.float32) * 0.3
        envelope[4, 4, 4, 2, 2, 2, 2] = 1.0  # Single quench
        logger.info("Envelope created successfully")
        
        # Step 4: Create detector
        logger.info("Step 4: Creating QuenchDetector...")
        config = {
            'amplitude_threshold': 0.5,
            'detuning_threshold': 0.1,
            'gradient_threshold': 0.3,
            'carrier_frequency': 1.0,
            'use_cuda': False,
            'min_quench_size': 1
        }
        
        detector = QuenchDetector(domain_7d, config)
        logger.info("Detector created successfully")
        
        # Step 5: Test detection
        logger.info("Step 5: Testing quench detection...")
        start_time = time.time()
        results = detector.detect_quenches(envelope)
        detection_time = time.time() - start_time
        
        logger.info(f"Detection completed in: {detection_time:.3f}s")
        logger.info(f"Quenches detected: {results['quenches_detected']}")
        logger.info(f"Total quenches: {results['total_quenches']}")
        logger.info(f"Amplitude quenches: {len(results['amplitude_quenches'])}")
        logger.info(f"Detuning quenches: {len(results['detuning_quenches'])}")
        logger.info(f"Gradient quenches: {len(results['gradient_quenches'])}")
        
        if results['quench_locations']:
            logger.info(f"First quench location: {results['quench_locations'][0]}")
            logger.info(f"First quench type: {results['quench_types'][0]}")
            logger.info(f"First quench strength: {results['quench_strengths'][0]:.3f}")
        
        logger.info("Minimal test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error in minimal test: {e}")
        return False

def test_quench_detector_step_by_step():
    """Test QuenchDetector step by step with progress tracking."""
    logger.info("Starting step-by-step QuenchDetector test...")
    
    steps = [
        ("Importing domain classes", lambda: __import__('bhlff.core.domain.config')),
        ("Importing domain 7D", lambda: __import__('bhlff.core.domain.domain_7d')),
        ("Importing quench detector", lambda: __import__('bhlff.core.bvp.quench_detector')),
        ("Creating spatial config", lambda: SpatialConfig(N_x=4, N_y=4, N_z=4)),
        ("Creating phase config", lambda: PhaseConfig(N_phi_1=2, N_phi_2=2, N_phi_3=2)),
        ("Creating temporal config", lambda: TemporalConfig(N_t=3)),
        ("Creating domain", lambda: Domain7D(spatial_config, phase_config, temporal_config)),
        ("Creating envelope", lambda: np.random.random((4, 4, 4, 2, 2, 2, 3)).astype(np.float32)),
        ("Creating detector", lambda: QuenchDetector(domain_7d, {'use_cuda': False})),
        ("Detecting quenches", lambda: detector.detect_quenches(envelope))
    ]
    
    results = {}
    
    for i, (step_name, step_func) in enumerate(steps):
        logger.info(f"Step {i+1}/{len(steps)}: {step_name}...")
        try:
            start_time = time.time()
            result = step_func()
            step_time = time.time() - start_time
            logger.info(f"  ✓ Completed in {step_time:.3f}s")
            results[step_name] = result
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            return False
    
    logger.info("Step-by-step test completed successfully!")
    return True

def main():
    """Main test function with resource monitoring."""
    logger.info("=" * 60)
    logger.info("QuenchDetector Lightweight Test")
    logger.info("=" * 60)
    
    # Test 1: Minimal test
    logger.info("Test 1: Minimal QuenchDetector test")
    success1 = test_quench_detector_minimal()
    
    if not success1:
        logger.error("Minimal test failed!")
        return
    
    # Test 2: Step-by-step test
    logger.info("\nTest 2: Step-by-step QuenchDetector test")
    success2 = test_quench_detector_step_by_step()
    
    if not success2:
        logger.error("Step-by-step test failed!")
        return
    
    logger.info("=" * 60)
    logger.info("All tests completed successfully!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
