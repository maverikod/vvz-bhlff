#!/usr/bin/env python3
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Minimal QuenchDetector test to avoid memory issues and hanging.
"""

import numpy as np
import time
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def test_quench_detector_minimal():
    """Test QuenchDetector with minimal resources."""
    logger.info("Testing QuenchDetector with minimal resources...")

    try:
        # Import classes
        from bhlff.core.domain.config import SpatialConfig, PhaseConfig, TemporalConfig
        from bhlff.core.domain.domain_7d import Domain7D
        from bhlff.core.bvp.quench_detector import QuenchDetector

        # Create minimal domain
        spatial_config = SpatialConfig(N_x=4, N_y=4, N_z=4)
        phase_config = PhaseConfig(N_phi_1=2, N_phi_2=2, N_phi_3=2)
        temporal_config = TemporalConfig(N_t=3)
        domain_7d = Domain7D(spatial_config, phase_config, temporal_config)

        # Create minimal envelope
        shape = (4, 4, 4, 2, 2, 2, 3)
        envelope = np.random.random(shape).astype(np.float32) * 0.1
        envelope[2, 2, 2, 1, 1, 1, 1] = 1.0  # Single quench event

        logger.info(f"Envelope shape: {shape}")
        logger.info(f"Envelope size: {envelope.nbytes / 1024:.2f} KB")
        logger.info(f"Max envelope value: {np.max(envelope)}")

        # Test CPU
        logger.info("Testing CPU version...")
        config_cpu = {"use_cuda": False, "amplitude_threshold": 0.5}
        detector_cpu = QuenchDetector(domain_7d, config_cpu)

        start_time = time.time()
        results_cpu = detector_cpu.detect_quenches(envelope)
        cpu_time = time.time() - start_time

        logger.info(
            f"CPU results: {results_cpu['total_quenches']} quenches in {cpu_time:.3f}s"
        )

        # Test CUDA with timeout
        logger.info("Testing CUDA version...")
        config_cuda = {"use_cuda": True, "amplitude_threshold": 0.5}
        detector_cuda = QuenchDetector(domain_7d, config_cuda)

        start_time = time.time()
        try:
            results_cuda = detector_cuda.detect_quenches(envelope)
            cuda_time = time.time() - start_time

            logger.info(
                f"CUDA results: {results_cuda['total_quenches']} quenches in {cuda_time:.3f}s"
            )

            # Compare results
            if cpu_time > 0 and cuda_time > 0:
                speedup = cpu_time / cuda_time
                logger.info(f"CUDA speedup: {speedup:.2f}x")

            # Check if results are similar
            if results_cpu["total_quenches"] == results_cuda["total_quenches"]:
                logger.info("✓ CPU and CUDA results match!")
            else:
                logger.warning(
                    f"⚠ Results differ: CPU={results_cpu['total_quenches']}, CUDA={results_cuda['total_quenches']}"
                )

            return True

        except Exception as e:
            logger.error(f"CUDA test failed: {e}")
            return False

    except Exception as e:
        logger.error(f"Error in minimal test: {e}")
        return False


def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("QuenchDetector Minimal Test")
    logger.info("=" * 60)

    success = test_quench_detector_minimal()

    if success:
        logger.info("=" * 60)
        logger.info("Minimal test completed successfully!")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("Minimal test failed!")
        logger.error("=" * 60)


if __name__ == "__main__":
    main()
