#!/usr/bin/env python3
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Demonstration of enhanced block processing for 7D BVP computations.

This script demonstrates the capabilities of the enhanced block processing system
for 7D BVP computations with intelligent memory management and adaptive block sizing.

Physical Meaning:
    Demonstrates intelligent block-based processing for 7D phase field computations
    with adaptive memory management and processing optimization.

Example:
    python scripts/demo_enhanced_block_processing.py
"""

import sys
import os
import numpy as np
import time
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bhlff.core.domain.enhanced_block_processor import (
    EnhancedBlockProcessor,
    ProcessingConfig,
    ProcessingMode,
)
from bhlff.core.domain import Domain
from bhlff.core.bvp.bvp_block_processing_system import (
    BVPBlockProcessingSystem,
    BVPBlockConfig,
)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def demonstrate_basic_block_processing():
    """Demonstrate basic block processing capabilities."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Basic Block Processing")
    print("=" * 60)

    # Create test domain
    domain = Domain(L=1.0, N=32, dimensions=3)
    print(f"Domain: {domain.shape}, Total elements: {np.prod(domain.shape):,}")

    # Create processing configuration
    config = ProcessingConfig(
        mode=ProcessingMode.ADAPTIVE,
        max_memory_usage=0.8,
        min_block_size=4,
        max_block_size=16,
        overlap_ratio=0.1,
        enable_memory_optimization=True,
        enable_adaptive_sizing=True,
        enable_parallel_processing=True,
    )

    # Create processor
    processor = EnhancedBlockProcessor(domain, config)
    print(
        f"Processor initialized with block size: {processor.base_processor.block_size}"
    )

    # Create test field
    field = np.random.random(domain.shape) + 1j * np.random.random(domain.shape)
    print(f"Test field size: {field.nbytes / (1024**2):.2f} MB")

    # Test different operations
    operations = ["fft", "ifft", "bvp_solve"]

    for operation in operations:
        print(f"\nTesting operation: {operation}")
        start_time = time.time()

        try:
            result = processor.process_7d_field(field, operation=operation)
            processing_time = time.time() - start_time

            print(f"  ✓ Success: {processing_time:.3f}s")
            print(f"  Result shape: {result.shape}")
            print(f"  Result dtype: {result.dtype}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    # Show processing statistics
    stats = processor.get_processing_stats()
    print(f"\nProcessing Statistics:")
    print(f"  CUDA Available: {stats['cuda_available']}")
    print(f"  Current Block Size: {stats['current_block_size']}")
    print(f"  Memory Usage: {stats['memory_usage']}")

    # Cleanup
    processor.cleanup()
    print("  ✓ Cleanup completed")


def demonstrate_bvp_block_processing():
    """Demonstrate BVP-specific block processing."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: BVP Block Processing")
    print("=" * 60)

    # Create test domain
    domain = Domain(L=1.0, N=24, dimensions=3)
    print(f"Domain: {domain.shape}, Total elements: {np.prod(domain.shape):,}")

    # Create BVP configuration
    config = BVPBlockConfig(
        block_size=12,
        overlap_ratio=0.1,
        max_memory_usage=0.8,
        envelope_tolerance=1e-6,
        max_envelope_iterations=20,
        quench_detection_enabled=False,  # Disable for demo
        impedance_calculation_enabled=False,  # Disable for demo
        enable_adaptive_sizing=True,
        enable_memory_optimization=True,
        enable_parallel_processing=True,
        enable_gpu_acceleration=True,
    )

    # Create BVP processor
    bvp_processor = BVPBlockProcessingSystem(domain, config)
    print(f"BVP Processor initialized with block size: {config.block_size}")

    # Create test source
    source = np.random.random(domain.shape) + 1j * np.random.random(domain.shape)
    print(f"Source field size: {source.nbytes / (1024**2):.2f} MB")

    # Solve BVP envelope equation
    print("\nSolving BVP envelope equation...")
    start_time = time.time()

    try:
        envelope = bvp_processor.solve_envelope_blocked(
            source, max_iterations=10, tolerance=1e-4  # Reduced for demo
        )
        processing_time = time.time() - start_time

        print(f"  ✓ BVP envelope solved: {processing_time:.3f}s")
        print(f"  Envelope shape: {envelope.shape}")
        print(f"  Envelope dtype: {envelope.dtype}")

        # Check solution quality
        residual = np.linalg.norm(envelope - source) / np.linalg.norm(source)
        print(f"  Relative residual: {residual:.2e}")

    except Exception as e:
        print(f"  ✗ BVP solution failed: {e}")

    # Show BVP processing statistics
    stats = bvp_processor.get_processing_stats()
    print(f"\nBVP Processing Statistics:")
    print(f"  Envelope Solves: {stats['envelope_solves']}")
    print(f"  Blocks Processed: {stats['blocks_processed']}")
    print(f"  Processing Time: {stats['processing_time']:.3f}s")
    print(f"  Memory Usage: {stats['memory_usage']}")

    # Cleanup
    bvp_processor.cleanup()
    print("  ✓ BVP cleanup completed")


def demonstrate_memory_optimization():
    """Demonstrate memory optimization capabilities."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Memory Optimization")
    print("=" * 60)

    # Test with different domain sizes
    domain_sizes = [16, 24, 32]

    for size in domain_sizes:
        print(f"\nTesting domain size: {size}x{size}x{size}")

        try:
            # Create domain
            domain = Domain(L=1.0, N=size, dimensions=3)

            # Create optimized configuration
            config = ProcessingConfig(
                mode=ProcessingMode.ADAPTIVE,
                max_memory_usage=0.7,  # Conservative memory usage
                min_block_size=4,
                max_block_size=min(12, size),  # Adaptive to domain size
                overlap_ratio=0.1,
                enable_memory_optimization=True,
                enable_adaptive_sizing=True,
                enable_parallel_processing=True,
            )

            # Create processor
            processor = EnhancedBlockProcessor(domain, config)

            # Create test field
            field = np.random.random(domain.shape) + 1j * np.random.random(domain.shape)
            field_size_mb = field.nbytes / (1024**2)

            print(f"  Field size: {field_size_mb:.2f} MB")
            print(f"  Block size: {processor.base_processor.block_size}")

            # Test processing
            start_time = time.time()
            result = processor.process_7d_field(field, operation="fft")
            processing_time = time.time() - start_time

            print(f"  ✓ Processing time: {processing_time:.3f}s")
            print(f"  Result shape: {result.shape}")

            # Show memory usage
            stats = processor.get_processing_stats()
            print(f"  Memory usage: {stats['memory_usage']}")

            # Cleanup
            processor.cleanup()

        except MemoryError:
            print(f"  ✗ Memory insufficient for size {size}")
        except Exception as e:
            print(f"  ✗ Error: {e}")


def demonstrate_adaptive_processing():
    """Demonstrate adaptive processing capabilities."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Adaptive Processing")
    print("=" * 60)

    # Create domain
    domain = Domain(L=1.0, N=32, dimensions=3)

    # Create adaptive configuration
    config = ProcessingConfig(
        mode=ProcessingMode.ADAPTIVE,
        max_memory_usage=0.8,
        min_block_size=4,
        max_block_size=16,
        overlap_ratio=0.1,
        enable_memory_optimization=True,
        enable_adaptive_sizing=True,
        enable_parallel_processing=True,
    )

    # Create processor
    processor = EnhancedBlockProcessor(domain, config)

    # Test with different field characteristics
    test_fields = [
        (
            "Small field",
            np.random.random((16, 16, 16)) + 1j * np.random.random((16, 16, 16)),
        ),
        (
            "Medium field",
            np.random.random((24, 24, 24)) + 1j * np.random.random((24, 24, 24)),
        ),
        (
            "Large field",
            np.random.random((32, 32, 32)) + 1j * np.random.random((32, 32, 32)),
        ),
    ]

    for field_name, field in test_fields:
        print(f"\nTesting {field_name}:")
        print(f"  Shape: {field.shape}")
        print(f"  Size: {field.nbytes / (1024**2):.2f} MB")

        # Optimize for this field
        processor.optimize_for_field(field)

        # Get optimized settings
        stats = processor.get_processing_stats()
        print(f"  Optimized block size: {stats['current_block_size']}")

        # Test processing
        try:
            start_time = time.time()
            result = processor.process_7d_field(field, operation="bvp_solve")
            processing_time = time.time() - start_time

            print(f"  ✓ Processing time: {processing_time:.3f}s")
            print(f"  Result shape: {result.shape}")

        except Exception as e:
            print(f"  ✗ Processing failed: {e}")

    # Cleanup
    processor.cleanup()
    print("\n  ✓ Adaptive processing demonstration completed")


def main():
    """Main demonstration function."""
    print("Enhanced Block Processing Demonstration")
    print("=" * 60)
    print("This demonstration shows the capabilities of the enhanced block")
    print("processing system for 7D BVP computations with intelligent")
    print("memory management and adaptive block sizing.")

    # Setup logging
    setup_logging()

    try:
        # Run demonstrations
        demonstrate_basic_block_processing()
        demonstrate_bvp_block_processing()
        demonstrate_memory_optimization()
        demonstrate_adaptive_processing()

        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("The enhanced block processing system provides:")
        print("  ✓ Intelligent memory management")
        print("  ✓ Adaptive block sizing")
        print("  ✓ GPU acceleration with CPU fallback")
        print("  ✓ BVP-specific optimizations")
        print("  ✓ Efficient data flow")
        print("  ✓ Resource cleanup")

    except Exception as e:
        print(f"\n✗ Demonstration failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
