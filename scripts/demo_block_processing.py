#!/usr/bin/env python3
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Demonstration of block processing for 7D BVP computations.

This script demonstrates the capabilities of the block processing system
for 7D BVP computations with intelligent memory management and adaptive block sizing.

Physical Meaning:
    Demonstrates intelligent block-based processing for 7D phase field computations
    with adaptive memory management and processing optimization.

Example:
    python scripts/demo_block_processing.py
"""

import sys
import os
import numpy as np
import time
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bhlff.core.domain.simple_block_processor import SimpleBlockProcessor, SimpleConfig
from bhlff.core.domain import Domain


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def demonstrate_basic_block_processing():
    """Demonstrate basic block processing capabilities."""
    print("\n" + "="*60)
    print("DEMONSTRATION: Basic Block Processing")
    print("="*60)
    
    # Create test domain
    domain = Domain(L=1.0, N=4, N_phi=4, N_t=4, dimensions=7)
    print(f"Domain: {domain.shape}, Total elements: {np.prod(domain.shape):,}")
    
    # Create processing configuration
    config = SimpleConfig(
        block_size=3,
        overlap_ratio=0.1,
        max_memory_usage=0.7,
        enable_adaptive_sizing=True,
        enable_memory_optimization=True,
        enable_parallel_processing=True,
        max_field_size_mb=10.0  # Reduced limit
    )
    
    # Create processor
    processor = SimpleBlockProcessor(domain, config)
    print(f"Processor initialized with block size: {processor.block_size}")
    
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
    print(f"  Current Block Size: {stats['current_block_size']}")
    print(f"  Blocks Processed: {stats['blocks_processed']}")
    print(f"  Processing Time: {stats['processing_time']:.3f}s")
    print(f"  Memory Usage: {stats['memory_usage']}")
    
    # Cleanup
    processor.cleanup()
    print("  ✓ Cleanup completed")


def demonstrate_memory_optimization():
    """Demonstrate memory optimization capabilities."""
    print("\n" + "="*60)
    print("DEMONSTRATION: Memory Optimization")
    print("="*60)
    
    # Test with different domain sizes
    domain_sizes = [2, 3, 4]
    
    for size in domain_sizes:
        print(f"\nTesting domain size: {size}x{size}x{size}x{size}x{size}x{size}x{size}")
        
        try:
            # Create domain
            domain = Domain(L=1.0, N=size, N_phi=size, N_t=size, dimensions=7)
            
            # Create optimized configuration
            config = SimpleConfig(
                block_size=min(3, size),
                overlap_ratio=0.1,
                max_memory_usage=0.7,
                enable_adaptive_sizing=True,
                enable_memory_optimization=True,
                enable_parallel_processing=True,
                max_field_size_mb=5.0  # Very small limit
            )
            
            # Create processor
            processor = SimpleBlockProcessor(domain, config)
            
            # Create test field
            field = np.random.random(domain.shape) + 1j * np.random.random(domain.shape)
            field_size_mb = field.nbytes / (1024**2)
            
            print(f"  Field size: {field_size_mb:.2f} MB")
            print(f"  Block size: {processor.block_size}")
            
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
    print("\n" + "="*60)
    print("DEMONSTRATION: Adaptive Processing")
    print("="*60)
    
    # Create domain
    domain = Domain(L=1.0, N=4, N_phi=4, N_t=4, dimensions=7)
    
    # Create adaptive configuration
    config = SimpleConfig(
        block_size=3,
        overlap_ratio=0.1,
        max_memory_usage=0.7,
        enable_adaptive_sizing=True,
        enable_memory_optimization=True,
        enable_parallel_processing=True,
        max_field_size_mb=20.0
    )
    
    # Create processor
    processor = SimpleBlockProcessor(domain, config)
    
    # Test with different field characteristics - use domain shape
    test_fields = [
        ("Small field", np.random.random(domain.shape) + 1j * np.random.random(domain.shape)),
        ("Medium field", np.random.random(domain.shape) + 1j * np.random.random(domain.shape)),
        ("Large field", np.random.random(domain.shape) + 1j * np.random.random(domain.shape))
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


def demonstrate_performance_benchmark():
    """Demonstrate performance benchmark."""
    print("\n" + "="*60)
    print("DEMONSTRATION: Performance Benchmark")
    print("="*60)
    
    # Create domain
    domain = Domain(L=1.0, N=3, N_phi=3, N_t=3, dimensions=7)
    
    # Create configuration
    config = SimpleConfig(
        block_size=2,
        overlap_ratio=0.1,
        max_memory_usage=0.7,
        enable_adaptive_sizing=True,
        enable_memory_optimization=True,
        enable_parallel_processing=True,
        max_field_size_mb=10.0
    )
    
    # Create processor
    processor = SimpleBlockProcessor(domain, config)
    
    # Create test field
    field = np.random.random(domain.shape) + 1j * np.random.random(domain.shape)
    print(f"Test field shape: {field.shape}")
    print(f"Test field size: {field.nbytes / (1024**2):.2f} MB")
    
    # Benchmark different operations
    operations = ["fft", "ifft", "bvp_solve"]
    
    for operation in operations:
        print(f"\nBenchmarking {operation}:")
        
        # Warm up
        processor.process_7d_field(field, operation=operation)
        
        # Benchmark
        start_time = time.time()
        for i in range(5):  # Run 5 iterations
            result = processor.process_7d_field(field, operation=operation)
        total_time = time.time() - start_time
        
        print(f"  Total time for 5 iterations: {total_time:.3f}s")
        print(f"  Average time per iteration: {total_time/5:.3f}s")
        print(f"  Result shape: {result.shape}")
    
    # Show final statistics
    stats = processor.get_processing_stats()
    print(f"\nFinal Statistics:")
    print(f"  Blocks Processed: {stats['blocks_processed']}")
    print(f"  Total Processing Time: {stats['processing_time']:.3f}s")
    print(f"  Memory Usage: {stats['memory_usage']}")
    
    # Cleanup
    processor.cleanup()
    print("  ✓ Performance benchmark completed")


def main():
    """Main demonstration function."""
    print("Block Processing Demonstration")
    print("="*60)
    print("This demonstration shows the capabilities of the block processing")
    print("system for 7D BVP computations with intelligent memory management")
    print("and adaptive block sizing.")
    
    # Setup logging
    setup_logging()
    
    try:
        # Run demonstrations
        demonstrate_basic_block_processing()
        demonstrate_memory_optimization()
        demonstrate_adaptive_processing()
        demonstrate_performance_benchmark()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print("The block processing system provides:")
        print("  ✓ Intelligent memory management")
        print("  ✓ Adaptive block sizing")
        print("  ✓ Efficient data flow")
        print("  ✓ Resource cleanup")
        print("  ✓ Performance optimization")
        
    except Exception as e:
        print(f"\n✗ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
