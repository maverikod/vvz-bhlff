#!/usr/bin/env python3
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Demonstration script for vectorized processing.

This script demonstrates the vectorized processing capabilities
for 7D phase field computations with CUDA acceleration.
"""

import sys
import os
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bhlff.core.domain import Domain
from bhlff.core.domain.vectorized_block_processor import VectorizedBlockProcessor
from bhlff.core.bvp.bvp_core.bvp_vectorized_processor import BVPVectorizedProcessor


def setup_logging():
    """Setup logging for demonstration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def demonstrate_vectorized_processing():
    """Demonstrate vectorized processing capabilities."""
    print("=" * 80)
    print("7D Phase Field Vectorized Processing Demonstration")
    print("=" * 80)
    
    # Setup logging
    setup_logging()
    
    # Create domain and config for vectorized processing
    domain = Domain(L=1.0, N=8, dimensions=7)
    config = {
        "carrier_frequency": 1e15,
        "envelope_equation": {
            "kappa_0": 1.0,
            "kappa_2": 0.1,
            "chi_prime": 1.0,
            "chi_double_prime_0": 0.1,
            "k0": 1.0
        }
    }
    
    print("\n1. Vectorized Block Processing")
    print("-" * 40)
    
    # Create vectorized block processor
    processor = VectorizedBlockProcessor(domain, block_size=4, use_cuda=True)
    
    # Get vectorization info
    vectorization_info = processor.get_vectorization_info()
    print(f"Vectorized processing: {vectorization_info['vectorized_processing']}")
    print(f"CUDA acceleration: {vectorization_info['cuda_acceleration']}")
    print(f"Batch processing: {vectorization_info['batch_processing']}")
    print(f"Available operations: {vectorization_info['vectorized_operations']}")
    
    print("\n2. Vectorized BVP Processing")
    print("-" * 40)
    
    # Create vectorized BVP processor
    bvp_processor = BVPVectorizedProcessor(domain, config, block_size=4, use_cuda=True)
    
    # Get BVP vectorization info
    bvp_info = bvp_processor.get_vectorized_bvp_info()
    print(f"BVP operations: {bvp_info['bvp_operations']}")
    print(f"Vectorized acceleration: {bvp_info['vectorized_acceleration']}")
    print(f"BVP parameters: {bvp_info['bvp_parameters']}")
    
    print("\n3. Memory Usage Analysis")
    print("-" * 40)
    
    # Get memory usage
    memory_usage = processor.get_memory_usage()
    print(f"Block memory usage: {memory_usage['block_memory_gb']:.2f} GB")
    print(f"Total memory usage: {memory_usage['total_memory_gb']:.2f} GB")
    print(f"Total blocks: {memory_usage['total_blocks']}")
    print(f"Blocks per dimension: {memory_usage['blocks_per_dimension']}")
    
    print("\n4. CUDA Information")
    print("-" * 40)
    
    if processor.use_cuda:
        cuda_info = processor.get_cuda_info()
        print(f"CUDA available: {cuda_info['cuda_available']}")
        print(f"Device ID: {cuda_info['device_id']}")
        print(f"Total GPU memory: {cuda_info['total_memory_gb']:.1f} GB")
        print(f"Free GPU memory: {cuda_info['free_memory_gb']:.1f} GB")
    else:
        print("CUDA not available, using CPU vectorization")
    
    print("\n5. Block Processing Demonstration")
    print("-" * 40)
    
    # Demonstrate block processing
    print("Processing blocks with vectorized operations...")
    
    # Process blocks with different operations
    operations = ["fft", "convolution", "gradient", "bvp_solve"]
    
    for operation in operations:
        print(f"\nProcessing with {operation} operation...")
        
        try:
            # Process blocks vectorized
            result = processor.process_blocks_vectorized(operation=operation, batch_size=4)
            print(f"  ✓ {operation} processing completed")
            print(f"  Result shape: {result.shape}")
            print(f"  Result dtype: {result.dtype}")
            
        except Exception as e:
            print(f"  ✗ {operation} processing failed: {e}")
    
    print("\n6. BVP Envelope Solution Demonstration")
    print("-" * 40)
    
    # Generate synthetic source
    source = generate_synthetic_source(domain.shape)
    print(f"Generated source shape: {source.shape}")
    print(f"Source memory usage: {source.nbytes / 1e9:.2f} GB")
    
    # Solve BVP envelope equation
    print("\nSolving BVP envelope equation with vectorized processing...")
    
    try:
        envelope = bvp_processor.solve_envelope_vectorized(
            source, 
            max_iterations=10, 
            tolerance=1e-6,
            batch_size=4
        )
        
        print(f"✓ BVP envelope solution completed")
        print(f"  Envelope shape: {envelope.shape}")
        print(f"  Envelope dtype: {envelope.dtype}")
        print(f"  Max amplitude: {np.max(np.abs(envelope)):.2e}")
        print(f"  Min amplitude: {np.min(np.abs(envelope)):.2e}")
        
    except Exception as e:
        print(f"✗ BVP envelope solution failed: {e}")
    
    print("\n7. Quench Detection Demonstration")
    print("-" * 40)
    
    if 'envelope' in locals():
        print("Detecting quenches with vectorized processing...")
        
        try:
            quench_results = bvp_processor.detect_quenches_vectorized(envelope, batch_size=4)
            
            print(f"✓ Quench detection completed")
            print(f"  Total quenches: {quench_results['total_quenches']}")
            print(f"  Detection method: {quench_results['detection_method']}")
            print(f"  Quench blocks: {len(quench_results['quench_blocks'])}")
            
        except Exception as e:
            print(f"✗ Quench detection failed: {e}")
    
    print("\n8. Impedance Computation Demonstration")
    print("-" * 40)
    
    if 'envelope' in locals():
        print("Computing impedance with vectorized processing...")
        
        try:
            impedance = bvp_processor.compute_impedance_vectorized(envelope, batch_size=4)
            
            print(f"✓ Impedance computation completed")
            print(f"  Impedance shape: {impedance.shape}")
            print(f"  Impedance dtype: {impedance.dtype}")
            print(f"  Max impedance: {np.max(np.abs(impedance)):.2e}")
            print(f"  Min impedance: {np.min(np.abs(impedance)):.2e}")
            
        except Exception as e:
            print(f"✗ Impedance computation failed: {e}")
    
    print("\n9. Performance Comparison")
    print("-" * 40)
    
    print("Vectorized processing advantages:")
    print("  ✓ CUDA acceleration for GPU processing")
    print("  ✓ Batch processing for multiple blocks")
    print("  ✓ Vectorized operations for maximum performance")
    print("  ✓ Memory-efficient block processing")
    print("  ✓ Automatic fallback to CPU when CUDA unavailable")
    
    print("\n10. 7D BVP Theory Integration")
    print("-" * 40)
    
    print("Vectorized processing is fully integrated with 7D BVP theory:")
    print("  ✓ 7D phase field envelope equation")
    print("  ✓ VBP envelope configurations")
    print("  ✓ Phase coherence and topological charge")
    print("  ✓ Memory-efficient 7D space-time processing")
    print("  ✓ CUDA-accelerated computations")
    
    print("\n" + "=" * 80)
    print("Vectorized Processing Demonstration Completed Successfully!")
    print("=" * 80)


def generate_synthetic_source(shape):
    """Generate synthetic source for demonstration."""
    # Generate 7D source with realistic properties
    source = np.zeros(shape, dtype=np.complex128)
    
    # Add spatial variations
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for l in range(shape[3]):
                    for m in range(shape[4]):
                        for n in range(shape[5]):
                            for o in range(shape[6]):
                                # Create realistic source pattern
                                r = np.sqrt(i**2 + j**2 + k**2 + l**2 + m**2 + n**2 + o**2)
                                source[i, j, k, l, m, n, o] = np.exp(-r**2 / 10.0) * np.exp(1j * r)
    
    return source


if __name__ == "__main__":
    demonstrate_vectorized_processing()

