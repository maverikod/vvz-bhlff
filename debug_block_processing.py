#!/usr/bin/env python3
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Debug script for block processing system.

This script debugs the block processing system with detailed logging
to identify where hanging occurs.
"""

import sys
import os
import numpy as np
import time
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from bhlff.core.domain.simple_block_processor import SimpleBlockProcessor, SimpleConfig
from bhlff.core.domain import Domain


def setup_debug_logging():
    """Setup debug logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def debug_initialization():
    """Debug processor initialization."""
    print("\n" + "="*60)
    print("DEBUG: Initialization")
    print("="*60)
    
    try:
        # Create very small test domain
        print("Creating domain...")
        domain = Domain(L=1.0, N=2, N_phi=2, N_t=2, dimensions=7)
        print(f"Domain created: {domain.shape}")
        print(f"Domain N: {domain.N}")
        print(f"Domain N_phi: {domain.N_phi}")
        print(f"Domain N_t: {domain.N_t}")
        print(f"Domain dimensions: {domain.dimensions}")
        
        # Create processing config
        print("Creating config...")
        config = SimpleConfig(
            block_size=2,
            overlap_ratio=0.1,
            max_memory_usage=0.7,
            enable_adaptive_sizing=True,
            enable_memory_optimization=True,
            enable_parallel_processing=True,
            max_field_size_mb=1.0
        )
        print("Config created")
        
        # Create processor
        print("Creating processor...")
        processor = SimpleBlockProcessor(domain, config)
        print("Processor created successfully")
        
        return processor
        
    except Exception as e:
        print(f"ERROR in initialization: {e}")
        import traceback
        traceback.print_exc()
        return None


def debug_cpu_processing(processor):
    """Debug CPU processing."""
    print("\n" + "="*60)
    print("DEBUG: CPU Processing")
    print("="*60)
    
    try:
        # Create very small field
        print("Creating test field...")
        field = np.random.random((2, 2, 2, 2, 2, 2, 2)) + 1j * np.random.random((2, 2, 2, 2, 2, 2, 2))
        print(f"Field created: {field.shape}, {field.nbytes} bytes")
        
        # Test CPU processing
        print("Starting CPU processing...")
        start_time = time.time()
        result = processor._process_cpu_optimized(field, "fft")
        processing_time = time.time() - start_time
        
        print(f"CPU processing completed: {processing_time:.3f}s")
        print(f"Result shape: {result.shape}")
        print(f"Result dtype: {result.dtype}")
        
        return True
        
    except Exception as e:
        print(f"ERROR in CPU processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def debug_block_operations(processor):
    """Debug block operations."""
    print("\n" + "="*60)
    print("DEBUG: Block Operations")
    print("="*60)
    
    try:
        # Create very small field
        print("Creating test field...")
        field = np.random.random((2, 2, 2, 2, 2, 2, 2)) + 1j * np.random.random((2, 2, 2, 2, 2, 2, 2))
        print(f"Field created: {field.shape}")
        
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
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                import traceback
                traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"ERROR in block operations: {e}")
        import traceback
        traceback.print_exc()
        return False


def debug_adaptive_processing():
    """Debug adaptive processing."""
    print("\n" + "="*60)
    print("DEBUG: Adaptive Processing")
    print("="*60)
    
    try:
        # Create very small domain
        print("Creating domain...")
        domain = Domain(L=1.0, N=2, N_phi=2, N_t=2, dimensions=7)
        print(f"Domain created: {domain.shape}")
        
        # Create config
        print("Creating config...")
        config = SimpleConfig(
            block_size=2,
            overlap_ratio=0.1,
            max_memory_usage=0.7,
            enable_adaptive_sizing=True,
            enable_memory_optimization=True,
            enable_parallel_processing=True,
            max_field_size_mb=1.0
        )
        print("Config created")
        
        # Create processor
        print("Creating processor...")
        processor = SimpleBlockProcessor(domain, config)
        print("Processor created")
        
        # Create very small field
        print("Creating test field...")
        field = np.random.random(domain.shape) + 1j * np.random.random(domain.shape)
        print(f"Field created: {field.shape}")
        
        # Test processing
        print("Starting processing...")
        start_time = time.time()
        result = processor.process_7d_field(field, operation="fft")
        processing_time = time.time() - start_time
        
        print(f"Processing completed: {processing_time:.3f}s")
        print(f"Result shape: {result.shape}")
        
        # Cleanup
        print("Cleaning up...")
        processor.cleanup()
        print("Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"ERROR in adaptive processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main debug function."""
    print("Block Processing Debug Script")
    print("="*60)
    print("This script debugs the block processing system with detailed logging")
    print("to identify where hanging occurs.")
    
    # Setup debug logging
    setup_debug_logging()
    
    try:
        # Debug initialization
        processor = debug_initialization()
        if processor is None:
            print("FAILED: Initialization")
            return
        
        # Debug CPU processing
        if not debug_cpu_processing(processor):
            print("FAILED: CPU Processing")
            return
        
        # Debug block operations
        if not debug_block_operations(processor):
            print("FAILED: Block Operations")
            return
        
        # Debug adaptive processing
        if not debug_adaptive_processing():
            print("FAILED: Adaptive Processing")
            return
        
        print("\n" + "="*60)
        print("DEBUG COMPLETED SUCCESSFULLY")
        print("="*60)
        print("All debug tests passed!")
        
    except Exception as e:
        print(f"\n✗ Debug failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
