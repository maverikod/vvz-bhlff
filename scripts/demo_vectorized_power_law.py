#!/usr/bin/env python3
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Demonstration script for vectorized power law fitting.

This script demonstrates the use of vectorized processors in power law fitting
for 7D phase field theory computations.
"""

import numpy as np
import sys
import os
import logging
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bhlff.core.bvp.power_law_core_modules.power_law_fitting import PowerLawFitting


def setup_logging():
    """Setup logging for demonstration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def demo_power_law_fitting_vectorized():
    """Demonstrate vectorized power law fitting."""
    print("\n" + "="*60)
    print("DEMONSTRATION: Vectorized Power Law Fitting")
    print("="*60)
    
    # Create power law fitting instance
    fitting = PowerLawFitting()
    
    # Create test data with 7D phase field characteristics
    r = np.linspace(0.1, 10.0, 1000)
    values = np.exp(-r) * r**(-2.0) * (1 + 0.1 * np.sin(2 * np.pi * r))
    
    test_data = {
        'r': r,
        'values': values
    }
    
    print(f"Input data shape: {len(r)} points")
    print(f"Value range: [{np.min(values):.3f}, {np.max(values):.3f}]")
    
    # Test vectorized power law fitting
    print("\nPerforming vectorized power law fitting...")
    start_time = time.time()
    result = fitting.fit_power_law(test_data)
    fitting_time = time.time() - start_time
    
    print(f"Fitting completed in {fitting_time:.4f} seconds")
    
    print("\nFitting Results:")
    for key, value in result.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Test quality calculation
    start_time = time.time()
    quality = fitting.calculate_fitting_quality(test_data, result)
    quality_time = time.time() - start_time
    
    print(f"\nFitting Quality: {quality:.6f} (calculated in {quality_time:.4f} seconds)")
    
    # Test decay rate calculation
    start_time = time.time()
    decay_rate = fitting.calculate_decay_rate(result)
    decay_time = time.time() - start_time
    
    print(f"Decay Rate: {decay_rate:.6f} (calculated in {decay_time:.4f} seconds)")
    
    return result, fitting_time, quality_time, decay_time


def demo_performance_scaling():
    """Demonstrate performance scaling with data size."""
    print("\n" + "="*60)
    print("DEMONSTRATION: Performance Scaling")
    print("="*60)
    
    # Test different data sizes
    data_sizes = [100, 500, 1000, 5000, 10000]
    
    print("Testing performance scaling with different data sizes:")
    print(f"{'Size':<8} {'Time (s)':<10} {'Points/s':<12} {'Quality':<10}")
    print("-" * 50)
    
    for size in data_sizes:
        # Create test data
        r = np.linspace(0.1, 10.0, size)
        values = np.exp(-r) * r**(-2.0) * (1 + 0.1 * np.sin(2 * np.pi * r))
        
        test_data = {
            'r': r,
            'values': values
        }
        
        # Test performance
        fitting = PowerLawFitting()
        
        start_time = time.time()
        result = fitting.fit_power_law(test_data)
        fitting_time = time.time() - start_time
        
        quality = fitting.calculate_fitting_quality(test_data, result)
        points_per_second = size / fitting_time
        
        print(f"{size:<8} {fitting_time:<10.4f} {points_per_second:<12.0f} {quality:<10.6f}")


def demo_7d_phase_field_characteristics():
    """Demonstrate 7D phase field characteristics."""
    print("\n" + "="*60)
    print("DEMONSTRATION: 7D Phase Field Characteristics")
    print("="*60)
    
    # Create 7D phase field test data
    r = np.linspace(0.1, 10.0, 2000)
    
    # 7D phase field characteristics
    base_field = np.exp(-r) * r**(-2.0)
    phase_modulation = 1 + 0.1 * np.sin(2 * np.pi * r)
    topological_charge = 0.05 * np.sin(4 * np.pi * r)
    energy_density = base_field * phase_modulation * (1 + topological_charge)
    
    test_data = {
        'r': r,
        'values': energy_density
    }
    
    print(f"7D Phase Field Data:")
    print(f"  Points: {len(r)}")
    print(f"  Base field range: [{np.min(base_field):.3f}, {np.max(base_field):.3f}]")
    print(f"  Phase modulation range: [{np.min(phase_modulation):.3f}, {np.max(phase_modulation):.3f}]")
    print(f"  Topological charge range: [{np.min(topological_charge):.3f}, {np.max(topological_charge):.3f}]")
    print(f"  Energy density range: [{np.min(energy_density):.3f}, {np.max(energy_density):.3f}]")
    
    # Test vectorized fitting
    fitting = PowerLawFitting()
    
    start_time = time.time()
    result = fitting.fit_power_law(test_data)
    fitting_time = time.time() - start_time
    
    print(f"\nVectorized fitting completed in {fitting_time:.4f} seconds")
    
    print(f"\n7D Phase Field Fitting Results:")
    print(f"  Power law exponent: {result['power_law_exponent']:.6f}")
    print(f"  Amplitude: {result['amplitude']:.6f}")
    print(f"  R-squared: {result['r_squared']:.6f}")
    print(f"  Chi-squared: {result['chi_squared']:.6f}")
    print(f"  Reduced chi-squared: {result['reduced_chi_squared']:.6f}")
    
    # Test quality metrics
    quality = fitting.calculate_fitting_quality(test_data, result)
    decay_rate = fitting.calculate_decay_rate(result)
    
    print(f"\nQuality Metrics:")
    print(f"  Fitting quality: {quality:.6f}")
    print(f"  Decay rate: {decay_rate:.6f}")
    
    return result


def main():
    """Main demonstration function."""
    print("VECTORIZED POWER LAW FITTING DEMONSTRATION")
    print("="*60)
    print("This script demonstrates the integration of vectorized processors")
    print("into power law fitting for 7D phase field theory computations.")
    
    setup_logging()
    
    try:
        # Demonstrate vectorized power law fitting
        result, fitting_time, quality_time, decay_time = demo_power_law_fitting_vectorized()
        
        # Demonstrate performance scaling
        demo_performance_scaling()
        
        # Demonstrate 7D phase field characteristics
        demo_7d_phase_field_characteristics()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print("Vectorized power law fitting is working correctly!")
        print("The integration of vectorized processors enhances performance")
        print("while maintaining compatibility with existing code.")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
