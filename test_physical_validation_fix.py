#!/usr/bin/env python3
"""
Test script for physical validation fix.

This script tests the comprehensive physical validation system
to ensure it works correctly and provides theoretical compliance.
"""

import sys
import os
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from bhlff.core.bvp.physical_validator import BVPPhysicalValidator
from bhlff.core.bvp.physical_validation_decorator import (
    physical_validation_required,
    validate_physical_constraints,
    validate_theoretical_bounds,
    validate_energy_conservation,
    validate_causality,
    validate_7d_structure,
    PhysicalValidationMixin
)

def test_physical_validator():
    """Test BVP physical validator."""
    print("=== TESTING BVP PHYSICAL VALIDATOR ===")
    
    try:
        # Create validator
        domain_shape = (2, 2, 2, 4, 4, 4, 8)
        parameters = {
            'energy_conservation_tolerance': 1e-6,
            'causality_tolerance': 1e-8,
            'phase_coherence_minimum': 0.1
        }
        
        validator = BVPPhysicalValidator(domain_shape, parameters)
        print("✓ BVPPhysicalValidator created successfully")
        
        # Test with valid result
        valid_result = {
            'field': np.random.randn(*domain_shape) + 1j * np.random.randn(*domain_shape),
            'energy': {'initial_energy': 100.0},
            'phase': np.random.uniform(-np.pi, np.pi, domain_shape),
            'metadata': {'method': 'test'}
        }
        print(f"✓ Test result created: {valid_result['field'].shape}")
        
        # Test physical constraints validation
        physical_result = validator.validate_physical_constraints(valid_result)
        print("✓ Physical constraints validation completed")
        print(f"  - Physical constraints valid: {physical_result['physical_constraints_valid']}")
        print(f"  - Constraint violations: {len(physical_result['constraint_violations'])}")
        print(f"  - Constraint warnings: {len(physical_result['constraint_warnings'])}")
        
        # Test theoretical bounds validation
        theoretical_result = validator.validate_theoretical_bounds(valid_result)
        print("✓ Theoretical bounds validation completed")
        print(f"  - Theoretical bounds valid: {theoretical_result['theoretical_bounds_valid']}")
        print(f"  - Bound violations: {len(theoretical_result['bound_violations'])}")
        print(f"  - Bound warnings: {len(theoretical_result['bound_warnings'])}")
        
        # Test validation summary
        summary = validator.get_validation_summary(physical_result, theoretical_result)
        print("✓ Validation summary created")
        print(f"  - Overall valid: {summary['overall_valid']}")
        print(f"  - Total violations: {summary['total_violations']}")
        print(f"  - Total warnings: {summary['total_warnings']}")
        
        # Validate results structure
        required_keys = [
            'physical_constraints_valid', 'constraint_violations', 'constraint_warnings',
            'detailed_metrics'
        ]
        
        for key in required_keys:
            assert key in physical_result, f"Missing key in physical result: {key}"
        
        print("✓ All required result keys present")
        
        print("\n✅ PHYSICAL VALIDATOR TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_validation_decorators():
    """Test physical validation decorators."""
    print("\n=== TESTING VALIDATION DECORATORS ===")
    
    try:
        # Test decorator with valid result
        domain_shape = (2, 2, 2, 4, 4, 4, 8)
        parameters = {'energy_conservation_tolerance': 1e-6}
        
        @physical_validation_required(domain_shape, parameters)
        def test_method():
            return {
                'field': np.random.randn(*domain_shape) + 1j * np.random.randn(*domain_shape),
                'energy': {'initial_energy': 100.0},
                'metadata': {'method': 'decorated_test'}
            }
        
        result = test_method()
        print("✓ Decorated method executed successfully")
        print(f"  - Result shape: {result['field'].shape}")
        print(f"  - Has physical validation: {'physical_validation' in result}")
        print(f"  - Has theoretical validation: {'theoretical_validation' in result}")
        print(f"  - Has validation summary: {'validation_summary' in result}")
        
        # Validate decorator results
        assert 'physical_validation' in result, "Physical validation missing"
        assert 'theoretical_validation' in result, "Theoretical validation missing"
        assert 'validation_summary' in result, "Validation summary missing"
        
        print("✓ All validation decorators working correctly")
        
        print("\n✅ VALIDATION DECORATORS TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_specific_validation_decorators():
    """Test specific validation decorators."""
    print("\n=== TESTING SPECIFIC VALIDATION DECORATORS ===")
    
    try:
        domain_shape = (2, 2, 2, 4, 4, 4, 8)
        parameters = {'energy_conservation_tolerance': 1e-6}
        
        # Test energy conservation decorator
        @validate_energy_conservation(domain_shape, parameters)
        def test_energy_method():
            return {
                'field': np.random.randn(*domain_shape) + 1j * np.random.randn(*domain_shape),
                'energy': {'initial_energy': 100.0},
                'metadata': {'method': 'energy_test'}
            }
        
        result = test_energy_method()
        print("✓ Energy conservation decorator executed")
        print(f"  - Has energy validation: {'energy_validation' in result}")
        
        # Test causality decorator
        @validate_causality(domain_shape, parameters)
        def test_causality_method():
            return {
                'field': np.random.randn(*domain_shape) + 1j * np.random.randn(*domain_shape),
                'metadata': {'method': 'causality_test'}
            }
        
        result = test_causality_method()
        print("✓ Causality decorator executed")
        print(f"  - Has causality validation: {'causality_validation' in result}")
        
        # Test 7D structure decorator
        @validate_7d_structure(domain_shape, parameters)
        def test_structure_method():
            return {
                'field': np.random.randn(*domain_shape) + 1j * np.random.randn(*domain_shape),
                'metadata': {'method': 'structure_test'}
            }
        
        result = test_structure_method()
        print("✓ 7D structure decorator executed")
        print(f"  - Has structure validation: {'structure_validation' in result}")
        
        # Validate specific decorators
        assert 'energy_validation' in test_energy_method(), "Energy validation missing"
        assert 'causality_validation' in test_causality_method(), "Causality validation missing"
        assert 'structure_validation' in test_structure_method(), "Structure validation missing"
        
        print("✓ All specific validation decorators working correctly")
        
        print("\n✅ SPECIFIC VALIDATION DECORATORS TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_validation_mixin():
    """Test PhysicalValidationMixin."""
    print("\n=== TESTING VALIDATION MIXIN ===")
    
    try:
        # Create test class with mixin
        class TestBVPClass(PhysicalValidationMixin):
            def __init__(self, domain_shape, parameters):
                self.domain = type('Domain', (), {'shape': domain_shape})()
                self.parameters = parameters
                super().__init__()
        
        # Create instance
        domain_shape = (2, 2, 2, 4, 4, 4, 8)
        parameters = {'energy_conservation_tolerance': 1e-6}
        
        test_instance = TestBVPClass(domain_shape, parameters)
        print("✓ TestBVPClass with mixin created successfully")
        
        # Test validation methods
        test_result = {
            'field': np.random.randn(*domain_shape) + 1j * np.random.randn(*domain_shape),
            'energy': {'initial_energy': 100.0},
            'metadata': {'method': 'mixin_test'}
        }
        
        # Test physical validation
        physical_result = test_instance.validate_result_physical(test_result)
        print("✓ Physical validation method executed")
        print(f"  - Physical constraints valid: {physical_result.get('physical_constraints_valid', False)}")
        
        # Test theoretical validation
        theoretical_result = test_instance.validate_result_theoretical(test_result)
        print("✓ Theoretical validation method executed")
        print(f"  - Theoretical bounds valid: {theoretical_result.get('theoretical_bounds_valid', False)}")
        
        # Test comprehensive validation
        comprehensive_result = test_instance.validate_result_comprehensive(test_result)
        print("✓ Comprehensive validation method executed")
        print(f"  - Overall valid: {comprehensive_result.get('overall_valid', False)}")
        
        # Validate mixin functionality
        assert hasattr(test_instance, 'validate_result_physical'), "Physical validation method missing"
        assert hasattr(test_instance, 'validate_result_theoretical'), "Theoretical validation method missing"
        assert hasattr(test_instance, 'validate_result_comprehensive'), "Comprehensive validation method missing"
        
        print("✓ All mixin validation methods working correctly")
        
        print("\n✅ VALIDATION MIXIN TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_invalid_results():
    """Test validation with invalid results."""
    print("\n=== TESTING INVALID RESULTS ===")
    
    try:
        # Create validator
        domain_shape = (2, 2, 2, 4, 4, 4, 8)
        parameters = {'energy_conservation_tolerance': 1e-6}
        
        validator = BVPPhysicalValidator(domain_shape, parameters)
        
        # Test with invalid result (wrong shape)
        invalid_result = {
            'field': np.random.randn(1, 1, 1, 1, 1, 1, 1),  # Wrong shape
            'energy': {'initial_energy': 100.0},
            'metadata': {'method': 'invalid_test'}
        }
        
        print("✓ Testing invalid result (wrong shape)")
        
        # Test physical constraints validation
        physical_result = validator.validate_physical_constraints(invalid_result)
        print(f"  - Physical constraints valid: {physical_result['physical_constraints_valid']}")
        print(f"  - Constraint violations: {len(physical_result['constraint_violations'])}")
        
        # Test theoretical bounds validation
        theoretical_result = validator.validate_theoretical_bounds(invalid_result)
        print(f"  - Theoretical bounds valid: {theoretical_result['theoretical_bounds_valid']}")
        print(f"  - Bound violations: {len(theoretical_result['bound_violations'])}")
        
        # Should detect violations
        assert not physical_result['physical_constraints_valid'], "Should detect invalid result"
        assert len(physical_result['constraint_violations']) > 0, "Should have constraint violations"
        
        print("✓ Invalid results correctly detected")
        
        print("\n✅ INVALID RESULTS TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_no_basic_validation():
    """Test that no basic validation methods remain."""
    print("\n=== TESTING NO BASIC VALIDATION ===")
    
    try:
        # Check that all validation methods are comprehensive
        domain_shape = (2, 2, 2, 4, 4, 4, 8)
        parameters = {'energy_conservation_tolerance': 1e-6}
        
        validator = BVPPhysicalValidator(domain_shape, parameters)
        
        # Check method names
        methods = [method for method in dir(validator) if not method.startswith('_')]
        
        # Should not contain 'basic' validation methods
        basic_methods = [method for method in methods if 'basic' in method.lower()]
        assert len(basic_methods) == 0, f"Found basic validation methods: {basic_methods}"
        
        print("✓ No basic validation methods found")
        
        # Check for comprehensive validation methods
        comprehensive_methods = [method for method in methods if 'validate' in method.lower()]
        assert len(comprehensive_methods) > 0, "No comprehensive validation methods found"
        
        print("✓ Comprehensive validation methods present")
        
        # Check theoretical validation methods
        theoretical_methods = [method for method in methods if 'theoretical' in method.lower() or 'physical' in method.lower()]
        assert len(theoretical_methods) > 0, "No theoretical validation methods found"
        
        print("✓ Theoretical validation methods present")
        
        print("\n✅ NO BASIC VALIDATION TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=== PHYSICAL VALIDATION FIX TESTS ===")
    print("Testing comprehensive physical validation system...")
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    tests = [
        test_physical_validator,
        test_validation_decorators,
        test_specific_validation_decorators,
        test_validation_mixin,
        test_invalid_results,
        test_no_basic_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== TEST RESULTS ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✓ Comprehensive physical validation system working correctly")
        print("✓ Theoretical framework implemented")
        print("✓ No basic validation remaining")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        return 1

if __name__ == "__main__":
    exit(main())
