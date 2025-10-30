"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests to ensure legacy code has been properly removed.
"""

import pytest
import os
import re


class TestLegacyCodeAbsence:
    """
    Tests to verify that legacy code has been properly removed.
    """

    def test_no_legacy_test_classes(self):
        """
        Test that legacy test class aliases have been removed.
        """
        # Check that legacy test classes are not present
        legacy_classes = [
            "TestPhysicalValidation",
            "TestFrequencyDependentPropertiesPhysics",
            "TestNonlinearCoefficientsPhysics",
            "TestFFTSolver7DValidation",
        ]

        for class_name in legacy_classes:
            # Search for class definitions in test files
            found = False
            for root, dirs, files in os.walk("tests"):
                for file in files:
                    if file.endswith(".py"):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                                if f"class {class_name}(" in content:
                                    found = True
                                    break
                        except:
                            continue
                if found:
                    break

            assert not found, f"Found legacy class {class_name} in test files"

    def test_no_legacy_validation_methods(self):
        """
        Test that legacy validation methods have been removed.
        """
        # Check that legacy validation methods are not present
        legacy_methods = [
            "validate_interference_patterns(",
            "validate_interference_frequencies(",
        ]

        for method in legacy_methods:
            # Search for method definitions in validation files
            found = False
            for root, dirs, files in os.walk(
                "bhlff/models/level_c/beating/validation_basic"
            ):
                for file in files:
                    if file.endswith(".py"):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                                if f"def {method}" in content:
                                    found = True
                                    break
                        except:
                            continue
                if found:
                    break

            assert not found, f"Found legacy method {method} in validation files"

    def test_no_basic_method_names(self):
        """
        Test that legacy 'basic' method names have been removed.
        """
        from bhlff.core.fft.bvp_basic.bvp_basic_core import BVPCoreSolver

        methods = [m for m in dir(BVPCoreSolver) if not m.startswith("_")]
        # Allow only the renamed legacy compatibility method
        disallowed = [m for m in methods if "solve_envelope_basic" in m]
        assert len(disallowed) == 0, f"Found legacy basic methods: {disallowed}"

    def test_legacy_files_still_exist_for_compatibility(self):
        """
        Test that legacy files with re-export still exist for backward compatibility.
        """
        # These files should still exist as they provide backward compatibility
        legacy_files = [
            "bhlff/core/bvp/bvp_envelope_equation_7d.py",
            "bhlff/core/bvp/bvp_postulates_7d.py",
        ]

        for file_path in legacy_files:
            assert os.path.exists(
                file_path
            ), f"Legacy compatibility file {file_path} should exist"

    def test_legacy_files_contain_deprecation_warnings(self):
        """
        Test that legacy files contain proper deprecation warnings.
        """
        legacy_files = [
            "bhlff/core/bvp/bvp_envelope_equation_7d.py",
            "bhlff/core/bvp/bvp_postulates_7d.py",
        ]

        for file_path in legacy_files:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                assert (
                    "DEPRECATED" in content
                ), f"File {file_path} should contain deprecation warning"
                assert (
                    "Legacy" in content
                ), f"File {file_path} should be marked as legacy"
