"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for absence of classical mass terms in 7D BVP theory implementation.

This module tests that no classical mass terms (k₀²|a|², λ⟨a,a⟩, mass attributes)
are present in the codebase, ensuring compliance with 7D BVP theory.
"""

import pytest
import numpy as np
import ast
import os
from pathlib import Path


class TestNoMassTerms:
    """
    Test class for verifying absence of classical mass terms.
    
    Physical Meaning:
        Ensures that the codebase follows 7D BVP theory principles
        by not using classical mass terms that contradict the theory.
    """
    
    def test_no_k0_mass_terms_in_energy_computer(self):
        """
        Test that energy_computer.py does not contain k₀²|a|² mass terms.
        
        Physical Meaning:
            The 7D BVP theory explicitly states "no explicit mass term",
            so k₀²|a|² terms should not be present in energy calculations.
        """
        energy_file = Path("bhlff/core/bvp/postulates/power_balance/energy_computer.py")
        
        if not energy_file.exists():
            pytest.skip("Energy computer file not found")
        
        with open(energy_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for k0**2 * |a|**2 patterns
        assert "k0**2 * np.abs(a) ** 2" not in content, "Found k₀²|a|² mass term in energy computer"
        assert "k0**2 * np.abs(a)**2" not in content, "Found k₀²|a|² mass term in energy computer"
        assert "(mass term)" not in content, "Found mass term comment in energy computer"
    
    def test_no_lambda_mass_terms_in_abstract_solver(self):
        """
        Test that abstract_solver.py uses gradient-based energy instead of λ⟨a,a⟩.
        
        Physical Meaning:
            Classical potential energy λ⟨a,a⟩ should be replaced with
            gradient-based energy λ|∇a|² according to 7D BVP theory.
        """
        solver_file = Path("bhlff/solvers/base/abstract_solver.py")
        
        if not solver_file.exists():
            pytest.skip("Abstract solver file not found")
        
        with open(solver_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check that λ⟨a,a⟩ is replaced with gradient energy
        # Allow λ⟨a,a⟩ in comments about removal, but not in actual code
        assert "λ⟨a,a⟩" not in content.replace("No mass term λ⟨a,a⟩", ""), "Found classical λ⟨a,a⟩ mass term"
        assert "lambda_param * np.sum(field**2)" not in content, "Found classical mass term"
        assert "λ|∇a|²" in content, "Missing gradient-based energy term"
        assert "gradient_energy" in content, "Missing gradient energy computation"
    
    def test_no_mass_attribute_in_particle(self):
        """
        Test that Particle class does not have mass attribute.
        
        Physical Meaning:
            In 7D BVP theory, particles are characterized by phase properties
            and topological charge, not classical mass.
        """
        particle_file = Path("bhlff/models/level_f/multi_particle.py")
        
        if not particle_file.exists():
            pytest.skip("Multi-particle file not found")
        
        with open(particle_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check that mass attribute is removed
        assert "mass: float = 1.0" not in content, "Found mass attribute in Particle class"
        assert "mass: float" not in content, "Found mass attribute in Particle class"
        assert "No mass attribute" in content, "Missing comment about removed mass attribute"
    
    def test_no_phase_mass_in_cosmology(self):
        """
        Test that cosmology models do not use phase_mass.
        
        Physical Meaning:
            Phase mass is a classical concept that contradicts 7D BVP theory.
        """
        cosmology_file = Path("bhlff/models/level_g/cosmology.py")
        
        if not cosmology_file.exists():
            pytest.skip("Cosmology file not found")
        
        with open(cosmology_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check that phase_mass is removed
        assert "self.phase_mass" not in content, "Found phase_mass in cosmology"
        assert "phase_mass = " not in content, "Found phase_mass assignment in cosmology"
        assert "No phase_mass" in content, "Missing comment about removed phase_mass"
    
    def test_no_phase_mass_in_evolution(self):
        """
        Test that evolution models do not use phase_mass.
        
        Physical Meaning:
            Phase mass is a classical concept that contradicts 7D BVP theory.
        """
        evolution_file = Path("bhlff/models/level_g/evolution.py")
        
        if not evolution_file.exists():
            pytest.skip("Evolution file not found")
        
        with open(evolution_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check that phase_mass is removed
        assert "self.phase_mass" not in content, "Found phase_mass in evolution"
        assert "phase_mass = " not in content, "Found phase_mass assignment in evolution"
        assert "No phase_mass" in content, "Missing comment about removed phase_mass"
    
    def test_energy_functional_uses_derivatives(self):
        """
        Test that energy functionals use derivative terms instead of mass terms.
        
        Physical Meaning:
            Energy in 7D BVP theory should be based on field derivatives,
            not on field values themselves (mass terms).
        """
        energy_file = Path("bhlff/core/bvp/postulates/power_balance/energy_computer.py")
        
        if not energy_file.exists():
            pytest.skip("Energy computer file not found")
        
        with open(energy_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check that energy uses derivative terms
        assert "grad_spatial_sq" in content, "Missing spatial gradient terms"
        assert "grad_phase_sq" in content, "Missing phase gradient terms"
        assert "laplacian_sq" in content, "Missing Laplacian terms"
        assert "grad_total_sq" in content, "Missing total gradient terms"
        
        # Check that no mass terms are present
        assert "k0**2" not in content, "Found k₀² mass term"
        # Allow "mass term" in comments about removal, but not in actual code
        content_without_comments = content.replace("No mass term k₀²|a|² - removed according to 7D BVP theory", "").replace("(no mass term)", "")
        assert "mass term" not in content_without_comments, "Found mass term comment"
    
    def test_energy_computation_uses_gradients(self):
        """
        Test that energy computation in abstract solver uses gradients.
        
        Physical Meaning:
            Energy computation should use field gradients |∇a|²
            instead of field values |a|² according to 7D BVP theory.
        """
        solver_file = Path("bhlff/solvers/base/abstract_solver.py")
        
        if not solver_file.exists():
            pytest.skip("Abstract solver file not found")
        
        with open(solver_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check that gradient-based energy is used
        assert "field_gradient" in content, "Missing field gradient computation"
        assert "gradient_energy" in content, "Missing gradient energy computation"
        assert "λ|∇a|²" in content, "Missing gradient-based energy formula"
        
        # Check that classical mass terms are not used
        assert "np.sum(field**2)" not in content, "Found classical mass term"
        # Allow λ⟨a,a⟩ in comments about removal, but not in actual code
        assert "λ⟨a,a⟩" not in content.replace("No mass term λ⟨a,a⟩", ""), "Found classical mass term notation"
    
    def test_no_mass_terms_in_entire_codebase(self):
        """
        Test that no mass terms exist anywhere in the codebase.
        
        Physical Meaning:
            Comprehensive check that no classical mass terms
            remain anywhere in the codebase.
        """
        # Patterns to search for
        mass_patterns = [
            "k0**2 * np.abs(a) ** 2",
            "k0**2 * np.abs(a)**2", 
            "lambda_param * np.sum(field**2)",
            "mass: float = 1.0",
            "self.phase_mass",
            "mass_matrix",
            "defect_mass",
            "particle.mass"
        ]
        
        # Search in Python files
        python_files = []
        for root, dirs, files in os.walk("bhlff"):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        violations = []
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern in mass_patterns:
                    if pattern in content:
                        violations.append(f"{file_path}: {pattern}")
            except Exception:
                continue  # Skip files that can't be read
        
        assert len(violations) == 0, f"Found mass terms in codebase: {violations}"
    
    def test_energy_density_no_mass_terms(self):
        """
        Test that energy density computation does not contain mass terms.
        
        Physical Meaning:
            Energy density should be computed using only derivative terms
            according to 7D BVP theory principles.
        """
        energy_file = Path("bhlff/core/bvp/postulates/power_balance/energy_computer.py")
        
        if not energy_file.exists():
            pytest.skip("Energy computer file not found")
        
        with open(energy_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the file to check energy density computation
        tree = ast.parse(content)
        
        # Find function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name == "compute_energy_density":
                    # Check function body for mass terms
                    function_source = ast.get_source_segment(content, node)
                    assert function_source is not None
                    
                    # Check for mass terms in the function
                    assert "k0**2" not in function_source, "Found k₀² mass term in energy density"
                    assert "np.abs(a) ** 2" not in function_source, "Found |a|² mass term"
                    assert "mass term" not in function_source, "Found mass term comment"
                    
                    # Check that derivative terms are present
                    assert "grad_spatial_sq" in function_source, "Missing spatial gradient"
                    assert "grad_phase_sq" in function_source, "Missing phase gradient"
                    assert "laplacian_sq" in function_source, "Missing Laplacian term"
                    break
