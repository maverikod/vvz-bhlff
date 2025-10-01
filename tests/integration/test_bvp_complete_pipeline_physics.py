"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Complete BVP pipeline physical validation tests.

This module provides comprehensive integration tests for the complete
BVP pipeline, ensuring end-to-end physical consistency and theoretical
correctness of the 7D BVP theory implementation.

Physical Meaning:
    Tests validate the complete BVP pipeline:
    - 7D space-time structure M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ
    - BVP envelope equation solution
    - All 9 BVP postulates validation
    - Energy conservation throughout pipeline
    - Physical consistency across all components
    - Theoretical correctness of results

Mathematical Foundation:
    Validates the complete 7D BVP theory:
    - Envelope equation: ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    - U(1)³ phase structure: a = |a|e^(iφ₁)e^(iφ₂)e^(iφ₃)
    - Energy conservation: ∂E/∂t + ∇·S = 0
    - All 9 BVP postulates simultaneously

Example:
    >>> pytest tests/integration/test_bvp_complete_pipeline_physics.py -v
"""

# Import all individual test modules
from test_bvp_core_pipeline_physics import TestBVPCorePipelinePhysics
from test_bvp_interface_pipeline_physics import TestBVPInterfacePipelinePhysics
from test_bvp_quench_dynamics_physics import TestBVPQuenchDynamicsPhysics
from test_bvp_impedance_calculation_physics import TestBVPImpedanceCalculationPhysics
from test_bvp_phase_vector_physics import TestBVPPhaseVectorPhysics
