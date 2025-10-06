"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical validation tests for BVP postulates.

This module provides comprehensive physical validation tests for all 9 BVP
postulates, ensuring they correctly implement the theoretical foundations
of the Base High-Frequency Field in 7D space-time.

Physical Meaning:
    Tests validate that each BVP postulate correctly implements its
    specific physical property:
    1. Carrier Primacy - high-frequency carrier dominance
    2. Scale Separation - separation between carrier and envelope
    3. BVP Rigidity - field stability and coherence
    4. U(1)³ Phase Structure - phase coherence and topology
    5. Quenches - phase transition dynamics
    6. Tail Resonatorness - resonance properties
    7. Transition Zone - nonlinear interface behavior
    8. Core Renormalization - renormalization effects
    9. Power Balance - energy conservation

Mathematical Foundation:
    Each postulate implements specific mathematical conditions that
    must be satisfied for physical consistency of the BVP theory.

Example:
    >>> pytest tests/unit/test_core/test_bvp_postulates_physics.py -v
"""

# Import all individual test modules
from test_carrier_primacy_postulate_physics import TestCarrierPrimacyPostulatePhysics
from test_scale_separation_postulate_physics import TestScaleSeparationPostulatePhysics
from test_bvp_rigidity_postulate_physics import TestBVPRigidityPostulatePhysics
from test_u1_phase_structure_postulate_physics import (
    TestU1PhaseStructurePostulatePhysics,
)
from test_quenches_postulate_physics import TestQuenchesPostulatePhysics
from test_tail_resonatorness_postulate_physics import (
    TestTailResonatornessPostulatePhysics,
)
from test_transition_zone_postulate_physics import TestTransitionZonePostulatePhysics
from test_core_renormalization_postulate_physics import (
    TestCoreRenormalizationPostulatePhysics,
)
from test_power_balance_postulate_physics import TestPowerBalancePostulatePhysics
from test_bvp_postulates_integration_physics import TestBVPPostulatesIntegrationPhysics
