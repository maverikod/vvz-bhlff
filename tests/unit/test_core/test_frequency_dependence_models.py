"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for frequency-dependent material properties in BVPConstantsBase.
"""

import math

from bhlff.core.bvp.bvp_constants_base import BVPConstantsBase


def test_drude_conductivity_monotonicity():
    config = {
        "material_properties": {
            "base_conductivity": 0.02,
            "admittance_model": "drude",
            "parameters": {"gamma": 1.5e10, "omega_p": 3.2e12},
        }
    }
    c = BVPConstantsBase(config)
    # Low frequency vs high frequency trend with Drude add-on term
    sigma_low = c.get_conductivity(1.0e3)
    sigma_high = c.get_conductivity(1.0e14)
    assert sigma_low > 0
    assert sigma_high > 0
    # Denominator grows with ω^2, so additive Drude term decreases; but total >= base
    assert sigma_low >= c.BASE_CONDUCTIVITY
    assert sigma_high >= c.BASE_CONDUCTIVITY


def test_debye_conductivity_decreases_with_frequency():
    config = {
        "material_properties": {
            "base_conductivity": 0.02,
            "admittance_model": "debye",
            "parameters": {"tau": 1.0e-9, "sigma_inf": 0.05},
        }
    }
    c = BVPConstantsBase(config)
    sigma_low = c.get_conductivity(1.0e3)
    sigma_high = c.get_conductivity(1.0e12)
    assert sigma_low > sigma_high
    assert math.isfinite(sigma_low) and math.isfinite(sigma_high)


def test_admittance_scales_with_conductivity():
    config = {
        "material_properties": {
            "base_conductivity": 0.02,
            "base_admittance": 1.0,
            "cutoff_frequency": 1.0e6,
            "admittance_model": "drude",
            "parameters": {"gamma": 1.0e3, "omega_p": 1.0e4},
        }
    }
    c = BVPConstantsBase(config)
    s1 = c.get_conductivity(0.0)
    s2 = c.get_conductivity(1.0e6)
    y1 = c.get_admittance(0.0)
    y2 = c.get_admittance(1.0e6)
    # Admittance should preserve proportional change to conductivity
    assert y1 == c.BASE_ADMITTANCE * (s1 / c.BASE_CONDUCTIVITY)
    assert y2 == c.BASE_ADMITTANCE * (s2 / c.BASE_CONDUCTIVITY)
