"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP integration with levels A-G implementation.

This module provides the main integration interface between the BVP framework
and all levels A-G of the 7D phase field theory, ensuring that BVP
serves as the central backbone for all system components.

Physical Meaning:
    BVP serves as the central framework where all observed "modes"
    are envelope modulations and beatings of the Base High-Frequency Field.
    This module provides the unified interface for levels A-G to interact with BVP.

Mathematical Foundation:
    Each level provides specific mathematical operations that work
    with BVP envelope data, transforming it according to level-specific
    requirements while maintaining BVP framework compliance.

Example:
    >>> integration = BVPLevelIntegration(bvp_core)
    >>> level_a_data = integration.get_level_a_data(envelope)
    >>> level_b_data = integration.get_level_b_data(envelope)
"""

import numpy as np
from typing import Dict, Any

from .bvp_core import BVPCore
from .bvp_level_interfaces_abc import LevelAInterface, LevelBInterface, LevelCInterface
from .bvp_level_interfaces_def import LevelDInterface, LevelEInterface, LevelFInterface
from .bvp_level_interfaces_g import LevelGInterface


class BVPLevelIntegration:
    """
    Main BVP level integration interface.

    Physical Meaning:
        Provides unified interface for integrating BVP with all levels A-G,
        ensuring BVP serves as the central backbone for the entire system.
    """

    def __init__(self, bvp_core: BVPCore):
        """
        Initialize BVP level integration.

        Physical Meaning:
            Sets up integration interfaces for all levels A-G with
            the BVP core framework.
        """
        self.bvp_core = bvp_core
        self.constants = bvp_core.constants

        # Initialize level interfaces
        self.level_a = LevelAInterface(bvp_core)
        self.level_b = LevelBInterface(bvp_core)
        self.level_c = LevelCInterface(bvp_core)
        self.level_d = LevelDInterface(bvp_core)
        self.level_e = LevelEInterface(bvp_core)
        self.level_f = LevelFInterface(bvp_core)
        self.level_g = LevelGInterface(bvp_core)

    def get_level_a_data(self, envelope: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Get Level A data from BVP envelope."""
        return self.level_a.process_bvp_data(envelope, **kwargs)

    def get_level_b_data(self, envelope: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Get Level B data from BVP envelope."""
        return self.level_b.process_bvp_data(envelope, **kwargs)

    def get_level_c_data(self, envelope: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Get Level C data from BVP envelope."""
        return self.level_c.process_bvp_data(envelope, **kwargs)

    def get_level_d_data(self, envelope: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Get Level D data from BVP envelope."""
        return self.level_d.process_bvp_data(envelope, **kwargs)

    def get_level_e_data(self, envelope: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Get Level E data from BVP envelope."""
        return self.level_e.process_bvp_data(envelope, **kwargs)

    def get_level_f_data(self, envelope: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Get Level F data from BVP envelope."""
        return self.level_f.process_bvp_data(envelope, **kwargs)

    def get_level_g_data(self, envelope: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Get Level G data from BVP envelope."""
        return self.level_g.process_bvp_data(envelope, **kwargs)

    def get_all_levels_data(self, envelope: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Get data for all levels A-G from BVP envelope."""
        return {
            "level_a": self.get_level_a_data(envelope, **kwargs),
            "level_b": self.get_level_b_data(envelope, **kwargs),
            "level_c": self.get_level_c_data(envelope, **kwargs),
            "level_d": self.get_level_d_data(envelope, **kwargs),
            "level_e": self.get_level_e_data(envelope, **kwargs),
            "level_f": self.get_level_f_data(envelope, **kwargs),
            "level_g": self.get_level_g_data(envelope, **kwargs),
        }

    def validate_bvp_integration(self, envelope: np.ndarray) -> bool:
        """
        Validate BVP integration with all levels.

        Physical Meaning:
            Ensures that BVP envelope data is properly integrated
            with all levels A-G and maintains framework compliance.
        """
        try:
            # Test all level interfaces
            level_a_data = self.get_level_a_data(envelope)
            level_b_data = self.get_level_b_data(envelope)
            level_c_data = self.get_level_c_data(envelope)
            level_d_data = self.get_level_d_data(envelope)
            level_e_data = self.get_level_e_data(envelope)
            level_f_data = self.get_level_f_data(envelope)
            level_g_data = self.get_level_g_data(envelope)

            # Check that all levels return valid data
            return (
                level_a_data is not None
                and level_b_data is not None
                and level_c_data is not None
                and level_d_data is not None
                and level_e_data is not None
                and level_f_data is not None
                and level_g_data is not None
            )
        except Exception:
            return False
