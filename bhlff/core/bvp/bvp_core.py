"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Legacy BVP core module - DEPRECATED.

This module is deprecated. Use the new modular structure:
- bhlff.core.bvp.bvp_core.BVPCore
- Individual modules in bhlff.core.bvp.bvp_core.*

The BVP core has been refactored into separate modules following
the 1 class = 1 file principle and size limits.

Physical Meaning:
    This legacy module contained the complete BVP core implementation
    in a single file, which violated project standards. The implementation
    has been moved to individual modules for better maintainability.

Example:
    # OLD (deprecated):
    # from bhlff.core.bvp.bvp_core import BVPCore
    
    # NEW (recommended):
    from bhlff.core.bvp.bvp_core import BVPCore
"""

# Import from the new modular structure
from .bvp_core import BVPCore
from .bvp_core import BVPCoreOperations, BVPCore7DInterface

# Re-export for backward compatibility
__all__ = ["BVPCore", "BVPCoreOperations", "BVPCore7DInterface"]
