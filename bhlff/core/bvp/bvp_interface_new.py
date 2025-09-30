"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

BVP Interface implementation according to step 00 specification.

This module provides backward compatibility for the BVP interface,
redirecting to the new modular interface package.

Theoretical Background:
    The BVP interface serves as the connection point between the BVP
    envelope and other system components. This module provides
    backward compatibility while the new modular interface package
    is used internally.

Example:
    >>> interface = BVPInterface(bvp_core)
    >>> tail_data = interface.interface_with_tail(envelope)
    >>> transition_data = interface.interface_with_transition_zone(envelope)
    >>> core_data = interface.interface_with_core(envelope)
"""

from .interface import BVPInterface as ModularBVPInterface

# Create backward compatibility alias
BVPInterface = ModularBVPInterface
