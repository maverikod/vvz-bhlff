"""
Base command class for hypothesis testing.
"""

import abc
import logging
from typing import Dict, Any, List
import numpy as np
from bhlff.core.domain import Domain


class BaseCommand(abc.ABC):
    """Base class for all hypothesis testing commands."""
    
    def __init__(self, verbose: bool = True):
        """Initialize command."""
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def create_minimal_domain(self) -> Domain:
        """Create minimal 7D domain for testing."""
        return Domain(L=1.0, N=4, dimensions=7, N_phi=2, N_t=4, T=1.0)
    
    def create_test_field(self, domain: Domain) -> np.ndarray:
        """Create test field with standing wave patterns."""
        field = np.random.randn(*domain.shape) + 1j * np.random.randn(*domain.shape)
        
        # Add standing wave patterns (Friedel-like waves)
        for i in range(domain.N):
            for j in range(domain.N):
                for k in range(domain.N):
                    for phi1 in range(domain.N_phi):
                        for phi2 in range(domain.N_phi):
                            for phi3 in range(domain.N_phi):
                                for t in range(domain.N_t):
                                    # Standing waves in spatial coordinates
                                    spatial_wave = (np.sin(2*np.pi*i/domain.N) * 
                                                  np.cos(2*np.pi*j/domain.N) * 
                                                  np.sin(2*np.pi*k/domain.N))
                                    # Standing waves in phase coordinates
                                    phase_wave = (np.cos(2*np.pi*phi1/domain.N_phi) * 
                                                np.sin(2*np.pi*phi2/domain.N_phi) * 
                                                np.cos(2*np.pi*phi3/domain.N_phi))
                                    # Temporal modulation
                                    temporal_wave = np.cos(2*np.pi*t/domain.N_t)
                                    
                                    field[i,j,k,phi1,phi2,phi3,t] = (spatial_wave * 
                                                                     phase_wave * 
                                                                     temporal_wave)
        
        # Add high-amplitude regions for testing
        field[2:3, 2:3, 2:3, 1:2, 1:2, 1:2, 2:3] *= 10.0
        
        # Ensure we have high amplitude regions
        field[1:2, 1:2, 1:2, 0:1, 0:1, 0:1, 1:2] = 5.0 + 5.0j
        
        return field
    
    @abc.abstractmethod
    def execute(self) -> Dict[str, Any]:
        """Execute the command."""
        pass
    
    def print_result(self, result: Dict[str, Any]):
        """Print command result."""
        status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
        print(f"{result['name']} - {status}")
        
        if result.get("success", False) and "details" in result:
            details = result["details"]
            for key, value in details.items():
                print(f"  {key}: {value}")
        
        if "error" in result:
            print(f"  Error: {result['error']}")
