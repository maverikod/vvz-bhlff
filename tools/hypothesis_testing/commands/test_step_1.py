"""
Test Step 1: Power Law Analyzer - Stepwise Structure.
"""

import numpy as np
from typing import Dict, Any
from .base import BaseCommand
from bhlff.models.level_b.power_law_analyzer import LevelBPowerLawAnalyzer
from bhlff.models.level_b.node_analyzer import LevelBNodeAnalyzer


class TestStep1Command(BaseCommand):
    """Test Step 1: Power Law Analyzer - Stepwise Structure."""
    
    def execute(self) -> Dict[str, Any]:
        """Execute Step 1 test."""
        self.logger.info("Testing Step 1: Power Law Analyzer - Stepwise Structure...")
        
        try:
            # Create minimal domain and substrate
            domain = self.create_minimal_domain()
            substrate = self.create_test_substrate(domain)
            center = [domain.N//2, domain.N//2, domain.N//2]
            
            # Test Power Law Analyzer on substrate
            self.logger.info("Testing Power Law Analyzer on substrate...")
            power_analyzer = LevelBPowerLawAnalyzer(use_cuda=False)
            power_result = power_analyzer.analyze_stepwise_tail(substrate, 1.5, center)
            
            power_success = power_result.get("stepwise_structure", False)
            print(f"Power Law stepwise structure: {power_success}")
            print(f"Power Law result: {power_result}")
            
            # Test Node Analyzer on substrate
            print("Testing Node Analyzer on substrate...")
            node_analyzer = LevelBNodeAnalyzer(use_cuda=False)
            node_result = node_analyzer.check_stepwise_structure(substrate, center)
            
            node_success = node_result.get("stepwise_structure", False)
            print(f"Node stepwise structure: {node_success}")
            print(f"Node result: {node_result}")
            
            # Overall success
            success = power_success and node_success
            self.logger.info(f"✅ Step 1: Stepwise structure = {success}")
            
            return {
                "step": 1,
                "name": "Power Law Analyzer - Stepwise Structure",
                "success": success,
                "details": {
                    "power_law_stepwise": power_success,
                    "node_stepwise": node_success,
                    "power_law_layers": len(power_result.get("layers", [])),
                    "power_law_q_factors": power_result.get("q_factors", []),
                    "power_law_quantization": power_result.get("quantization", False),
                    "node_level_quantization": node_result.get("level_quantization", False),
                    "node_discrete_layers": node_result.get("discrete_layers", False),
                    "substrate_shape": substrate.shape,
                    "substrate_range": f"{np.min(substrate):.3f} - {np.max(substrate):.3f}"
                }
            }
        except Exception as e:
            self.logger.error(f"❌ Step 1 failed: {e}")
            return {
                "step": 1, 
                "name": "Power Law Analyzer", 
                "success": False, 
                "error": str(e)
            }
