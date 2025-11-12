"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test script to verify GPU logging and block processing usage.
"""

import logging
import numpy as np
from bhlff.core.domain import Domain
from bhlff.core.domain.parameters import Parameters
from bhlff.core.fft.fft_solver_7d_basic import FFTSolver7DBasic

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    logger.info("Starting GPU logging test")
    
    # Create small domain for testing
    domain = Domain(dimensions=7, N=64, N_phi=4, N_t=4, L=8*3.14159)
    params = Parameters(mu=1.0, beta=1.0, lambda_param=0.0)
    
    logger.info(f"Domain shape: {domain.shape}")
    
    # Create solver
    solver = FFTSolver7DBasic(domain, params)
    logger.info("Solver created")
    
    # Create small source field
    source = np.random.randn(*domain.shape).astype(np.complex128) * 0.1
    source_size_mb = source.nbytes / (1024**2)
    logger.info(f"Source created: shape={source.shape}, size={source_size_mb:.2f}MB")
    
    # Solve
    logger.info("Starting solve_stationary...")
    solution = solver.solve_stationary(source)
    logger.info(f"Solution shape: {solution.shape if hasattr(solution, 'shape') else 'N/A'}")
    logger.info("Test completed successfully")

if __name__ == "__main__":
    main()

