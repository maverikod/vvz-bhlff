"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Debug script for radial profile computation with detailed logging.
Tests GPU utilization and block processing.
"""

import sys
import logging
import numpy as np

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_radial_profile.log')
    ]
)

logger = logging.getLogger(__name__)

# Check CUDA availability
logger.info("=" * 80)
logger.info("CHECKING CUDA AVAILABILITY")
logger.info("=" * 80)

try:
    import cupy as cp
    CUDA_AVAILABLE = cp.cuda.is_available()
    logger.info(f"CuPy imported: True")
    logger.info(f"CUDA available: {CUDA_AVAILABLE}")
    
    if CUDA_AVAILABLE:
        try:
            mem_info = cp.cuda.runtime.memGetInfo()
            free_mem = mem_info[0] / 1e9
            total_mem = mem_info[1] / 1e9
            used_mem = (mem_info[1] - mem_info[0]) / 1e9
            logger.info(f"GPU Memory: Free={free_mem:.2f}GB, Total={total_mem:.2f}GB, Used={used_mem:.2f}GB")
            
            device = cp.cuda.Device()
            logger.info(f"GPU Device: {device.id}")
            logger.info(f"GPU Compute Capability: {device.compute_capability}")
        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")
    else:
        logger.warning("CUDA is not available!")
except ImportError as e:
    logger.error(f"Failed to import cupy: {e}")
    CUDA_AVAILABLE = False
    cp = None

logger.info("=" * 80)

# Import project modules
logger.info("IMPORTING PROJECT MODULES")
logger.info("=" * 80)

try:
    from bhlff.core.base.domain import Domain
    from bhlff.core.base.parameters import Parameters
    from bhlff.core.fft.fft_solver_7d_basic import FFTSolver7DBasic
    from bhlff.models.level_b.stepwise.radial_profile import RadialProfileComputer
    from bhlff.core.arrays.field_array import FieldArray
    logger.info("All modules imported successfully")
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    sys.exit(1)

logger.info("=" * 80)

# Create test domain
logger.info("CREATING TEST DOMAIN")
logger.info("=" * 80)

domain = Domain(
    L=8 * np.pi,
    N=128,
    N_phi=8,
    N_t=8,
    T=1.0,
    dimensions=7
)

logger.info(f"Domain shape: {domain.shape}")
logger.info(f"Domain size: {np.prod(domain.shape)} elements")
logger.info(f"Domain memory (complex128): {np.prod(domain.shape) * 16 / 1e9:.3f}GB")

params = Parameters(mu=1.0, beta=1.0, lambda_param=0.0)
logger.info(f"Parameters: mu={params.mu}, beta={params.beta}, lambda={params.lambda_param}")

logger.info("=" * 80)

# Create source field
logger.info("CREATING SOURCE FIELD")
logger.info("=" * 80)

def create_neutralized_gaussian(domain, sigma_cells=2.0):
    """Create neutralized Gaussian source."""
    dx = domain.L / domain.N
    x = np.linspace(0, domain.L, domain.N, endpoint=False)
    y = np.linspace(0, domain.L, domain.N, endpoint=False)
    z = np.linspace(0, domain.L, domain.N, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    
    center = [domain.L / 2, domain.L / 2, domain.L / 2]
    sigma = sigma_cells * dx
    
    distances_sq = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
    exponent = -distances_sq / (2 * sigma ** 2)
    exponent = np.clip(exponent, -700, None)
    g = np.exp(exponent)
    
    g_mean = np.mean(g)
    s = g - g_mean
    
    # Create 7D array using FieldArray
    s_7d_field = FieldArray(shape=domain.shape, dtype=np.complex128)
    s_3d = s[:, :, :, None, None, None, None]
    s_3d_broadcast = np.broadcast_to(s_3d, domain.shape)
    s_7d_field[:] = s_3d_broadcast[:]
    
    logger.info(f"Source field shape: {s_7d_field.array.shape}")
    logger.info(f"Source field type: {type(s_7d_field.array)}")
    logger.info(f"Source field is memmap: {isinstance(s_7d_field.array, np.memmap)}")
    logger.info(f"Source field memory: {s_7d_field.array.nbytes / 1e9:.3f}GB")
    
    return s_7d_field.array

source = create_neutralized_gaussian(domain, sigma_cells=2.0)
logger.info("=" * 80)

# Solve
logger.info("SOLVING PHASE FIELD EQUATION")
logger.info("=" * 80)

try:
    solver = FFTSolver7DBasic(domain, params)
    logger.info("Solver created successfully")
    
    solution = solver.solve_stationary(source)
    logger.info(f"Solution type: {type(solution)}")
    
    if isinstance(solution, FieldArray):
        solution_array = solution.array
        logger.info("Solution is FieldArray")
    else:
        solution_array = solution
        logger.info("Solution is numpy array")
    
    logger.info(f"Solution shape: {solution_array.shape}")
    logger.info(f"Solution type: {type(solution_array)}")
    logger.info(f"Solution is memmap: {isinstance(solution_array, np.memmap)}")
    logger.info(f"Solution memory: {solution_array.nbytes / 1e9:.3f}GB")
    
    if CUDA_AVAILABLE:
        mem_info = cp.cuda.runtime.memGetInfo()
        free_mem = mem_info[0] / 1e9
        logger.info(f"GPU Memory after solve: Free={free_mem:.2f}GB")
except Exception as e:
    logger.error(f"Failed to solve: {e}", exc_info=True)
    sys.exit(1)

logger.info("=" * 80)

# Compute radial profile
logger.info("COMPUTING RADIAL PROFILE")
logger.info("=" * 80)

center = [domain.N // 2] * 3
logger.info(f"Center: {center}")

try:
    profiler = RadialProfileComputer(use_cuda=True, gpu_memory_ratio=0.8)
    logger.info(f"RadialProfileComputer created")
    logger.info(f"profiler.use_cuda: {profiler.use_cuda}")
    logger.info(f"profiler.gpu_memory_ratio: {profiler.gpu_memory_ratio}")
    
    if CUDA_AVAILABLE:
        mem_info_before = cp.cuda.runtime.memGetInfo()
        free_mem_before = mem_info_before[0] / 1e9
        logger.info(f"GPU Memory before compute: Free={free_mem_before:.2f}GB")
    
    logger.info("Calling profiler.compute()...")
    profile = profiler.compute(solution_array, center)
    
    logger.info(f"Profile computed successfully")
    logger.info(f"Profile keys: {profile.keys()}")
    logger.info(f"Profile r shape: {profile['r'].shape}")
    logger.info(f"Profile A shape: {profile['A'].shape}")
    logger.info(f"Profile r range: [{profile['r'].min():.4f}, {profile['r'].max():.4f}]")
    logger.info(f"Profile A range: [{profile['A'].min():.4f}, {profile['A'].max():.4f}]")
    
    if CUDA_AVAILABLE:
        mem_info_after = cp.cuda.runtime.memGetInfo()
        free_mem_after = mem_info_after[0] / 1e9
        used_mem = (mem_info_before[0] - mem_info_after[0]) / 1e9
        logger.info(f"GPU Memory after compute: Free={free_mem_after:.2f}GB")
        logger.info(f"GPU Memory used during compute: {used_mem:.2f}GB")
        logger.info(f"GPU Memory utilization: {used_mem / mem_info_before[1] * 1e9 * 100:.2f}%")
        
except Exception as e:
    logger.error(f"Failed to compute radial profile: {e}", exc_info=True)
    sys.exit(1)

logger.info("=" * 80)
logger.info("TEST COMPLETED SUCCESSFULLY")
logger.info("=" * 80)


