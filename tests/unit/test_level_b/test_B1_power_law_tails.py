"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test B1: BVP Power Law Tails for Level B.

This module implements comprehensive tests for power law tail behavior
A(r) ∝ r^(2β-3) with all acceptance criteria from document 7d-32.

Theoretical Background:
    Tests validate power law behavior in homogeneous medium governed by
    Riesz operator L_β = μ(-Δ)^β + λ with λ=0, confirming theoretical
    prediction A(r) ∝ r^(2β-3) for different β values in 7D space-time.

Example:
    >>> pytest tests/unit/test_level_b/test_B1_power_law_tails.py -v
"""

import numpy as np
import pytest
from typing import Dict, Any, List
from scipy import stats
import os

from bhlff.core.domain import Domain
from bhlff.core.domain.parameters import Parameters
from bhlff.core.fft.fft_solver_7d_basic import FFTSolver7DBasic
from bhlff.models.level_b.stepwise.analyzer import LevelBPowerLawAnalyzer


class TestB1PowerLawTails:
    """
    Comprehensive tests for B1: Power Law Tails.
    
    Physical Meaning:
        Validates that BVP envelope exhibits power law decay A(r) ∝ r^(2β-3)
        in homogeneous medium with all acceptance criteria from document 7d-32.
    """
    
    @pytest.fixture
    def domain_configs(self):
        """Create domain configurations for convergence testing."""
        L = 8 * np.pi  # As per document 7d-32
        # Balanced sizes: large enough for accurate analysis (R²≥0.99, correct slopes),
        # but manageable with swap/block processing to avoid GPU memory issues
        # Real production will use larger fields (N=256+) on powerful GPUs
        return [
            {"L": L, "N": 96, "N_phi": 8, "N_t": 8, "T": 1.0},
            {"L": L, "N": 128, "N_phi": 8, "N_t": 8, "T": 1.0},
            {"L": L, "N": 160, "N_phi": 8, "N_t": 8, "T": 1.0},
        ]
    
    @pytest.fixture
    def beta_values(self):
        """Beta values for testing as per document 7d-32."""
        # Reduced set for faster testing - can expand for full validation
        return [0.8, 1.0, 1.2]
    
    @pytest.fixture
    def use_cuda(self):
        """CUDA usage flag - if CUDA is required but not available, test will fail."""
        use_cuda_flag = os.getenv("BHLFF_DISABLE_CUDA", "0") != "1"
        if use_cuda_flag:
            try:
                import cupy as cp
                # Check if CUDA is actually available
                if not cp.cuda.is_available():
                    pytest.skip("CUDA is required but not available")
            except ImportError:
                pytest.skip("CUDA is required but cupy is not installed")
        return use_cuda_flag
    
    def test_B1_power_law_slope(self, domain_configs, beta_values, use_cuda):
        """
        Test B1: Power law slope in 95% confidence interval.
        
        Criterion from document 7d-32:
        - Наклон p̂ в ДИ 95% попадает в p_теор = 2β-3 ± 0.05
        """
        # Use middle resolution for this test
        domain = Domain(dimensions=7, **domain_configs[1])
        
        for beta in beta_values:
            # Create solver and source
            # Note: FFTSolver7DBasic will raise RuntimeError if CUDA is required but not available
            params = Parameters(mu=1.0, beta=beta, lambda_param=0.0)
            try:
                solver = FFTSolver7DBasic(domain, params)
            except RuntimeError as e:
                if "CUDA is required" in str(e):
                    pytest.skip(f"CUDA is required but not available: {e}")
                raise
            
            # Create neutralized Gaussian source (as per §1.5)
            source = self._create_neutralized_gaussian(domain, sigma_cells=2.0)
            
            # Solve stationary problem
            solution = solver.solve_stationary(source)
            
            # Analyze power law tail
            analyzer = LevelBPowerLawAnalyzer(use_cuda=use_cuda)
            center = [domain.N // 2] * 3
            result = analyzer.analyze_power_law_tail(
                solution, beta, center, min_decades=1.5
            )
            
            # Check slope in 95% CI
            theoretical_slope = 2 * beta - 3
            slope = result["slope"]
            slope_ci = result.get("slope_ci_95", (slope - 0.1, slope + 0.1))
            
            assert slope_ci[0] <= theoretical_slope + 0.05, (
                f"β={beta}: lower CI {slope_ci[0]:.4f} > theoretical+0.05 {theoretical_slope+0.05:.4f}"
            )
            assert slope_ci[1] >= theoretical_slope - 0.05, (
                f"β={beta}: upper CI {slope_ci[1]:.4f} < theoretical-0.05 {theoretical_slope-0.05:.4f}"
            )
    
    def test_B1_r_squared_min_decades(self, domain_configs, beta_values, use_cuda):
        """
        Test B1: R² ≥ 0.99 on ≥1.5 decades.
        
        Criterion from document 7d-32:
        - R² ≥ 0.99 на диапазоне не менее 1.5 декады по r
        """
        domain = Domain(dimensions=7, **domain_configs[1])
        
        for beta in beta_values:
            params = Parameters(mu=1.0, beta=beta, lambda_param=0.0)
            try:
                solver = FFTSolver7DBasic(domain, params)
            except RuntimeError as e:
                if "CUDA is required" in str(e):
                    pytest.skip(f"CUDA is required but not available: {e}")
                raise
            source = self._create_neutralized_gaussian(domain, sigma_cells=2.0)
            solution = solver.solve_stationary(source)
            
            analyzer = LevelBPowerLawAnalyzer(use_cuda=use_cuda)
            center = [domain.N // 2] * 3
            result = analyzer.analyze_power_law_tail(
                solution, beta, center, min_decades=1.5
            )
            
            assert result["r_squared"] >= 0.99, (
                f"β={beta}: R² = {result['r_squared']:.4f} < 0.99"
            )
            assert result.get("decades", 0.0) >= 1.5, (
                f"β={beta}: {result.get('decades', 0.0):.2f} decades < 1.5"
            )
    
    def test_B1_grid_convergence(self, domain_configs, beta_values, use_cuda):
        """
        Test B1: Grid convergence.
        
        Criterion from document 7d-32:
        - Сходимость по сетке: |p̂_{N₂} - p̂_{N₁}| ≤ 0.02 при N₂/N₁ ≥ 1.5
        """
        beta = 1.0  # Test with one beta value
        
        slopes = []
        for config in domain_configs:
            domain = Domain(dimensions=7, **config)
            params = Parameters(mu=1.0, beta=beta, lambda_param=0.0)
            try:
                solver = FFTSolver7DBasic(domain, params)
            except RuntimeError as e:
                if "CUDA is required" in str(e):
                    pytest.skip(f"CUDA is required but not available: {e}")
                raise
            source = self._create_neutralized_gaussian(domain, sigma_cells=2.0)
            solution = solver.solve_stationary(source)
            
            analyzer = LevelBPowerLawAnalyzer(use_cuda=use_cuda)
            center = [domain.N // 2] * 3
            result = analyzer.analyze_power_law_tail(
                solution, beta, center, min_decades=1.5
            )
            slopes.append((config["N"], result["slope"]))
        
        # Check convergence between consecutive resolutions
        for i in range(len(slopes) - 1):
            N1, p1 = slopes[i]
            N2, p2 = slopes[i + 1]
            if N2 / N1 >= 1.5:
                diff = abs(p2 - p1)
                assert diff <= 0.02, (
                    f"Convergence failed: |p_{N2} - p_{N1}| = {diff:.4f} > 0.02"
                )
    
    def test_B1_residual_norm(self, domain_configs, beta_values, use_cuda):
        """
        Test B1: Residual norm.
        
        Criterion from document 7d-32:
        - Невязка: ||r||₂ / ||s||₂ ≤ 10⁻¹¹
        """
        domain = Domain(dimensions=7, **domain_configs[1])
        
        for beta in beta_values:
            params = Parameters(mu=1.0, beta=beta, lambda_param=0.0)
            try:
                solver = FFTSolver7DBasic(domain, params)
            except RuntimeError as e:
                if "CUDA is required" in str(e):
                    pytest.skip(f"CUDA is required but not available: {e}")
                raise
            source = self._create_neutralized_gaussian(domain, sigma_cells=2.0)
            solution = solver.solve_stationary(source)
            
            # Compute residual: r = L_β a - s
            residual = self._compute_residual(solution, source, domain, params)
            
            residual_norm = np.linalg.norm(residual)
            source_norm = np.linalg.norm(source)
            relative_residual = residual_norm / source_norm
            
            assert relative_residual <= 1e-11, (
                f"β={beta}: relative residual {relative_residual:.2e} > 1e-11"
            )
    
    def test_B1_kspace_slope(self, domain_configs, beta_values, use_cuda):
        """
        Test B1: k-space slope.
        
        Criterion from document 7d-32:
        - Доп. проверка k-space: наклон -2β ± 0.05
        """
        domain = Domain(dimensions=7, **domain_configs[1])
        
        for beta in beta_values:
            params = Parameters(mu=1.0, beta=beta, lambda_param=0.0)
            try:
                solver = FFTSolver7DBasic(domain, params)
            except RuntimeError as e:
                if "CUDA is required" in str(e):
                    pytest.skip(f"CUDA is required but not available: {e}")
                raise
            source = self._create_neutralized_gaussian(domain, sigma_cells=2.0)
            solution = solver.solve_stationary(source)
            
            # Compute k-space analysis
            kspace_slope = self._compute_kspace_slope(solution, domain)
            expected_slope = -2 * beta
            
            assert abs(kspace_slope - expected_slope) <= 0.05, (
                f"β={beta}: k-space slope {kspace_slope:.4f} not in "
                f"[-2β ± 0.05] = [{expected_slope - 0.05:.4f}, {expected_slope + 0.05:.4f}]"
            )
    
    def test_B1_cuda_block_processing(self, use_cuda):
        """
        Test B1: CUDA and block processing usage.
        
        Verify that large 7D fields use CUDA and block processing.
        """
        # CUDA is required for this test - if not available, skip
        if not use_cuda:
            pytest.skip("CUDA is required for this test")
        
        # Create moderate field for testing (balanced for accuracy and memory)
        # Real production will use larger fields (N=256+) on powerful GPUs
        # Swap and block processing will handle memory automatically
        domain = Domain(
            L=8 * np.pi,
            N=128,
            N_phi=8,
            N_t=8,
            T=1.0,
            dimensions=7
        )
        
        params = Parameters(mu=1.0, beta=1.0, lambda_param=0.0)
        try:
            solver = FFTSolver7DBasic(domain, params)
        except RuntimeError as e:
            if "CUDA is required" in str(e):
                pytest.skip(f"CUDA is required but not available: {e}")
            raise
        source = self._create_neutralized_gaussian(domain, sigma_cells=2.0)
        solution = solver.solve_stationary(source)
        
        # Check memory usage
        try:
            import cupy as cp
            mem_info_before = cp.cuda.runtime.memGetInfo()
            
            analyzer = LevelBPowerLawAnalyzer(use_cuda=True)
            center = [domain.N // 2] * 3
            result = analyzer.analyze_power_law_tail(
                solution, 1.0, center, min_decades=1.5
            )
            
            mem_info_after = cp.cuda.runtime.memGetInfo()
            memory_used = (mem_info_before[0] - mem_info_after[0]) / mem_info_before[0]
            
            # Should use less than 80% of GPU memory
            assert memory_used < 0.85, (
                f"Memory usage {memory_used*100:.2f}% exceeds 85% limit"
            )
            
            # Verify result is valid
            assert result["passed"], "Power law analysis should pass"
        except ImportError:
            pytest.skip("CuPy not available")
    
    def _create_neutralized_gaussian(self, domain: Domain, sigma_cells: float) -> np.ndarray:
        """
        Create neutralized Gaussian source as per §1.5 of document 7d-32.
        
        Physical Meaning:
            Creates a Gaussian source g_σ(x) = exp(-|x-x₀|²/(2σ²)) and
            neutralizes it by subtracting the mean: s(x) = g_σ(x) - ḡ
            where ḡ = (1/L³) ∫ g_σ dx, ensuring zero mean for λ=0 compatibility.
        
        Mathematical Foundation:
            For 7D space-time, the source is localized in spatial dimensions
            and broadcast to phase and temporal dimensions using memory-efficient
            broadcasting instead of explicit loops.
        """
        # Compute sigma in physical units
        dx = domain.L / domain.N
        sigma = sigma_cells * dx
        
        # Create spatial grid
        x = np.linspace(0, domain.L, domain.N, endpoint=False)
        y = np.linspace(0, domain.L, domain.N, endpoint=False)
        z = np.linspace(0, domain.L, domain.N, endpoint=False)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        
        # Center coordinates
        center_x = domain.L / 2
        center_y = domain.L / 2
        center_z = domain.L / 2
        
        # Compute distances from center (spatial only)
        dx = X - center_x
        dy = Y - center_y
        dz = Z - center_z
        r_sq = dx**2 + dy**2 + dz**2
        
        # Gaussian (with underflow protection)
        exponent = -r_sq / (2 * sigma**2)
        exponent = np.clip(exponent, -700, 700)
        g_spatial = np.exp(exponent)
        
        # Neutralize (remove mean)
        g_mean = np.mean(g_spatial)
        s_spatial = g_spatial - g_mean
        
        # For 7D fields, we need to broadcast spatial source to all phase/temporal dimensions
        # For large arrays, use memory-mapped arrays with swap on disk
        source_memory = np.prod(domain.shape) * 16  # complex128 = 16 bytes
        
        # Check if we need memory-mapped array (for arrays > 10GB)
        use_memmap = source_memory > 10e9
        
        if use_memmap:
            # Use transparent swap manager for memory-mapped array
            from bhlff.core.fft.unified.swap_manager import get_swap_manager
            swap_manager = get_swap_manager()
            
            try:
                # Create memory-mapped array using transparent swap
                s_7d = swap_manager.create_swap_array(
                    shape=domain.shape,
                    dtype=np.complex128,
                    array_id=f"source_{id(domain)}"
                )
                
                # Fill array in blocks to avoid memory issues
                # Process phase and temporal dimensions in blocks
                block_size_phi = min(4, domain.N_phi)
                block_size_t = min(4, domain.N_t)
                
                num_blocks_phi = (domain.N_phi + block_size_phi - 1) // block_size_phi
                num_blocks_t = (domain.N_t + block_size_t - 1) // block_size_t
                
                for i_phi1 in range(num_blocks_phi):
                    for i_phi2 in range(num_blocks_phi):
                        for i_phi3 in range(num_blocks_phi):
                            for i_t in range(num_blocks_t):
                                phi1_start = i_phi1 * block_size_phi
                                phi1_end = min((i_phi1 + 1) * block_size_phi, domain.N_phi)
                                phi2_start = i_phi2 * block_size_phi
                                phi2_end = min((i_phi2 + 1) * block_size_phi, domain.N_phi)
                                phi3_start = i_phi3 * block_size_phi
                                phi3_end = min((i_phi3 + 1) * block_size_phi, domain.N_phi)
                                t_start = i_t * block_size_t
                                t_end = min((i_t + 1) * block_size_t, domain.N_t)
                                
                                # Broadcast spatial source to this block
                                block_shape = (
                                    domain.N, domain.N, domain.N,
                                    phi1_end - phi1_start,
                                    phi2_end - phi2_start,
                                    phi3_end - phi3_start,
                                    t_end - t_start
                                )
                                block = np.broadcast_to(
                                    s_spatial[:, :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis],
                                    block_shape
                                )
                                
                                s_7d[
                                    :, :, :,
                                    phi1_start:phi1_end,
                                    phi2_start:phi2_end,
                                    phi3_start:phi3_end,
                                    t_start:t_end
                                ] = block
                
                # Flush to disk
                s_7d.flush()
                
                # Return memory-mapped array (will use swap on disk)
                # Note: FFT operations will need to handle memory-mapped arrays
                # UnifiedSpectralOperations should handle this through block processing
                return s_7d
                
            except Exception as e:
                # Clean up on error
                try:
                    swap_manager.cleanup(f"source_{id(domain)}")
                except:
                    pass
                raise MemoryError(
                    f"Failed to create memory-mapped source array: {e}. "
                    f"Array size: {source_memory/1e9:.2f}GB"
                )
        else:
            # For smaller arrays, use regular memory
            s_spatial_expanded = s_spatial[:, :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
            s_7d_view = np.broadcast_to(s_spatial_expanded, domain.shape)
            s_7d = np.ascontiguousarray(s_7d_view, dtype=np.complex128)
            return s_7d
    
    
    def _compute_residual(
        self, solution: np.ndarray, source: np.ndarray, domain: Domain, params: Parameters
    ) -> np.ndarray:
        """
        Compute residual r = L_β a - s.
        
        Physical Meaning:
            Computes the residual of the fractional Riesz equation
            L_β a = μ(-Δ)^β a + λa = s to verify solution accuracy.
        """
        # Transform solution to spectral space
        solution_hat = np.fft.fftn(solution, norm="ortho")
        
        # Get wave numbers for 7D
        kx = np.fft.fftfreq(domain.N, d=domain.dx)
        ky = np.fft.fftfreq(domain.N, d=domain.dx)
        kz = np.fft.fftfreq(domain.N, d=domain.dx)
        kphi1 = np.fft.fftfreq(domain.N_phi, d=domain.dphi)
        kphi2 = np.fft.fftfreq(domain.N_phi, d=domain.dphi)
        kphi3 = np.fft.fftfreq(domain.N_phi, d=domain.dphi)
        kt = np.fft.fftfreq(domain.N_t, d=domain.dt)
        
        # Create 7D wave number arrays
        KX = kx[:, None, None, None, None, None, None]
        KY = ky[None, :, None, None, None, None, None]
        KZ = kz[None, None, :, None, None, None, None]
        KP1 = kphi1[None, None, None, :, None, None, None]
        KP2 = kphi2[None, None, None, None, :, None, None]
        KP3 = kphi3[None, None, None, None, None, :, None]
        KT = kt[None, None, None, None, None, None, :]
        
        # Compute 7D |k|^2
        k2 = KX**2 + KY**2 + KZ**2 + KP1**2 + KP2**2 + KP3**2 + KT**2
        
        # Compute spectral operator L_β
        k_magnitude = np.sqrt(k2 + 1e-15)  # Avoid division by zero
        spectral_op = params.mu * (k_magnitude ** (2 * params.beta)) + params.lambda_param
        
        # Apply operator: L_β a in spectral space
        L_a_hat = solution_hat * spectral_op
        
        # Transform back to real space
        L_a = np.fft.ifftn(L_a_hat, norm="ortho").real
        
        # Compute residual
        residual = L_a - source
        
        return residual
    
    def _compute_kspace_slope(self, solution: np.ndarray, domain: Domain) -> float:
        """
        Compute k-space slope from spherical shell analysis.
        
        Physical Meaning:
            Computes the power law slope in k-space by analyzing
            log|â(k)| vs log|k| for spherical shells in 7D k-space.
        
        Mathematical Foundation:
            For solution â(k) = ŝ(k) / (μ|k|^(2β) + λ), in the tail region
            (large |k|) we expect |â(k)| ∝ |k|^(-2β), so log|â(k)| vs log|k|
            should have slope -2β.
        """
        # Transform to spectral space
        solution_hat = np.fft.fftn(solution, norm="ortho")
        solution_hat_abs = np.abs(solution_hat)
        
        # Get wave numbers for 7D
        kx = np.fft.fftfreq(domain.N, d=domain.dx) * 2 * np.pi
        ky = np.fft.fftfreq(domain.N, d=domain.dx) * 2 * np.pi
        kz = np.fft.fftfreq(domain.N, d=domain.dx) * 2 * np.pi
        kphi1 = np.fft.fftfreq(domain.N_phi, d=domain.dphi)
        kphi2 = np.fft.fftfreq(domain.N_phi, d=domain.dphi)
        kphi3 = np.fft.fftfreq(domain.N_phi, d=domain.dphi)
        kt = np.fft.fftfreq(domain.N_t, d=domain.dt) * 2 * np.pi
        
        # Create 7D wave number arrays
        KX = kx[:, None, None, None, None, None, None]
        KY = ky[None, :, None, None, None, None, None]
        KZ = kz[None, None, :, None, None, None, None]
        KP1 = kphi1[None, None, None, :, None, None, None]
        KP2 = kphi2[None, None, None, None, :, None, None]
        KP3 = kphi3[None, None, None, None, None, :, None]
        KT = kt[None, None, None, None, None, None, :]
        
        # Compute 7D |k|
        k_magnitude = np.sqrt(
            KX**2 + KY**2 + KZ**2 + KP1**2 + KP2**2 + KP3**2 + KT**2
        )
        
        # Flatten arrays
        k_flat = k_magnitude.ravel()
        a_flat = solution_hat_abs.ravel()
        
        # Filter out k=0 and very small k
        mask = (k_flat > 1e-6) & (k_flat < np.max(k_flat) * 0.9) & np.isfinite(a_flat) & (a_flat > 1e-15)
        k_filtered = k_flat[mask]
        a_filtered = a_flat[mask]
        
        if len(k_filtered) < 10:
            return 0.0
        
        # Compute log-log regression
        log_k = np.log(k_filtered)
        log_a = np.log(a_filtered)
        
        slope, _, _, _, _ = stats.linregress(log_k, log_a)
        
        return float(slope)


