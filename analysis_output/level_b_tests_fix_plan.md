# Author: Vasiliy Zdanovskiy
# email: vasilyvz@gmail.com

# Пошаговый план исправления и тестирования уровня B

## Общая структура плана

План разделен на этапы:
1. **Этап 1**: Исправление тестов B1 (Power Law Tails)
2. **Этап 2**: Создание тестов B2 (Spherical Nodes Absence)
3. **Этап 3**: Исправление тестов B3 (Topological Charge)
4. **Этап 4**: Исправление тестов B4 (Zone Separation)
5. **Этап 5**: Интеграция CUDA и блочной обработки
6. **Этап 6**: Тесты производительности и больших полей
7. **Этап 7**: Финальная валидация

---

## Этап 1: Исправление тестов B1 (Power Law Tails)

### Цель этапа
Создать полноценные тесты B1, проверяющие все критерии из документа 7d-32 с использованием CUDA и блочной обработки.

### Шаг 1.1: Создание нового теста B1

**Файл:** `tests/unit/test_level_b/test_B1_power_law_tails.py`

**Содержание:**
```python
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test B1: BVP Power Law Tails for Level B.

This module implements comprehensive tests for power law tail behavior
A(r) ∝ r^(2β-3) with all acceptance criteria from document 7d-32.

Theoretical Background:
    Tests validate power law behavior in homogeneous medium governed by
    Riesz operator L_β = μ(-Δ)^β + λ with λ=0, confirming theoretical
    prediction A(r) ∝ r^(2β-3) for different β values.

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
from bhlff.core.sources.bvp_source import BVPSource
from bhlff.models.level_b.power_law_analyzer import LevelBPowerLawAnalyzer
from bhlff.core.bvp.bvp_core.bvp_fft_solver import BVPFFTSolver


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
        return [
            {"L": L, "N": 256, "N_phi": 16, "N_t": 32, "T": 1.0},
            {"L": L, "N": 384, "N_phi": 16, "N_t": 32, "T": 1.0},
            {"L": L, "N": 512, "N_phi": 16, "N_t": 32, "T": 1.0},
        ]
    
    @pytest.fixture
    def beta_values(self):
        """Beta values for testing as per document 7d-32."""
        return [0.6, 0.8, 1.0, 1.2, 1.4]
    
    @pytest.fixture
    def use_cuda(self):
        """CUDA usage flag."""
        return os.getenv("BHLFF_DISABLE_CUDA", "0") != "1"
    
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
            params = Parameters(mu=1.0, beta=beta, lambda_param=0.0)
            solver = BVPFFTSolver(domain, params)
            
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
            
            assert slope_ci[0] <= theoretical_slope + 0.05
            assert slope_ci[1] >= theoretical_slope - 0.05, (
                f"β={beta}: slope {slope:.4f} not in CI for theoretical {theoretical_slope:.4f}"
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
            solver = BVPFFTSolver(domain, params)
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
            assert result["decades"] >= 1.5, (
                f"β={beta}: {result['decades']:.2f} decades < 1.5"
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
            solver = BVPFFTSolver(domain, params)
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
            solver = BVPFFTSolver(domain, params)
            source = self._create_neutralized_gaussian(domain, sigma_cells=2.0)
            solution = solver.solve_stationary(source)
            
            # Compute residual: r = L_β a - s
            residual = solver.compute_residual(solution, source)
            
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
            solver = BVPFFTSolver(domain, params)
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
        if not use_cuda:
            pytest.skip("CUDA not available")
        
        # Create large field requiring block processing
        domain = Domain(
            L=8 * np.pi,
            N=512,
            N_phi=32,
            N_t=64,
            T=1.0,
            dimensions=7
        )
        
        params = Parameters(mu=1.0, beta=1.0, lambda_param=0.0)
        solver = BVPFFTSolver(domain, params)
        source = self._create_neutralized_gaussian(domain, sigma_cells=2.0)
        solution = solver.solve_stationary(source)
        
        # Check memory usage
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
    
    def _create_neutralized_gaussian(self, domain: Domain, sigma_cells: float) -> np.ndarray:
        """Create neutralized Gaussian source as per §1.5 of document 7d-32."""
        # Implementation of neutralized Gaussian
        # g_σ(x) = exp(-|x-x₀|²/(2σ²))
        # s(x) = g_σ(x) - ḡ where ḡ = (1/L³) ∫ g_σ dx
        pass
    
    def _compute_kspace_slope(self, solution: np.ndarray, domain: Domain) -> float:
        """Compute k-space slope from spherical shell analysis."""
        # Implementation of k-space slope computation
        # Linear regression log|â(k)| vs log|k| for spherical shells
        pass
```

### Шаг 1.2: Обновление анализатора для поддержки всех критериев

**Файл:** `bhlff/models/level_b/power_law_analyzer.py` (или соответствующий файл)

**Изменения:**
1. Добавить метод `analyze_power_law_tail()` с параметром `min_decades=1.5`
2. Добавить вычисление 95% доверительного интервала для наклона
3. Добавить проверку количества декад
4. Добавить метод `compute_kspace_slope()` для k-space анализа

### Шаг 1.3: Тестирование шага 1

**Команды:**
```bash
# Запустить тесты B1
pytest tests/unit/test_level_b/test_B1_power_law_tails.py -v

# Проверить покрытие
pytest tests/unit/test_level_b/test_B1_power_law_tails.py --cov=bhlff.models.level_b --cov-report=html
```

**Критерии успеха:**
- Все тесты B1 проходят
- Покрытие кода ≥ 90%
- Все критерии из документа 7d-32 проверяются

---

## Этап 2: Создание тестов B2 (Spherical Nodes Absence)

### Цель этапа
Создать полноценные тесты B2 для проверки отсутствия сферических стоячих узлов.

### Шаг 2.1: Создание нового теста B2

**Файл:** `tests/unit/test_level_b/test_B2_spherical_nodes_absence.py`

**Содержание:**
```python
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test B2: BVP Spherical Nodes Absence for Level B.

This module implements comprehensive tests for absence of spherical
standing nodes with all acceptance criteria from document 7d-32.

Theoretical Background:
    Tests validate that in homogeneous "interval-free" BVP medium,
    spherical standing nodes do not form due to absence of poles in
    spectral symbol D(k) = μk^(2β).
"""

import numpy as np
import pytest
from scipy.signal import savgol_filter
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.domain.parameters import Parameters
from bhlff.core.sources.bvp_source import BVPSource
from bhlff.models.level_b.node_analyzer import LevelBNodeAnalyzer
from bhlff.core.bvp.bvp_core.bvp_fft_solver import BVPFFTSolver


class TestB2SphericalNodesAbsence:
    """
    Comprehensive tests for B2: Spherical Nodes Absence.
    
    Physical Meaning:
        Validates that BVP envelope does not exhibit spherical standing
        nodes in homogeneous medium, ensuring monotonic decay.
    """
    
    @pytest.fixture
    def beta_values(self):
        """Beta values for testing (at least 3 as per document)."""
        return [0.8, 1.0, 1.2]
    
    @pytest.fixture
    def domain(self):
        """Create test domain."""
        return Domain(
            L=8 * np.pi,
            N=384,
            N_phi=16,
            N_t=32,
            T=1.0,
            dimensions=7
        )
    
    def test_B2_sign_changes(self, domain, beta_values):
        """
        Test B2: Number of sign changes ≤ 1.
        
        Criterion from document 7d-32:
        - Число изменений знака Z ≤ 1
        """
        for beta in beta_values:
            params = Parameters(mu=1.0, beta=beta, lambda_param=0.0)
            solver = BVPFFTSolver(domain, params)
            source = self._create_neutralized_gaussian(domain, sigma_cells=2.0)
            solution = solver.solve_stationary(source)
            
            # Compute radial profile
            analyzer = LevelBNodeAnalyzer(use_cuda=True)
            center = [domain.N // 2] * 3
            radial_profile = analyzer._compute_radial_profile(solution, center)
            
            # Smooth with Savitzky-Golay filter (window ≥ 9, polynomial 3)
            A = radial_profile["A"]
            A_smooth = savgol_filter(A, window_length=min(9, len(A)//2*2-1), polyorder=3)
            
            # Compute derivative
            A_prime = np.gradient(A_smooth)
            
            # Count sign changes
            sign_changes = np.sum(np.diff(np.sign(A_prime)) != 0)
            
            assert sign_changes <= 1, (
                f"β={beta}: {sign_changes} sign changes > 1"
            )
    
    def test_B2_maxima_minima(self, domain, beta_values):
        """
        Test B2: Number of local maxima-minima ≤ 2.
        
        Criterion from document 7d-32:
        - Число локальных максимумов-минимумов M ≤ 2
        """
        for beta in beta_values:
            params = Parameters(mu=1.0, beta=beta, lambda_param=0.0)
            solver = BVPFFTSolver(domain, params)
            source = self._create_neutralized_gaussian(domain, sigma_cells=2.0)
            solution = solver.solve_stationary(source)
            
            analyzer = LevelBNodeAnalyzer(use_cuda=True)
            center = [domain.N // 2] * 3
            radial_profile = analyzer._compute_radial_profile(solution, center)
            
            A = radial_profile["A"]
            A_smooth = savgol_filter(A, window_length=min(9, len(A)//2*2-1), polyorder=3)
            
            # Find local extrema
            from scipy.signal import argrelextrema
            maxima = argrelextrema(A_smooth, np.greater)[0]
            minima = argrelextrema(A_smooth, np.less)[0]
            total_extrema = len(maxima) + len(minima)
            
            assert total_extrema <= 2, (
                f"β={beta}: {total_extrema} extrema > 2"
            )
    
    def test_B2_minimum_amplitude(self, domain, beta_values):
        """
        Test B2: Minimum amplitude > 10⁻¹⁴ max A.
        
        Criterion from document 7d-32:
        - min_r A(r) > 10⁻¹⁴ max A на [r_min, r_max]
        """
        for beta in beta_values:
            params = Parameters(mu=1.0, beta=beta, lambda_param=0.0)
            solver = BVPFFTSolver(domain, params)
            source = self._create_neutralized_gaussian(domain, sigma_cells=2.0)
            solution = solver.solve_stationary(source)
            
            analyzer = LevelBNodeAnalyzer(use_cuda=True)
            center = [domain.N // 2] * 3
            radial_profile = analyzer._compute_radial_profile(solution, center)
            
            r = radial_profile["r"]
            A = radial_profile["A"]
            
            # Filter range [r_min, r_max] where r_min = 4Δ, r_max = L/4
            r_min = 4 * (domain.L / domain.N)
            r_max = domain.L / 4
            mask = (r >= r_min) & (r <= r_max)
            
            if np.any(mask):
                A_filtered = A[mask]
                min_A = np.min(A_filtered)
                max_A = np.max(A_filtered)
                threshold = 1e-14 * max_A
                
                assert min_A > threshold, (
                    f"β={beta}: min A = {min_A:.2e} ≤ {threshold:.2e} = 10⁻¹⁴ max A"
                )
    
    def _create_neutralized_gaussian(self, domain: Domain, sigma_cells: float) -> np.ndarray:
        """Create neutralized Gaussian source."""
        # Implementation
        pass
```

### Шаг 2.2: Обновление NodeAnalyzer

**Файл:** `bhlff/models/level_b/node_analyzer.py`

**Изменения:**
1. Добавить метод `_compute_radial_profile()` если отсутствует
2. Добавить метод `check_spherical_nodes()` с проверкой всех критериев B2

### Шаг 2.3: Тестирование шага 2

**Команды:**
```bash
pytest tests/unit/test_level_b/test_B2_spherical_nodes_absence.py -v
```

**Критерии успеха:**
- Все тесты B2 проходят
- Все критерии из документа 7d-32 проверяются

---

## Этап 3: Исправление тестов B3 (Topological Charge)

### Цель этапа
Разделить тесты B3 на B3-S (2D синтетика) и B3-P (3D PDE) с проверкой всех критериев.

### Шаг 3.1: Создание теста B3-S (2D синтетика)

**Файл:** `tests/unit/test_level_b/test_B3S_topological_charge_synthetic.py`

**Содержание:**
```python
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test B3-S: Topological Charge (2D Synthetic) for Level B.

This module implements tests for topological charge computation
on synthetic 2D fields with acceptance criteria from document 7d-32.

Theoretical Background:
    Tests validate topological charge quantization on synthetic
    2D fields a_syn(x,y) = f(ρ) e^(iq arg(x-x₀+i(y-y₀))) with
    known charge q ∈ {±1, ±2}.
"""

import numpy as np
import pytest
from typing import List

from bhlff.core.domain import Domain
from bhlff.models.level_b.node_analyzer import LevelBNodeAnalyzer


class TestB3STopologicalChargeSynthetic:
    """
    Tests for B3-S: Topological Charge (2D Synthetic).
    
    Physical Meaning:
        Validates topological charge computation on synthetic 2D fields
        with known charge values.
    """
    
    @pytest.fixture
    def domain_2d(self):
        """Create 2D domain as per document 7d-32."""
        return Domain(L=2 * np.pi, N=512, dimensions=2)
    
    @pytest.fixture
    def q_values(self):
        """Charge values for testing."""
        return [-2, -1, 1, 2]
    
    @pytest.fixture
    def N_values(self):
        """Grid resolutions for testing."""
        return [256, 512, 1024]
    
    def test_B3S_charge_accuracy(self, domain_2d, q_values, N_values):
        """
        Test B3-S: Charge accuracy |q̄ - q| ≤ 0.01.
        
        Criterion from document 7d-32:
        - |q̄ - q| ≤ 0.01 для всех q ∈ {±1, ±2} и N ∈ {256, 512, 1024}
        """
        for N in N_values:
            domain = Domain(L=2 * np.pi, N=N, dimensions=2)
            for q in q_values:
                # Create synthetic field
                field = self._create_synthetic_field(domain, q, xi_cells=3)
                
                # Compute topological charge
                analyzer = LevelBNodeAnalyzer(use_cuda=True)
                center = [domain.N // 2, domain.N // 2]
                
                # Compute charge for multiple radii
                radii = [6, 8, 10]  # In cells
                charges = []
                for rho in radii:
                    charge = analyzer.compute_topological_charge_2d(
                        field, center, rho
                    )
                    charges.append(charge)
                
                # Average charge
                q_bar = np.mean(charges)
                
                assert abs(q_bar - q) <= 0.01, (
                    f"N={N}, q={q}: |q̄ - q| = {abs(q_bar - q):.4f} > 0.01"
                )
    
    def test_B3S_charge_dispersion(self, domain_2d, q_values):
        """
        Test B3-S: Charge dispersion ≤ 0.01.
        
        Criterion from document 7d-32:
        - Дисперсия оценок по ρ ≤ 0.01
        """
        domain = Domain(L=2 * np.pi, N=512, dimensions=2)
        
        for q in q_values:
            field = self._create_synthetic_field(domain, q, xi_cells=3)
            analyzer = LevelBNodeAnalyzer(use_cuda=True)
            center = [domain.N // 2, domain.N // 2]
            
            # Compute charge for multiple radii
            radii = [6, 8, 10]
            charges = []
            for rho in radii:
                charge = analyzer.compute_topological_charge_2d(
                    field, center, rho
                )
                charges.append(charge)
            
            # Compute dispersion
            dispersion = np.std(charges)
            
            assert dispersion <= 0.01, (
                f"q={q}: dispersion {dispersion:.4f} > 0.01"
            )
    
    def _create_synthetic_field(
        self, domain: Domain, q: int, xi_cells: float
    ) -> np.ndarray:
        """
        Create synthetic field as per §4.2 (S) of document 7d-32.
        
        a_syn(x,y) = f(ρ) e^(iq arg(x-x₀+i(y-y₀)))
        where f(ρ) = sqrt(ρ²/(ρ²+ξ²))
        """
        # Implementation
        pass
```

### Шаг 3.2: Создание теста B3-P (3D PDE)

**Файл:** `tests/unit/test_level_b/test_B3P_topological_charge_pde.py`

**Содержание:**
```python
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test B3-P: Topological Charge (3D PDE) for Level B.

This module implements tests for topological charge computation
on 3D PDE solutions with acceptance criteria from document 7d-32.
"""

import numpy as np
import pytest

from bhlff.core.domain import Domain
from bhlff.core.domain.parameters import Parameters
from bhlff.core.bvp.bvp_core.bvp_fft_solver import BVPFFTSolver
from bhlff.models.level_b.node_analyzer import LevelBNodeAnalyzer


class TestB3PTopologicalChargePDE:
    """
    Tests for B3-P: Topological Charge (3D PDE).
    
    Physical Meaning:
        Validates topological charge computation on 3D PDE solutions
        with azimuthal source excitation.
    """
    
    @pytest.fixture
    def domain_3d(self):
        """Create 3D domain as per document 7d-32."""
        return Domain(L=4 * np.pi, N=384, dimensions=3)
    
    @pytest.fixture
    def N_values(self):
        """Grid resolutions for convergence testing."""
        return [256, 384]
    
    @pytest.fixture
    def q_values(self):
        """Charge values for testing."""
        return [-1, 1]
    
    def test_B3P_charge_accuracy(self, domain_3d, q_values):
        """
        Test B3-P: Charge accuracy |q̄ - q| ≤ 0.05.
        
        Criterion from document 7d-32:
        - |q̄ - q| ≤ 0.05
        """
        for q in q_values:
            # Create azimuthal source
            params = Parameters(mu=1.0, beta=1.0, lambda_param=0.0)
            solver = BVPFFTSolver(domain_3d, params)
            source = self._create_azimuthal_source(domain_3d, q, sigma_cells=2.0)
            
            # Solve stationary problem
            solution = solver.solve_stationary(source)
            
            # Extract slice at z = L/2
            z_slice_idx = domain_3d.N // 2
            solution_slice = solution[:, :, z_slice_idx]
            
            # Compute topological charge on slice
            analyzer = LevelBNodeAnalyzer(use_cuda=True)
            center = [domain_3d.N // 2, domain_3d.N // 2]
            
            radii = [6, 8, 10]
            charges = []
            for rho in radii:
                charge = analyzer.compute_topological_charge_2d(
                    solution_slice, center, rho
                )
                charges.append(charge)
            
            q_bar = np.mean(charges)
            
            assert abs(q_bar - q) <= 0.05, (
                f"q={q}: |q̄ - q| = {abs(q_bar - q):.4f} > 0.05"
            )
    
    def test_B3P_grid_stability(self, q_values, N_values):
        """
        Test B3-P: Grid stability |q̄_{N₂} - q̄_{N₁}| ≤ 0.02.
        
        Criterion from document 7d-32:
        - Стабильность по сетке: |q̄_{N₂} - q̄_{N₁}| ≤ 0.02
        """
        q = 1  # Test with one charge value
        
        charges = []
        for N in N_values:
            domain = Domain(L=4 * np.pi, N=N, dimensions=3)
            params = Parameters(mu=1.0, beta=1.0, lambda_param=0.0)
            solver = BVPFFTSolver(domain, params)
            source = self._create_azimuthal_source(domain, q, sigma_cells=2.0)
            solution = solver.solve_stationary(source)
            
            z_slice_idx = domain.N // 2
            solution_slice = solution[:, :, z_slice_idx]
            
            analyzer = LevelBNodeAnalyzer(use_cuda=True)
            center = [domain.N // 2, domain.N // 2]
            
            radii = [6, 8, 10]
            slice_charges = []
            for rho in radii:
                charge = analyzer.compute_topological_charge_2d(
                    solution_slice, center, rho
                )
                slice_charges.append(charge)
            
            charges.append((N, np.mean(slice_charges)))
        
        # Check stability between resolutions
        for i in range(len(charges) - 1):
            N1, q1 = charges[i]
            N2, q2 = charges[i + 1]
            diff = abs(q2 - q1)
            assert diff <= 0.02, (
                f"Stability failed: |q̄_{N2} - q̄_{N1}| = {diff:.4f} > 0.02"
            )
    
    def _create_azimuthal_source(
        self, domain: Domain, q: int, sigma_cells: float
    ) -> np.ndarray:
        """
        Create azimuthal source as per §4.2 (P) of document 7d-32.
        
        s(x) = g_σ(x) e^(iqθ(x)) - s̄
        where θ is azimuthal angle in (x,y) plane
        """
        # Implementation
        pass
```

### Шаг 3.3: Обновление NodeAnalyzer для поддержки B3-S и B3-P

**Файл:** `bhlff/models/level_b/node_analyzer.py`

**Изменения:**
1. Добавить метод `compute_topological_charge_2d()` для 2D вычислений
2. Добавить поддержку вычисления на срезе для 3D полей

### Шаг 3.4: Тестирование шага 3

**Команды:**
```bash
pytest tests/unit/test_level_b/test_B3S_topological_charge_synthetic.py -v
pytest tests/unit/test_level_b/test_B3P_topological_charge_pde.py -v
```

**Критерии успеха:**
- Все тесты B3-S и B3-P проходят
- Все критерии из документа 7d-32 проверяются

---

## Этап 4: Исправление тестов B4 (Zone Separation)

### Цель этапа
Дополнить тесты B4 проверкой сходимости и согласованности хвоста.

### Шаг 4.1: Обновление теста B4

**Файл:** `tests/unit/test_level_b/test_B4_zone_separation.py`

**Содержание:**
```python
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Test B4: Zone Separation for Level B.

This module implements comprehensive tests for zone separation
with all acceptance criteria from document 7d-32.
"""

import numpy as np
import pytest

from bhlff.core.domain import Domain
from bhlff.core.domain.parameters import Parameters
from bhlff.core.bvp.bvp_core.bvp_fft_solver import BVPFFTSolver
from bhlff.models.level_b.zone_analyzer import LevelBZoneAnalyzer
from bhlff.models.level_b.power_law_analyzer import LevelBPowerLawAnalyzer


class TestB4ZoneSeparation:
    """
    Comprehensive tests for B4: Zone Separation.
    
    Physical Meaning:
        Validates quantitative separation of zones (core/transition/tail)
        with convergence and consistency checks.
    """
    
    @pytest.fixture
    def domain_configs(self):
        """Domain configurations for convergence testing."""
        L = 8 * np.pi
        return [
            {"L": L, "N": 256, "N_phi": 16, "N_t": 32, "T": 1.0},
            {"L": L, "N": 512, "N_phi": 16, "N_t": 32, "T": 1.0},
        ]
    
    @pytest.fixture
    def beta(self):
        """Beta value for testing (as per document)."""
        return 1.0
    
    def test_B4_zone_convergence(self, domain_configs, beta):
        """
        Test B4: Zone radius convergence ≤ 5%.
        
        Criterion from document 7d-32:
        - Сходимость: |r_core(N₂) - r_core(N₁)|/r_core(N₁) ≤ 5%
        - Аналогично для r_tail
        """
        params = Parameters(mu=1.0, beta=beta, lambda_param=0.0)
        
        results = []
        for config in domain_configs:
            domain = Domain(dimensions=7, **config)
            solver = BVPFFTSolver(domain, params)
            source = self._create_neutralized_gaussian(domain, sigma_cells=2.0)
            solution = solver.solve_stationary(source)
            
            analyzer = LevelBZoneAnalyzer(use_cuda=True)
            center = [domain.N // 2] * 3
            thresholds = {
                "S_core": 1.0,
                "C_core": 1.0,
                "S_tail": 0.3,
                "C_tail": 0.3,
            }
            
            result = analyzer.separate_zones(
                solution, center, thresholds, domain_size=domain.L
            )
            results.append((config["N"], result["r_core"], result["r_tail"]))
        
        # Check convergence
        N1, r_core1, r_tail1 = results[0]
        N2, r_core2, r_tail2 = results[1]
        
        core_convergence = abs(r_core2 - r_core1) / r_core1
        tail_convergence = abs(r_tail2 - r_tail1) / r_tail1
        
        assert core_convergence <= 0.05, (
            f"r_core convergence {core_convergence*100:.2f}% > 5%"
        )
        assert tail_convergence <= 0.05, (
            f"r_tail convergence {tail_convergence*100:.2f}% > 5%"
        )
    
    def test_B4_zone_ordering(self, domain_configs, beta):
        """
        Test B4: Zone ordering 0 < r_core < r_tail < L/4.
        
        Criterion from document 7d-32:
        - Упорядочение: 0 < r_core < r_tail < L/4
        """
        domain = Domain(dimensions=7, **domain_configs[1])
        params = Parameters(mu=1.0, beta=beta, lambda_param=0.0)
        solver = BVPFFTSolver(domain, params)
        source = self._create_neutralized_gaussian(domain, sigma_cells=2.0)
        solution = solver.solve_stationary(source)
        
        analyzer = LevelBZoneAnalyzer(use_cuda=True)
        center = [domain.N // 2] * 3
        thresholds = {
            "S_core": 1.0,
            "C_core": 1.0,
            "S_tail": 0.3,
            "C_tail": 0.3,
        }
        
        result = analyzer.separate_zones(
            solution, center, thresholds, domain_size=domain.L
        )
        
        r_core = result["r_core"]
        r_tail = result["r_tail"]
        L = domain.L
        
        assert r_core > 0, "r_core should be positive"
        assert r_tail > r_core, "r_tail should be greater than r_core"
        assert r_tail < L / 4, "r_tail should be less than L/4"
    
    def test_B4_tail_consistency(self, domain_configs, beta):
        """
        Test B4: Tail consistency with B1.
        
        Criterion from document 7d-32:
        - Согласованность хвоста: |p̂ - (2β-3)| ≤ 0.05 на [r_tail, r_max]
        """
        domain = Domain(dimensions=7, **domain_configs[1])
        params = Parameters(mu=1.0, beta=beta, lambda_param=0.0)
        solver = BVPFFTSolver(domain, params)
        source = self._create_neutralized_gaussian(domain, sigma_cells=2.0)
        solution = solver.solve_stationary(source)
        
        # Separate zones
        zone_analyzer = LevelBZoneAnalyzer(use_cuda=True)
        center = [domain.N // 2] * 3
        thresholds = {
            "S_core": 1.0,
            "C_core": 1.0,
            "S_tail": 0.3,
            "C_tail": 0.3,
        }
        zone_result = zone_analyzer.separate_zones(
            solution, center, thresholds, domain_size=domain.L
        )
        r_tail = zone_result["r_tail"]
        
        # Analyze power law on tail region
        power_law_analyzer = LevelBPowerLawAnalyzer(use_cuda=True)
        result = power_law_analyzer.analyze_power_law_tail(
            solution, beta, center, min_decades=1.5, r_min=r_tail
        )
        
        slope = result["slope"]
        theoretical_slope = 2 * beta - 3
        
        assert abs(slope - theoretical_slope) <= 0.05, (
            f"Tail consistency failed: |p̂ - (2β-3)| = "
            f"{abs(slope - theoretical_slope):.4f} > 0.05"
        )
    
    def _create_neutralized_gaussian(self, domain: Domain, sigma_cells: float) -> np.ndarray:
        """Create neutralized Gaussian source."""
        # Implementation
        pass
```

### Шаг 4.2: Обновление ZoneAnalyzer

**Файл:** `bhlff/models/level_b/zone_analyzer.py`

**Изменения:**
1. Добавить поддержку параметра `domain_size` для проверки упорядочения
2. Улучшить вычисление индикаторов S(r) и C(r)

### Шаг 4.3: Тестирование шага 4

**Команды:**
```bash
pytest tests/unit/test_level_b/test_B4_zone_separation.py -v
```

**Критерии успеха:**
- Все тесты B4 проходят
- Все критерии из документа 7d-32 проверяются

---

## Этап 5: Интеграция CUDA и блочной обработки

### Цель этапа
Добавить проверки использования CUDA и блочной обработки во все основные тесты.

### Шаг 5.1: Создание утилит для проверки CUDA

**Файл:** `tests/unit/test_level_b/cuda_test_utils.py`

**Содержание:**
```python
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Utilities for testing CUDA and block processing usage in Level B tests.
"""

import numpy as np
from typing import Dict, Any, Optional
import os

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None


def check_cuda_usage(analyzer) -> bool:
    """Check if analyzer is using CUDA."""
    return hasattr(analyzer, 'use_cuda') and analyzer.use_cuda and CUDA_AVAILABLE


def check_memory_usage(max_fraction: float = 0.85) -> Dict[str, Any]:
    """
    Check GPU memory usage.
    
    Args:
        max_fraction: Maximum allowed memory fraction (default 0.85 for safety margin)
    
    Returns:
        Dictionary with memory usage information
    """
    if not CUDA_AVAILABLE:
        return {"cuda_available": False, "memory_used": 0.0}
    
    mem_info = cp.cuda.runtime.memGetInfo()
    free_before = mem_info[0]
    total = mem_info[1]
    used = total - free_before
    fraction = used / total if total > 0 else 0.0
    
    return {
        "cuda_available": True,
        "memory_used": fraction,
        "memory_used_bytes": used,
        "memory_total_bytes": total,
        "within_limit": fraction < max_fraction,
    }


def create_large_7d_field(
    N: int = 512,
    N_phi: int = 32,
    N_t: int = 64,
    target_memory_fraction: float = 0.75
) -> np.ndarray:
    """
    Create large 7D field that requires block processing.
    
    Args:
        N: Spatial resolution
        N_phi: Phase dimension resolution
        N_t: Temporal dimension resolution
        target_memory_fraction: Target memory fraction to use
    
    Returns:
        Large 7D field array
    """
    if CUDA_AVAILABLE:
        mem_info = cp.cuda.runtime.memGetInfo()
        total_memory = mem_info[1]
        target_memory = target_memory_fraction * total_memory
        
        # Calculate field size (complex128 = 16 bytes per element)
        bytes_per_element = 16
        target_elements = int(target_memory / bytes_per_element)
        
        # Adjust dimensions to fit target memory
        current_elements = N ** 3 * N_phi ** 3 * N_t
        if current_elements < target_elements:
            # Scale up dimensions proportionally
            scale = (target_elements / current_elements) ** (1/7)
            N = int(N * scale)
            N_phi = int(N_phi * scale)
            N_t = int(N_t * scale)
    
    shape = (N, N, N, N_phi, N_phi, N_phi, N_t)
    field = np.random.rand(*shape).astype(np.complex128)
    return field


def verify_block_processing_used(analyzer, field: np.ndarray) -> bool:
    """
    Verify that block processing is used for large fields.
    
    Args:
        analyzer: Analyzer instance
        field: Field array
    
    Returns:
        True if block processing should be used
    """
    if not CUDA_AVAILABLE:
        return False
    
    # Check if field size requires block processing
    field_size_bytes = field.nbytes
    mem_info = cp.cuda.runtime.memGetInfo()
    free_memory = mem_info[0]
    max_memory = 0.8 * free_memory
    
    # FFT requires ~4x memory
    required_memory = field_size_bytes * 4
    
    return required_memory > max_memory
```

### Шаг 5.2: Добавление проверок CUDA в основные тесты

**Изменения в тестах B1-B4:**
1. Добавить проверку использования CUDA
2. Добавить проверку использования памяти GPU
3. Добавить тесты для больших полей

**Пример для B1:**
```python
def test_B1_cuda_block_processing(self, use_cuda):
    """Test that B1 uses CUDA and block processing for large fields."""
    if not use_cuda:
        pytest.skip("CUDA not available")
    
    # Create large field requiring block processing
    large_field = create_large_7d_field(target_memory_fraction=0.75)
    
    # Check memory before
    mem_before = check_memory_usage()
    
    # Run analysis
    analyzer = LevelBPowerLawAnalyzer(use_cuda=True)
    result = analyzer.analyze_power_law_tail(large_field, 1.0, center)
    
    # Check memory after
    mem_after = check_memory_usage()
    
    # Verify CUDA is used
    assert check_cuda_usage(analyzer), "CUDA should be used"
    
    # Verify memory usage is reasonable
    assert mem_after["within_limit"], (
        f"Memory usage {mem_after['memory_used']*100:.2f}% exceeds limit"
    )
    
    # Verify block processing is used for large fields
    assert verify_block_processing_used(analyzer, large_field), (
        "Block processing should be used for large fields"
    )
```

### Шаг 5.3: Тестирование шага 5

**Команды:**
```bash
# Run all tests with CUDA checks
pytest tests/unit/test_level_b/ -v -k "cuda or block"

# Run with CUDA enabled
BHLFF_DISABLE_CUDA=0 pytest tests/unit/test_level_b/ -v
```

**Критерии успеха:**
- Все проверки CUDA проходят
- Память GPU используется эффективно (< 85%)
- Блочная обработка используется для больших полей

---

## Этап 6: Тесты производительности и больших полей

### Цель этапа
Создать тесты производительности для больших 7D полей с проверкой использования памяти.

### Шаг 6.1: Создание тестов производительности

**Файл:** `tests/unit/test_level_b/test_performance_large_fields.py`

**Содержание:**
```python
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Performance tests for Level B with large 7D fields.

This module tests performance and memory usage for large 7D fields
requiring block processing and CUDA acceleration.
"""

import numpy as np
import pytest
import time
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.domain.parameters import Parameters
from bhlff.core.bvp.bvp_core.bvp_fft_solver import BVPFFTSolver
from bhlff.models.level_b.power_law_analyzer import LevelBPowerLawAnalyzer
from bhlff.models.level_b.node_analyzer import LevelBNodeAnalyzer
from bhlff.models.level_b.zone_analyzer import LevelBZoneAnalyzer
from tests.unit.test_level_b.cuda_test_utils import (
    create_large_7d_field,
    check_memory_usage,
    verify_block_processing_used,
)


class TestPerformanceLargeFields:
    """
    Performance tests for large 7D fields.
    
    Physical Meaning:
        Validates that Level B analysis works correctly and efficiently
        on large 7D fields using CUDA and block processing.
    """
    
    @pytest.fixture
    def large_domain(self):
        """Create large 7D domain."""
        return Domain(
            L=8 * np.pi,
            N=512,
            N_phi=32,
            N_t=64,
            T=1.0,
            dimensions=7
        )
    
    def test_large_field_power_law_analysis(self, large_domain):
        """Test power law analysis on large field."""
        # Create large field
        field = create_large_7d_field(
            N=large_domain.N,
            N_phi=large_domain.N_phi,
            N_t=large_domain.N_t,
            target_memory_fraction=0.75
        )
        
        # Check memory before
        mem_before = check_memory_usage()
        
        # Run analysis
        analyzer = LevelBPowerLawAnalyzer(use_cuda=True)
        center = [large_domain.N // 2] * 3
        start_time = time.time()
        
        result = analyzer.analyze_power_law_tail(
            field, 1.0, center, min_decades=1.5
        )
        
        elapsed_time = time.time() - start_time
        mem_after = check_memory_usage()
        
        # Verify results
        assert result["passed"], "Analysis should pass"
        assert mem_after["within_limit"], "Memory usage should be within limit"
        assert verify_block_processing_used(analyzer, field), (
            "Block processing should be used"
        )
        
        # Performance check (should complete in reasonable time)
        assert elapsed_time < 300, f"Analysis took {elapsed_time:.2f}s > 300s"
    
    def test_large_field_zone_separation(self, large_domain):
        """Test zone separation on large field."""
        field = create_large_7d_field(
            N=large_domain.N,
            N_phi=large_domain.N_phi,
            N_t=large_domain.N_t,
            target_memory_fraction=0.75
        )
        
        analyzer = LevelBZoneAnalyzer(use_cuda=True)
        center = [large_domain.N // 2] * 3
        thresholds = {
            "S_core": 1.0,
            "C_core": 1.0,
            "S_tail": 0.3,
            "C_tail": 0.3,
        }
        
        mem_before = check_memory_usage()
        result = analyzer.separate_zones(field, center, thresholds)
        mem_after = check_memory_usage()
        
        assert result["passed"], "Zone separation should pass"
        assert mem_after["within_limit"], "Memory usage should be within limit"
    
    def test_7d_structure_preservation(self, large_domain):
        """Test that 7D structure is preserved during block processing."""
        field = create_large_7d_field(
            N=large_domain.N,
            N_phi=large_domain.N_phi,
            N_t=large_domain.N_t
        )
        
        original_shape = field.shape
        assert len(original_shape) == 7, "Field should be 7D"
        
        # Process with analyzer
        analyzer = LevelBPowerLawAnalyzer(use_cuda=True)
        center = [large_domain.N // 2] * 3
        result = analyzer.analyze_power_law_tail(field, 1.0, center)
        
        # Verify structure is preserved in result
        if "radial_profile" in result:
            # Radial profile should have proper structure
            assert "r" in result["radial_profile"]
            assert "A" in result["radial_profile"]
```

### Шаг 6.2: Тестирование шага 6

**Команды:**
```bash
pytest tests/unit/test_level_b/test_performance_large_fields.py -v -s
```

**Критерии успеха:**
- Все тесты производительности проходят
- Память GPU используется эффективно
- 7D структура сохраняется

---

## Этап 7: Финальная валидация

### Цель этапа
Провести полную валидацию всех тестов уровня B.

### Шаг 7.1: Запуск всех тестов

**Команды:**
```bash
# Run all Level B tests
pytest tests/unit/test_level_b/ -v

# Run with coverage
pytest tests/unit/test_level_b/ --cov=bhlff.models.level_b --cov-report=html --cov-report=term

# Run with CUDA
BHLFF_DISABLE_CUDA=0 pytest tests/unit/test_level_b/ -v

# Run performance tests
pytest tests/unit/test_level_b/test_performance_large_fields.py -v -s
```

### Шаг 7.2: Проверка покрытия кода

**Критерии:**
- Покрытие кода ≥ 90%
- Все методы анализаторов покрыты тестами
- Все критерии из документа 7d-32 проверяются

### Шаг 7.3: Проверка соответствия критериям

**Чеклист:**
- [ ] B1: Все критерии проверяются (наклон, R², сходимость, невязка, k-space)
- [ ] B2: Все критерии проверяются (знаки, экстремумы, минимум)
- [ ] B3-S: Все критерии проверяются (точность, дисперсия)
- [ ] B3-P: Все критерии проверяются (точность, стабильность)
- [ ] B4: Все критерии проверяются (сходимость, упорядочение, согласованность)
- [ ] CUDA: Использование проверяется во всех тестах
- [ ] Блочная обработка: Используется для больших полей
- [ ] Память GPU: Используется эффективно (< 85%)
- [ ] 7D структура: Сохраняется во всех операциях

### Шаг 7.4: Генерация финального отчета

**Команда:**
```bash
pytest tests/unit/test_level_b/ -v --tb=short --junitxml=level_b_tests.xml
```

**Отчет должен содержать:**
- Статус всех тестов (PASS/FAIL)
- Покрытие кода
- Время выполнения
- Использование памяти GPU

---

## Итоговый чеклист выполнения плана

### Этап 1: B1 Power Law Tails
- [ ] Создан файл `test_B1_power_law_tails.py`
- [ ] Реализованы все критерии из документа 7d-32
- [ ] Добавлена проверка CUDA и блочной обработки
- [ ] Все тесты проходят

### Этап 2: B2 Spherical Nodes Absence
- [ ] Создан файл `test_B2_spherical_nodes_absence.py`
- [ ] Реализованы все критерии из документа 7d-32
- [ ] Все тесты проходят

### Этап 3: B3 Topological Charge
- [ ] Создан файл `test_B3S_topological_charge_synthetic.py`
- [ ] Создан файл `test_B3P_topological_charge_pde.py`
- [ ] Реализованы все критерии из документа 7d-32
- [ ] Все тесты проходят

### Этап 4: B4 Zone Separation
- [ ] Создан/обновлен файл `test_B4_zone_separation.py`
- [ ] Реализованы все критерии из документа 7d-32
- [ ] Все тесты проходят

### Этап 5: CUDA и блочная обработка
- [ ] Создан файл `cuda_test_utils.py`
- [ ] Добавлены проверки CUDA во все тесты
- [ ] Добавлены проверки использования памяти
- [ ] Все проверки проходят

### Этап 6: Производительность
- [ ] Создан файл `test_performance_large_fields.py`
- [ ] Тесты для больших полей работают
- [ ] Память GPU используется эффективно

### Этап 7: Валидация
- [ ] Все тесты проходят
- [ ] Покрытие кода ≥ 90%
- [ ] Все критерии из документа 7d-32 проверяются
- [ ] Финальный отчет сгенерирован

---

## Примечания

1. **Порядок выполнения:** Этапы можно выполнять последовательно или параллельно (кроме этапа 7, который требует завершения всех предыдущих).

2. **Тестирование:** После каждого этапа рекомендуется запускать тесты для проверки работоспособности.

3. **Коммиты:** Рекомендуется делать коммиты после завершения каждого этапа.

4. **Документация:** После завершения всех этапов обновить документацию с описанием новых тестов.

5. **CI/CD:** После завершения всех этапов интегрировать тесты в CI/CD pipeline.

