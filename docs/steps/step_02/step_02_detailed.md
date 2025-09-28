# Step 02: 7D FFT Solver for Fractional Riesz Operator

## 🎯 Goal and Mathematical Foundation

### 7D Space-Time Structure
The fundamental space-time is **7-dimensional**:
- **M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ**
- **3 spatial coordinates** (x, y, z) - conventional geometry
- **3 phase coordinates** (φ₁, φ₂, φ₃) - internal field states  
- **1 temporal coordinate** (t) - evolution dynamics

### 7D Fractional Riesz Operator
Implement high-precision spectral solver for 7D equation:
```
L_β a = μ(-Δ)^β a + λa = s(x,φ,t)
```

**Parameters:**
- `μ > 0` - diffusion coefficient
- `β ∈ (0,2)` - fractional order  
- `λ ≥ 0` - damping parameter
- `(-Δ)^β` - fractional Laplacian in 7D space

### 7D Spectral Form Solution
In 7D k-space:
```
â(k_x, k_φ, k_t) = ŝ(k_x, k_φ, k_t) / (μ|k|^(2β) + λ)
```

where:
- **k_x** - spatial wave vectors (3D)
- **k_φ** - phase wave vectors (3D) 
- **k_t** - temporal frequency
- **|k|² = |k_x|² + |k_φ|² + k_t²** - 7D wave vector magnitude

### Physical Meaning
- **Fractional order β**: determines long-range interactions in 7D
  - β → 0: local interactions
  - β = 1: classical Laplacian
  - β → 2: long-range interactions
- **Coefficient μ**: phase field diffusion speed in 7D
- **Parameter λ**: damping/dissipation of modes
- **7D structure**: enables phase coherence across spatial and phase dimensions

## 🏗️ 7D System Architecture

### 1. Core Components

#### `FFTSolver7D` (src/bhlff/core/fft/fft_solver_7d.py)
```python
class FFTSolver7D:
    """
    High-precision spectral solver for fractional Riesz operator in 7D space-time.
    
    Physical Meaning:
        Solves the fractional Laplacian equation L_β a = μ(-Δ)^β a + λa = s(x,φ,t)
        in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ, representing the evolution of
        phase field configurations with U(1)³ phase structure.
        
    Mathematical Foundation:
        Implements 7D spectral solution: â(k_x, k_φ, k_t) = ŝ(k_x, k_φ, k_t) / (μ|k|^(2β) + λ)
        where |k|² = |k_x|² + |k_φ|² + k_t² is the 7D wave vector magnitude.
    """
    
    def __init__(self, domain: 'Domain', parameters: Dict[str, Any]):
        """Initialize FFT solver with domain and physics parameters."""
        
    def solve_stationary(self, source_field: np.ndarray) -> np.ndarray:
        """Solve stationary problem L_β a = s(x)."""
        
    def solve_time_dependent(self, source_field: np.ndarray, 
                           time_params: Dict[str, Any]) -> np.ndarray:
        """Solve time-dependent problem with temporal integration."""
        
    def get_spectral_coefficients(self) -> np.ndarray:
        """Get precomputed spectral coefficients D(k) = μ|k|^(2β) + λ."""
        
    def validate_solution(self, solution: np.ndarray, 
                         source: np.ndarray) -> Dict[str, float]:
        """Validate solution quality and compute residuals."""
```

#### `FractionalLaplacian` (src/bhlff/core/fft/frac_laplacian.py)
```python
class FractionalLaplacian:
    """
    Fractional Laplacian operator (-Δ)^β implementation.
    
    Physical Meaning:
        Represents the fractional derivative operator that governs
        non-local interactions in the phase field, with β controlling
        the range of interactions from local (β→0) to long-range (β→2).
        
    Mathematical Foundation:
        In spectral space: (-Δ)^β f → |k|^(2β) * f̂(k)
        where k is the wave vector.
    """
    
    def __init__(self, domain: 'Domain', beta: float):
        """Initialize fractional Laplacian with order β."""
        
    def apply(self, field: np.ndarray) -> np.ndarray:
        """Apply fractional Laplacian (-Δ)^β to field."""
        
    def get_spectral_coefficients(self) -> np.ndarray:
        """Get spectral coefficients |k|^(2β) for all wave vectors."""
        
    def handle_special_cases(self, k_magnitude: np.ndarray) -> np.ndarray:
        """Handle special cases: k=0, β→0, β→2."""
```

#### `TimeIntegrator` (src/bhlff/core/time/integrators.py)
```python
class TimeIntegrator:
    """
    Temporal integrator for time-dependent phase field equations.
    
    Physical Meaning:
        Integrates the time evolution of phase field configurations,
        maintaining energy conservation and numerical stability.
        
    Mathematical Foundation:
        Implements various integration schemes:
        - Crank-Nicolson: implicit, second-order accurate
        - Exponential: exact for linear problems
        - Adaptive: automatic time step control
    """
    
    def __init__(self, scheme: str, domain: 'Domain', 
                 physics_params: Dict[str, Any]):
        """Initialize temporal integrator."""
        
    def integrate(self, initial_field: np.ndarray, 
                 source_field: np.ndarray, time_params: Dict[str, Any]) -> np.ndarray:
        """Integrate field evolution over time."""
        
    def compute_stability_limit(self) -> float:
        """Compute maximum stable time step."""
        
    def adaptive_step_control(self, error_estimate: float) -> float:
        """Adaptive time step control based on error estimation."""
```

#### `SpectralOperations` (src/bhlff/core/fft/spectral_ops.py)
```python
class SpectralOperations:
    """
    Core spectral operations and FFT utilities.
    
    Physical Meaning:
        Provides efficient spectral transformations and operations
        required for solving phase field equations in k-space.
    """
    
    def __init__(self, domain: 'Domain', precision: str = "float64"):
        """Initialize spectral operations with domain and precision."""
        
    def setup_fft_plan(self, plan_type: str = "MEASURE") -> None:
        """Setup optimized FFT plan for efficiency."""
        
    def forward_fft(self, field: np.ndarray) -> np.ndarray:
        """Forward FFT with proper normalization."""
        
    def inverse_fft(self, spectral_field: np.ndarray) -> np.ndarray:
        """Inverse FFT with proper normalization."""
        
    def compute_wave_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute wave vectors kx, ky, kz for the domain."""
```

### 2. Критические требования к реализации

#### Точность вычислений
- **Обязательно**: `float64` для всех вычислений
- **Нормировка FFT**: строго фиксированная
  ```python
  # Прямое FFT
  â(m) = Σ_x a(x) * exp(-i k(m)·x) * Δ³
  
  # Обратное FFT  
  a(x) = (1/L³) * Σ_m â(m) * exp(i k(m)·x)
  ```
- **Контроль ошибок**: невязка ≤ 10⁻¹²
- **Валидация**: ортогональность невязки к решению

#### Производительность
- **FFT планы**: оптимизированные с типом "MEASURE"
- **Векторизация**: NumPy операции для всех вычислений
- **Кэширование**: спектральные коэффициенты, FFT планы
- **Параллелизация**: опционально (не влияет на точность)

#### Обработка особых случаев
- **k=0 мода**: D(0) = λ (если λ > 0)
- **λ=0 с ŝ(0)≠0**: должна быть ошибка (несовместимость)
- **Граничные случаи β**: β→0, β→2
- **Переполнение**: контроль при вычислении |k|^(2β)

## 📊 Алгоритмы и процедуры

### 1. Стационарный решатель (спектральный, прямой)

```python
def solve_stationary(self, source_field: np.ndarray) -> np.ndarray:
    """
    Solve stationary problem L_β a = s(x) using spectral method.
    
    Algorithm:
    1. Compute ŝ(m) = FFT[s(x)]
    2. For all m: D(m) = μ|k(m)|^(2β) + λ
    3. Handle k=0: D(0) = λ (if λ > 0)
    4. Compute â(m) = ŝ(m) / D(m)
    5. Transform back: a(x) = iFFT[â(m)]
    6. Validate: compute residual r = L_β a - s
    """
```

### 2. Временной решатель

```python
def solve_time_dependent(self, source_field: np.ndarray, 
                        time_params: Dict[str, Any]) -> np.ndarray:
    """
    Solve time-dependent problem with temporal integration.
    
    Algorithm:
    1. Setup time integrator (CN or exponential)
    2. Compute stability limit: dt ≤ c / (ν k_max^(2β) + λ)
    3. Integrate over time with adaptive step control
    4. Monitor energy conservation and stability
    """
```

### 3. Валидация решения

```python
def validate_solution(self, solution: np.ndarray, 
                     source: np.ndarray) -> Dict[str, float]:
    """
    Validate solution quality and compute metrics.
    
    Metrics:
    - Relative residual: ||r||₂ / ||s||₂ ≤ 10⁻¹²
    - Orthogonality: Re(Σ_m â*(m) r̂(m)) ≈ 0
    - Energy balance: |E_out - E_in| / E_in ≤ 1-3%
    """
```

## 🧪 Тесты и валидация

### 1. Аналитические тесты

#### A0.1. Плоская волна (стационар)
```python
def test_plane_wave_stationary():
    """
    Test: s(x) = exp(i k₀·x) → a(x) = exp(i k₀·x) / D(k₀)
    
    Parameters:
    - k₀ = (2π/L) * (1, 1, 1)
    - μ = 1.0, β = 1.0, λ = 0.1
    
    Criteria:
    - Amplitude error ≤ 10⁻¹²
    - Phase error ≤ 10⁻¹²
    - Anisotropy ≤ 10⁻¹² for same |k|
    """
```

#### A0.2. Многочастотный источник
```python
def test_multifrequency_source():
    """
    Test: s(x) = Σᵢ Aᵢ exp(i kᵢ·x) → superposition principle
    
    Parameters:
    - Multiple frequencies with different amplitudes
    - Check linearity and absence of aliasing
    
    Criteria:
    - Superposition principle holds
    - No aliasing artifacts
    - Frequency resolution maintained
    """
```

### 2. Численные тесты

#### A0.3. Сходимость по сетке
```python
def test_grid_convergence():
    """
    Test convergence with increasing grid resolution.
    
    Parameters:
    - N = 128 → 256 → 512
    - Fixed domain size L
    - Point source at center
    
    Criteria:
    - Solution converges as N increases
    - Convergence rate ≥ 2 (spectral accuracy)
    - No numerical artifacts
    """
```

#### A0.4. Энергетический баланс
```python
def test_energy_balance():
    """
    Test energy conservation and balance.
    
    Parameters:
    - Various source configurations
    - Different β values
    
    Criteria:
    - Energy balance ≤ 1-3%
    - Residual ||r||₂ / ||s||₂ ≤ 10⁻¹²
    - Orthogonality condition satisfied
    """
```

### 3. Граничные случаи

#### A0.5. Особые случаи
```python
def test_special_cases():
    """
    Test handling of special cases.
    
    Cases:
    - λ = 0 with ŝ(0) = 0 (valid)
    - λ = 0 with ŝ(0) ≠ 0 (should raise error)
    - β → 0 (local limit)
    - β → 2 (long-range limit)
    - k = 0 mode handling
    
    Criteria:
    - Correct error handling
    - Proper limit behavior
    - No numerical instabilities
    """
```

## ⚙️ Конфигурация и параметры

### Конфигурационный файл (configs/level_a/fft_solver.json)
```json
{
  "domain": {
    "L": 1.0,
    "N": 256,
    "dimensions": 3,
    "periodic": true
  },
  "physics": {
    "mu": 1.0,
    "beta": 1.0,
    "lambda": 0.0,
    "nu": 1.0
  },
  "solver": {
    "precision": "float64",
    "fft_plan": "MEASURE",
    "tolerance": 1e-12,
    "max_iterations": 1000
  },
  "time_integration": {
    "scheme": "crank_nicolson",
    "adaptive": true,
    "safety_factor": 0.8,
    "min_dt": 1e-6,
    "max_dt": 1e-2
  },
  "output": {
    "save_fields": true,
    "save_spectra": true,
    "save_analysis": true,
    "format": "hdf5"
  }
}
```

### Параметры по умолчанию
```python
DEFAULT_PARAMETERS = {
    "domain": {
        "L": 8.0 * np.pi,  # Recommended for 3D
        "N": 256,          # Good balance of accuracy/speed
        "dimensions": 3
    },
    "physics": {
        "mu": 1.0,         # Diffusion coefficient
        "beta": 1.0,       # Fractional order
        "lambda": 0.0,     # Damping parameter
        "nu": 1.0          # Viscosity (for time-dependent)
    },
    "solver": {
        "precision": "float64",
        "fft_plan": "MEASURE",
        "tolerance": 1e-12
    }
}
```

## 🔧 Интерфейс и API

### Основной интерфейс
```python
# Инициализация
solver = FFTSolver3D(domain, parameters)

# Стационарное решение
solution = solver.solve_stationary(source_field)

# Временное решение
time_evolution = solver.solve_time_dependent(
    source_field, 
    time_params={
        "t_final": 1.0,
        "dt": 0.01,
        "scheme": "crank_nicolson"
    }
)

# Валидация
metrics = solver.validate_solution(solution, source_field)
```

### Расширенный интерфейс
```python
# Получение спектральных коэффициентов
spectral_coeffs = solver.get_spectral_coefficients()

# Настройка точности
solver.set_precision("float64")
solver.set_tolerance(1e-12)

# Информация о решателе
info = solver.get_solver_info()
```

## 🚀 7D-специфичные оптимизации

### Управление памятью для O(N^7) масштабирования
```python
class MemoryManager7D:
    """
    Менеджер памяти для 7D вычислений.
    
    Physical Meaning:
        Управляет памятью для 7D фазовых полей, которые масштабируются
        как O(N^7), требуя специальных стратегий управления памятью.
        
    Mathematical Foundation:
        - Блочная декомпозиция: разбиение поля на блоки
        - Ленивая загрузка: загрузка данных по требованию
        - Компрессия: сжатие неактивных блоков
    """
    
    def __init__(self, domain_shape: Tuple[int, ...], max_memory_gb: float = 8.0):
        """
        Инициализация менеджера памяти.
        
        Args:
            domain_shape: Размеры 7D области
            max_memory_gb: Максимальное использование памяти в ГБ
        """
        self.domain_shape = domain_shape
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.block_size = self._calculate_optimal_block_size()
        self.active_blocks = {}
        self.compressed_blocks = {}
    
    def _calculate_optimal_block_size(self) -> Tuple[int, ...]:
        """
        Вычисление оптимального размера блока.
        
        Physical Meaning:
            Определяет размер блока, который помещается в доступную
            память и обеспечивает эффективную обработку.
        """
        # Расчет на основе доступной памяти и размеров области
        total_elements = np.prod(self.domain_shape)
        elements_per_gb = self.max_memory_bytes // (8 * 4)  # float64 + complex128
        
        # Блочная декомпозиция
        block_elements = min(total_elements, elements_per_gb // 4)
        block_size = int(block_elements ** (1/7))
        
        return tuple([block_size] * 7)
```

### Многомерные FFT планы
```python
class FFTPlan7D:
    """
    Оптимизированные FFT планы для 7D вычислений.
    
    Physical Meaning:
        Предвычисленные планы FFT для эффективного выполнения
        спектральных операций в 7D пространстве.
        
    Mathematical Foundation:
        - Планирование FFT: предвычисление оптимальных алгоритмов
        - Кэширование планов: переиспользование для повторных операций
        - Блочная обработка: FFT по блокам для больших полей
    """
    
    def __init__(self, domain_shape: Tuple[int, ...], precision: str = "float64"):
        """
        Инициализация FFT планов.
        
        Args:
            domain_shape: Размеры 7D области
            precision: Точность вычислений
        """
        self.domain_shape = domain_shape
        self.precision = precision
        self.plans = {}
        self._setup_fft_plans()
    
    def _setup_fft_plans(self) -> None:
        """
        Настройка FFT планов для 7D операций.
        
        Physical Meaning:
            Создает оптимизированные планы для всех необходимых
            FFT операций в 7D пространстве.
        """
        # Планы для прямого и обратного FFT
        self.plans['forward'] = self._create_fft_plan('forward')
        self.plans['inverse'] = self._create_fft_plan('inverse')
        
        # Планы для блочной обработки
        self.plans['block_forward'] = self._create_block_fft_plan('forward')
        self.plans['block_inverse'] = self._create_block_fft_plan('inverse')
    
    def execute_fft(self, field: np.ndarray, direction: str = 'forward') -> np.ndarray:
        """
        Выполнение оптимизированного FFT.
        
        Physical Meaning:
            Выполняет FFT операцию с использованием предвычисленных
            планов для максимальной эффективности.
            
        Args:
            field: 7D поле для преобразования
            direction: Направление ('forward' или 'inverse')
            
        Returns:
            Преобразованное поле
        """
        plan = self.plans[direction]
        return plan.execute(field)
```

### Кэширование спектральных коэффициентов
```python
class SpectralCoefficientCache:
    """
    Кэш для спектральных коэффициентов фракционного оператора.
    
    Physical Meaning:
        Кэширует спектральные коэффициенты μ|k|^(2β) + λ для
        повторного использования в вычислениях.
        
    Mathematical Foundation:
        - Предвычисление: коэффициенты зависят только от параметров
        - Кэширование: переиспользование для одинаковых параметров
        - Инвалидация: обновление при изменении параметров
    """
    
    def __init__(self, max_cache_size: int = 100):
        """
        Инициализация кэша.
        
        Args:
            max_cache_size: Максимальный размер кэша
        """
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.access_count = {}
    
    def get_coefficients(self, mu: float, beta: float, lambda_param: float, 
                        domain_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Получение спектральных коэффициентов из кэша.
        
        Physical Meaning:
            Возвращает спектральные коэффициенты для фракционного
            оператора, используя кэш для оптимизации.
            
        Args:
            mu: Коэффициент диффузии
            beta: Фракционный порядок
            lambda_param: Параметр затухания
            domain_shape: Размеры области
            
        Returns:
            Спектральные коэффициенты
        """
        cache_key = (mu, beta, lambda_param, domain_shape)
        
        if cache_key in self.cache:
            self.access_count[cache_key] += 1
            return self.cache[cache_key]
        
        # Вычисление новых коэффициентов
        coefficients = self._compute_coefficients(mu, beta, lambda_param, domain_shape)
        
        # Добавление в кэш
        self._add_to_cache(cache_key, coefficients)
        
        return coefficients
    
    def _compute_coefficients(self, mu: float, beta: float, lambda_param: float,
                            domain_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Вычисление спектральных коэффициентов.
        
        Physical Meaning:
            Вычисляет спектральные коэффициенты μ|k|^(2β) + λ
            для фракционного оператора Рисса.
        """
        # Создание волновых векторов
        k_vectors = []
        for i, n in enumerate(domain_shape):
            k = np.fft.fftfreq(n, 1.0/n)
            k_vectors.append(k)
        
        # Создание сетки волновых векторов
        K_mesh = np.meshgrid(*k_vectors, indexing='ij')
        k_magnitude = np.sqrt(sum(K**2 for K in K_mesh))
        
        # Вычисление коэффициентов
        coefficients = mu * (k_magnitude ** (2 * beta)) + lambda_param
        
        # Обработка k=0 моды
        if lambda_param == 0:
            coefficients[tuple([0] * len(domain_shape))] = 1.0
        
        return coefficients
```

## 📈 Критерии готовности

### Обязательные требования
- [ ] Реализован класс `FFTSolver3D` с полным API
- [ ] Реализован `FractionalLaplacian` с оптимизациями
- [ ] Реализованы временные интеграторы (CN, экспоненциальная)
- [ ] Реализован `MemoryManager7D` для управления памятью
- [ ] Реализован `FFTPlan7D` для оптимизированных FFT
- [ ] Реализован `SpectralCoefficientCache` для кэширования
- [ ] Все аналитические тесты A0.1-A0.5 проходят
- [ ] Численные тесты показывают правильную сходимость
- [ ] Обработка особых случаев работает корректно
- [ ] Производительность соответствует требованиям для 7D
- [ ] Документация и примеры созданы

### Стандартная схема нормализации FFT для 7D
```python
class FFTNormalization7D:
    """
    Стандартная схема нормализации FFT для 7D вычислений.
    
    Physical Meaning:
        Определяет стандартную схему нормализации для FFT операций
        в 7D пространстве, обеспечивая консистентность между
        прямым и обратным преобразованиями.
        
    Mathematical Foundation:
        - Прямое FFT: â(k) = Σ a(x) exp(-2πi k·x/N)
        - Обратное FFT: a(x) = (1/N^7) Σ â(k) exp(2πi k·x/N)
        - Нормализация: сохранение энергии в пространстве и времени
    """
    
    @staticmethod
    def forward_fft(field: np.ndarray) -> np.ndarray:
        """
        Прямое FFT с стандартной нормализацией.
        
        Physical Meaning:
            Выполняет прямое FFT преобразование с нормализацией,
            сохраняющей физический смысл спектральных компонент.
            
        Args:
            field: 7D поле в реальном пространстве
            
        Returns:
            Спектральное представление поля
        """
        return np.fft.fftn(field, norm='ortho')
    
    @staticmethod
    def inverse_fft(spectral_field: np.ndarray) -> np.ndarray:
        """
        Обратное FFT с стандартной нормализацией.
        
        Physical Meaning:
            Выполняет обратное FFT преобразование с нормализацией,
            восстанавливающей поле в реальном пространстве.
            
        Args:
            spectral_field: Спектральное представление поля
            
        Returns:
            Поле в реальном пространстве
        """
        return np.fft.ifftn(spectral_field, norm='ortho')
    
    @staticmethod
    def energy_conservation_check(real_field: np.ndarray, 
                                spectral_field: np.ndarray) -> float:
        """
        Проверка сохранения энергии при FFT.
        
        Physical Meaning:
            Проверяет, что энергия поля сохраняется при FFT
            преобразовании (теорема Парсеваля).
            
        Mathematical Foundation:
            Σ |a(x)|² = (1/N^7) Σ |â(k)|²
            
        Returns:
            Относительная ошибка сохранения энергии
        """
        real_energy = np.sum(np.abs(real_field)**2)
        spectral_energy = np.sum(np.abs(spectral_field)**2) / np.prod(real_field.shape)
        
        return abs(real_energy - spectral_energy) / real_energy
```

### Критерии качества
- **Точность**: невязка ≤ 10⁻¹²
- **Сходимость**: порядок ≥ 2 для спектральных методов
- **Стабильность**: энергетический баланс ≤ 3%
- **Производительность**: время решения < 1 сек для N=256
- **Надежность**: корректная обработка всех граничных случаев
- **Нормализация FFT**: сохранение энергии ≤ 10⁻¹⁵

### Метрики валидации
```python
VALIDATION_METRICS = {
    "residual_norm": 1e-12,      # ||r||₂ / ||s||₂
    "orthogonality": 1e-12,      # Re(Σ â* r̂)
    "energy_balance": 0.03,      # |E_out - E_in| / E_in
    "convergence_rate": 2.0,     # Spectral accuracy
    "anisotropy": 1e-12,         # Max relative deviation
    "stability_factor": 0.8      # Time step safety
}
```

## 🚀 Следующие шаги

После завершения Step 02:
1. **Step 03**: Создание временных интеграторов для динамических задач
2. **Step 04**: Реализация тестов уровня A (валидация решателей)
3. **Step 05**: Интеграция с моделями уровня B (фундаментальные свойства)

## 📚 Дополнительные ресурсы

### Теоретические основы
- Фракционный лапласиан и оператор Рисса
- Спектральные методы в периодических областях
- Численная стабильность и сходимость

### Практические аспекты
- Оптимизация FFT операций
- Обработка особых случаев в спектральных методах
- Валидация и тестирование численных решателей

### 7D-специфичные оптимизации
- **Управление памятью**: Стратегии для O(N^7) масштабирования
- **Многомерные FFT планы**: Оптимизация для 7D вычислений
- **Блочная обработка**: Разбиение больших полей на блоки
- **Кэширование спектральных коэффициентов**: Предвычисление для повторных операций

### Связанные модули
- `Domain`: управление вычислительной областью
- `Field`: представление полей и операций
- `Parameters`: управление физическими параметрами
- `Validation`: тесты и метрики качества
