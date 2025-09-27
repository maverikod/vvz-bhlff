# Step 06: Детализированное задание - Модели границ и ячеек уровня C

## Физическая основа и теоретический контекст

### Концептуальная основа уровня C

Уровень C представляет собой критически важный переход от фундаментальных свойств фазового поля (уровень B) к сложным многомодовым структурам (уровень D). На этом уровне изучается, как **границы и контрасты** в безынтервальном фазовом поле (БВП) создают **резонансные структуры** и **ячейки**, которые являются предшественниками частицоподобных объектов.

### Ключевые физические принципы

#### 1. Принцип последовательных резонаторов
Согласно теории 7D фазового поля, **частицы возникают как вложенные резонаторы** фазового поля. Каждый резонатор представляет собой:
- **Ядро** - область с высокой фазовой когерентностью
- **Переходная зона** - область изменения параметров среды
- **Хвост** - внешняя область с линейной динамикой

#### 2. Адмиттанса и резонансные моды
**Адмиттанса Y(ω)** - это комплексная функция, описывающая отклик системы на внешнее возбуждение:
```
Y(ω) = I(ω)/V(ω) = G(ω) + iB(ω)
```
где:
- G(ω) - активная проводимость (потери)
- B(ω) - реактивная проводимость (накопление энергии)

#### 3. Квенч-память и пиннинг
**Квенч** - это фазовый срыв, возникающий при превышении локального порога устойчивости. После квенча система "запоминает" это событие через:
- Изменение локальных параметров среды
- Создание диссипативных центров
- Формирование "замороженных" структур

## Детализированные математические модели

### 1. Модель огибающей с граничными условиями

#### Базовое уравнение
```
∇·(κ(|a|) ∇a) + k₀²χ(|a|) a = s(x), k₀ = ω₀/c_φ
```

где:
- **κ(|a|)** - "жёсткость" среды, зависящая от амплитуды
- **χ(|a|) = χ' + iχ''** - комплексная восприимчивость
  - χ' - накопление энергии
  - χ'' - потери и квенч-память
- **s(x)** - источник возбуждения

#### Энергетический функционал
```
E[a] = ∫_Ω (κ|∇a|² + k₀²χ'|a|²) dV + E_dissipative
```

где диссипативная часть включает:
- Потери на квенчах: ∫_Ω χ''(|a|)|a|² dV
- Память среды: ∫_Ω γ_memory |a|² dV

### 2. Модель сферической границы

#### Геометрия
- **Внутренняя область**: r < R_inner, параметры (κ₁, χ₁)
- **Оболочка**: R_inner < r < R_outer, параметры (κ₂, χ₂)
- **Внешняя область**: r > R_outer, параметры (κ₃, χ₃)

#### Контраст адмиттансы
```
η = |ΔY|/⟨Y⟩ = |Y_inside - Y_outside|/((Y_inside + Y_outside)/2)
```

#### Условия сшивки
На границах r = R_inner, R_outer:
- Непрерывность поля: a_inner = a_outer
- Непрерывность потока: κ₁(∂a/∂r)_inner = κ₂(∂a/∂r)_outer

### 3. ABCD модель для цепочки резонаторов

#### Матрица передачи
Для каждого слоя ℓ:
```
T_ℓ = [A_ℓ  B_ℓ]
      [C_ℓ  D_ℓ]
```

где элементы вычисляются через:
- **A_ℓ, D_ℓ**: коэффициенты прохождения
- **B_ℓ, C_ℓ**: коэффициенты отражения

#### Системная матрица
```
T_total = T_1 × T_2 × ... × T_N
```

#### Резонансные условия
Резонанс возникает при:
```
det(T_total - I) = 0
```

### 4. Модель квенч-памяти

#### Диссипативно-памятный член
```
∂a/∂t = L[a] + Γ_memory[a]
```

где:
```
Γ_memory[a] = -γ_memory ∫_0^t K(t-τ) a(τ) dτ
```

#### Ядро памяти (модель Дебая)
```
K(t) = (1/τ) exp(-t/τ)
```

#### Параметры памяти
- **γ_memory**: сила памяти (0 ≤ γ ≤ 1)
- **τ**: время релаксации памяти

## Детализированные спецификации тестов

### C1. Одна "стенка" (контраст адмиттансы)

#### Цель
Продемонстрировать **рождение ячейки/псевдорезонатора** при наличии контраста адмиттансы на сферической границе.

#### Физическая постановка
- **Геометрия**: Сферическая оболочка с радиусом R = 6π
- **Толщина оболочки**: 3 ячейки сетки
- **Контраст**: η = |ΔY|/⟨Y⟩ варьируется от 0 до 0.5
- **Параметры**: β ∈ {0.8, 1.0, 1.2}, ν = 1, λ = 0

#### Математическая постановка
```
∇·(κ(r) ∇a) + k₀²χ(r) a = s(x)
```

где:
- κ(r) = κ₁ для r < R, κ₂ для r > R
- χ(r) = χ₁ для r < R, χ₂ для r > R
- Контраст: η = |κ₂/χ₂ - κ₁/χ₁|/((κ₂/χ₂ + κ₁/χ₁)/2)

#### Наблюдаемые величины
1. **Адмиттанса Y(ω)**:
   ```
   Y(ω) = ∫_Ω a*(x) s(x) dV / ∫_Ω |a(x)|² dV
   ```

2. **Радиальные профили A(r)**:
   ```
   A(r) = (1/4π) ∫_S(r) |a(x)|² dS
   ```

3. **Резонансные моды**:
   - Частоты ω_n резонансов
   - Добротности Q_n = ω_n/Δω_n

#### Критерии приёмки
1. **При η = 0**: Нет пиков ≥ 8 дБ в Y(ω)
2. **При η ≥ 0.1**: 
   - Появляется ≥ 1 пик в Y(ω)
   - Локальные максимумы A(r) между ядром и оболочкой
   - Порог η* для появления первой моды определен
3. **Пассивность**: Re Y(ω) ≥ 0 для всех ω
4. **Сходимость**: Ошибки ω_n ≤ 3%, Q_n ≤ 10% при увеличении N

#### Алгоритм реализации
```python
def test_C1_single_wall():
    # 1. Создание геометрии
    domain = create_spherical_domain(L=32π, N=384)
    boundary = create_spherical_boundary(center, radius=6π, thickness=3)
    
    # 2. Свип по контрасту
    for eta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        # 3. Свип по частоте
        for omega in logspace(0.05, 5.0, 300):
            # 4. Решение стационарной задачи
            field = solve_stationary(domain, boundary, omega, eta)
            
            # 5. Вычисление адмиттансы
            Y = compute_admittance(field, source)
            
            # 6. Анализ профилей
            A_r = compute_radial_profiles(field)
            
            # 7. Поиск резонансов
            resonances = find_resonances(Y, threshold=8)  # дБ
    
    # 8. Анализ результатов
    return analyze_resonance_birth(eta_list, resonance_data)
```

### C2. Цепочка резонаторов (ABCD)

#### Цель
Продемонстрировать **принцип последовательных резонаторов** и согласование с ABCD-моделью.

#### Физическая постановка
- **Геометрия**: 2-5 вложенных сферических оболочек
- **Радиусы**: R/π = 6, 9, 12, 15, 18
- **Толщины**: 3 ячейки каждая
- **Контрасты**: Разные η для каждого слоя

#### Математическая постановка
```
T_total = T_1 × T_2 × ... × T_5
```

где для каждого слоя ℓ:
```
T_ℓ = [cos(k_ℓ ΔR_ℓ)  (1/k_ℓ)sin(k_ℓ ΔR_ℓ)]
      [-k_ℓ sin(k_ℓ ΔR_ℓ)  cos(k_ℓ ΔR_ℓ)]
```

#### ABCD анализ
1. **Вычисление матриц передачи** для каждого слоя
2. **Системная матрица**: T_total = ∏ T_ℓ
3. **Резонансные условия**: det(T_total - I) = 0
4. **Сравнение с численными результатами**

#### Критерии приёмки
1. **≥ 3 пика** в Y(ω) ≥ 8 дБ
2. **Ошибки ABCD-прогноза**:
   - Общие ошибки ≤ 10%
   - Ошибки на пиках ≤ 5%
   - Частоты пиков ≤ 5%
   - Добротности Q ≤ 10%
3. **Пассивность**: Re Y(ω) ≥ 0

#### Алгоритм реализации
```python
def test_C2_chain_abcd():
    # 1. Создание цепочки резонаторов
    shells = create_resonator_chain(radii=[6π, 9π, 12π, 15π, 18π])
    
    # 2. ABCD модель
    abcd_model = ABCDModel(shells)
    
    # 3. Численное решение
    for omega in logspace(0.05, 6.0, 360):
        field = solve_stationary(domain, shells, omega)
        Y_numerical = compute_admittance(field)
        
        # 4. ABCD прогноз
        Y_abcd = abcd_model.predict(omega)
        
        # 5. Сравнение
        error = compute_error(Y_numerical, Y_abcd)
    
    # 6. Анализ системных мод
    system_modes = abcd_model.find_system_modes()
    numerical_modes = extract_numerical_modes(Y_numerical)
    
    return compare_modes(system_modes, numerical_modes)
```

### C3. Квенч-память/пиннинг

#### Цель
Продемонстрировать **статичность огибающей** относительно дефекта при наличии квенч-памяти.

#### Физическая постановка
- **Режим**: Временной
- **Источник**: Биения двух тонов с ω₁, ω₂
- **Память**: Свип по γ ∈ [0, γ_max], τ ∈ [0.3, 3.0]
- **Окно усреднения**: Δt_avg = 10/(ω₁ + ω₂)

#### Математическая постановка
```
∂a/∂t = L[a] + Γ_memory[a] + s(x,t)
```

где:
```
Γ_memory[a] = -γ ∫_0^t (1/τ) exp(-(t-τ)/τ) a(τ) dτ
s(x,t) = s₁(x) e^(-iω₁t) + s₂(x) e^(-iω₂t)
```

#### Наблюдаемые величины
1. **Эффективная интенсивность**:
   ```
   I_eff(x,t) = MA{|a(x,t)|²}
   ```

2. **Скорость дрейфа ячеек**:
   ```
   v_cell = Δx_max / Δt
   ```
   где Δx_max - смещение максимума кросс-корреляции

3. **Кросс-корреляция**:
   ```
   C(t,Δt) = ∫ I_eff(x,t) I_eff(x,t+Δt) dx
   ```

#### Критерии приёмки
1. **При γ = 0**: v_cell ≈ Δω/|Δk|, отклонение ≤ 10%
2. **При γ ≥ γ***: v_cell ≤ 10⁻³ L/T₀ (заморожено)
3. **Жаккар-индекс**: ≥ 0.95 на длинных окнах
4. **Порог γ***: Определен в свипе

#### Алгоритм реализации
```python
def test_C3_quench_memory():
    # 1. Создание геометрии с памятью
    domain = create_domain_with_memory()
    
    # 2. Свип по параметрам памяти
    for gamma in [0.0, 0.2, 0.4, 0.6, 0.8]:
        for tau in [0.5, 1.0, 2.0]:
            # 3. Временная эволюция
            time_series = evolve_in_time(domain, gamma, tau)
            
            # 4. Анализ дрейфа
            I_eff = compute_effective_intensity(time_series)
            v_cell = compute_cell_drift(I_eff)
            
            # 5. Кросс-корреляция
            xcorr = compute_cross_correlation(I_eff)
            
            # 6. Жаккар-индекс
            jaccard = compute_jaccard_index(I_eff)
    
    # 7. Определение порога заморозки
    gamma_star = find_freezing_threshold(gamma_list, v_cell_list)
    
    return analyze_pinning_effects(gamma_star, v_cell_data)
```

### C4. Биения двух близких мод

#### Цель
Измерить **скорость дрейфа решётки** и её подавление пиннингом.

#### Физическая постановка
- **Частоты**: ω₁,₂ = ω₀ ± Δω/2
- **Относительная разность**: Δω/ω₀ ∈ {0.02, 0.05}
- **Режимы**: Без слоёв (фон) и со слоями
- **Амплитуды**: Одинаковые для обеих мод

#### Математическая постановка
```
s(x,t) = s₁(x) e^(-iω₁t) + s₂(x) e^(-iω₂t)
```

где:
- s₁(x) = s₂(x) = g_σ - ḡ (одинаковые профили)
- |ω₂ - ω₁| = Δω << ω₁,₂

#### Теоретическая скорость дрейфа
```
v_cell^pred = Δω / |k₂ - k₁|
```

где k₁,₂ - волновые векторы соответствующих мод.

#### Критерии приёмки
1. **Без пиннинга**: |v_cell^num - v_cell^pred|/v_cell^pred ≤ 10%
2. **С пиннингом**: v_cell^num/v_cell^pred ≤ 0.1 (подавление ≥10×)
3. **Согласованность**: Результаты C4 согласуются с C3

#### Алгоритм реализации
```python
def test_C4_mode_beating():
    # 1. Настройка частот
    omega_0 = 1.0
    delta_omega_ratios = [0.02, 0.05]
    
    for delta_omega_ratio in delta_omega_ratios:
        omega_1 = omega_0 - delta_omega_ratio * omega_0 / 2
        omega_2 = omega_0 + delta_omega_ratio * omega_0 / 2
        
        # 2. Тест без пиннинга (фон)
        field_background = create_dual_mode_field(omega_1, omega_2)
        time_series_bg = evolve_in_time(field_background)
        v_cell_bg = compute_drift_velocity(time_series_bg)
        
        # 3. Тест с пиннингом
        field_pinned = create_dual_mode_field_with_pinning(omega_1, omega_2)
        time_series_pin = evolve_in_time(field_pinned)
        v_cell_pin = compute_drift_velocity(time_series_pin)
        
        # 4. Теоретическое значение
        k_1, k_2 = compute_wave_vectors(omega_1, omega_2)
        v_cell_theory = abs(omega_2 - omega_1) / abs(k_2 - k_1)
        
        # 5. Анализ ошибок
        error_bg = abs(v_cell_bg - v_cell_theory) / v_cell_theory
        suppression = v_cell_pin / v_cell_bg
    
    return analyze_beating_results(error_bg, suppression)
```

## Детализированные требования к реализации

### 1. Структура модулей

#### Основной модуль: `src/bhlff/models/level_c/`
```
level_c/
├── __init__.py
├── boundaries.py          # Модели границ
├── resonators.py          # Модели резонаторов
├── memory.py              # Квенч-память и пиннинг
├── beating.py             # Биения мод
├── abcd_model.py          # ABCD модель
└── analysis.py            # Анализ результатов
```

#### Модуль границ: `boundaries.py`
```python
class BoundaryGeometry:
    """
    Базовый класс для геометрий границ.
    
    Physical Meaning:
        Определяет интерфейс для различных геометрий границ,
        поддерживающих сложные формы и контрасты адмиттанса.
    """
    
    def compute_admittance_contrast(self, frequency: float) -> float:
        """Вычисление контраста адмиттанса."""
        pass
    
    def apply_boundary_conditions(self, field: np.ndarray) -> np.ndarray:
        """Применение граничных условий."""
        pass
    
    def find_resonance_modes(self, frequency_range: Tuple[float, float]) -> List[ResonanceMode]:
        """Поиск резонансных мод."""
        pass

class SphericalBoundary(BoundaryGeometry):
    """
    Сферическая граница с контрастом адмиттанса.
    
    Physical Meaning:
        Представляет сферическую оболочку с различными
        материальными свойствами, создающую контраст
        адмиттанса и резонансные структуры.
    """
    
    def __init__(self, center: np.ndarray, radius: float, 
                 thickness: float, contrast: float):
        """Инициализация сферической границы."""
        self.center = np.array(center)
        self.radius = radius
        self.thickness = thickness
        self.contrast = contrast
    
    def compute_admittance_contrast(self, frequency: float) -> float:
        """Вычисление контраста адмиттанса на заданной частоте."""
        return self.contrast * (1 + 0.1 * frequency)  # Частотная зависимость
    
    def apply_boundary_conditions(self, field: np.ndarray) -> np.ndarray:
        """Применение граничных условий к полю."""
        # Создание маски для сферической оболочки
        mask = self._create_spherical_shell_mask(field.shape)
        
        # Применение контраста
        field_with_boundary = field.copy()
        field_with_boundary[mask] *= (1 + self.contrast)
        
        return field_with_boundary

class CylindricalBoundary(BoundaryGeometry):
    """
    Цилиндрическая граница с контрастом адмиттанса.
    
    Physical Meaning:
        Представляет цилиндрическую оболочку для изучения
        анизотропных резонансных структур.
    """
    
    def __init__(self, center: np.ndarray, radius: float, height: float,
                 thickness: float, contrast: float, axis: str = 'z'):
        """Инициализация цилиндрической границы."""
        self.center = np.array(center)
        self.radius = radius
        self.height = height
        self.thickness = thickness
        self.contrast = contrast
        self.axis = axis
    
    def apply_boundary_conditions(self, field: np.ndarray) -> np.ndarray:
        """Применение граничных условий для цилиндра."""
        mask = self._create_cylindrical_shell_mask(field.shape)
        field_with_boundary = field.copy()
        field_with_boundary[mask] *= (1 + self.contrast)
        return field_with_boundary

class EllipsoidalBoundary(BoundaryGeometry):
    """
    Эллипсоидальная граница с контрастом адмиттанса.
    
    Physical Meaning:
        Представляет эллипсоидальную оболочку для изучения
        деформированных резонансных структур.
    """
    
    def __init__(self, center: np.ndarray, semi_axes: np.ndarray,
                 thickness: float, contrast: float):
        """Инициализация эллипсоидальной границы."""
        self.center = np.array(center)
        self.semi_axes = np.array(semi_axes)
        self.thickness = thickness
        self.contrast = contrast
    
    def apply_boundary_conditions(self, field: np.ndarray) -> np.ndarray:
        """Применение граничных условий для эллипсоида."""
        mask = self._create_ellipsoidal_shell_mask(field.shape)
        field_with_boundary = field.copy()
        field_with_boundary[mask] *= (1 + self.contrast)
        return field_with_boundary

class ComplexBoundary(BoundaryGeometry):
    """
    Сложная граница из нескольких геометрий.
    
    Physical Meaning:
        Объединяет несколько границ для создания сложных
        резонансных структур с множественными модами.
    """
    
    def __init__(self, boundaries: List[BoundaryGeometry]):
        """Инициализация сложной границы."""
        self.boundaries = boundaries
    
    def apply_boundary_conditions(self, field: np.ndarray) -> np.ndarray:
        """Применение граничных условий для всех границ."""
        result_field = field.copy()
        for boundary in self.boundaries:
            result_field = boundary.apply_boundary_conditions(result_field)
        return result_field
```

#### Модуль резонаторов: `resonators.py`
```python
class ResonatorChain:
    """
    Chain of nested spherical resonators.
    
    Physical Meaning:
        Represents a system of nested spherical shells that form
        a chain of resonators, enabling complex resonance patterns
        and mode coupling effects.
    """
    
    def __init__(self, shells: List[SphericalShell]):
        """Initialize resonator chain."""
        
    def compute_system_admittance(self, frequency: float) -> complex:
        """Compute total system admittance."""
        
    def find_system_modes(self, frequency_range: Tuple[float, float]) -> List[SystemMode]:
        """Find system resonance modes."""
        
    def analyze_mode_coupling(self, modes: List[SystemMode]) -> ModeCouplingAnalysis:
        """Analyze coupling between modes."""
```

#### Модуль памяти: `memory.py`
```python
class QuenchMemory:
    """
    Quench memory and pinning effects.
    
    Physical Meaning:
        Implements memory effects from quench events, where the system
        "remembers" past phase transitions and creates pinning centers
        that suppress field drift and stabilize patterns.
    """
    
    def __init__(self, gamma: float, tau: float, spatial_distribution: np.ndarray):
        """Initialize quench memory."""
        
    def apply_memory_kernel(self, field_history: List[np.ndarray]) -> np.ndarray:
        """Apply memory kernel to field history."""
        
    def compute_drift_velocity(self, field_evolution: np.ndarray) -> float:
        """Compute cell drift velocity from field evolution."""
        
    def analyze_pinning_strength(self, gamma_range: List[float]) -> PinningAnalysis:
        """Analyze pinning strength vs memory parameter."""
```

#### Модуль ABCD: `abcd_model.py`
```python
class ABCDModel:
    """
    ABCD (transmission matrix) model for resonator chains.
    
    Physical Meaning:
        Implements the transmission matrix method for analyzing
        cascaded resonators, providing analytical predictions
        for resonance frequencies and quality factors.
    """
    
    def __init__(self, resonators: List[Resonator]):
        """Initialize ABCD model."""
        
    def compute_transmission_matrix(self, frequency: float) -> np.ndarray:
        """Compute 2x2 transmission matrix."""
        
    def find_resonance_conditions(self, frequency_range: Tuple[float, float]) -> List[float]:
        """Find frequencies satisfying resonance conditions."""
        
    def compare_with_numerical(self, numerical_results: NumericalResults) -> ComparisonResults:
        """Compare with numerical simulation results."""
```

### 2. Алгоритмы анализа

#### Анализ адмиттансы
```python
def analyze_admittance_spectrum(field: np.ndarray, source: np.ndarray, 
                               frequency_range: Tuple[float, float]) -> AdmittanceSpectrum:
    """
    Analyze admittance spectrum over frequency range.
    
    Physical Meaning:
        Computes the complex admittance Y(ω) = I(ω)/V(ω) which
        characterizes the system's response to external excitation
        and reveals resonance frequencies and quality factors.
    """
    frequencies = np.logspace(np.log10(frequency_range[0]), 
                             np.log10(frequency_range[1]), 300)
    
    admittance = np.zeros(len(frequencies), dtype=complex)
    
    for i, freq in enumerate(frequencies):
        # Solve stationary problem
        field_freq = solve_stationary_frequency(field, source, freq)
        
        # Compute admittance
        numerator = np.sum(field_freq.conj() * source)
        denominator = np.sum(np.abs(field_freq)**2)
        admittance[i] = numerator / denominator
    
    # Find resonances
    resonances = find_peaks(np.abs(admittance), height=8)  # 8 dB threshold
    
    return AdmittanceSpectrum(frequencies, admittance, resonances)
```

#### Анализ радиальных профилей
```python
def analyze_radial_profiles(field: np.ndarray, center: np.ndarray, 
                           max_radius: float) -> RadialProfiles:
    """
    Analyze radial profiles of field amplitude.
    
    Physical Meaning:
        Computes the radial distribution of field amplitude A(r),
        revealing the spatial structure of resonance modes and
        identifying regions of field concentration.
    """
    # Create radial grid
    r_max = max_radius
    r_points = np.linspace(0, r_max, 100)
    
    # Compute radial profiles
    A_r = np.zeros_like(r_points)
    
    for i, r in enumerate(r_points):
        # Create spherical shell at radius r
        shell_mask = create_spherical_shell_mask(field.shape, center, r, dr=0.1)
        
        # Average field amplitude over shell
        A_r[i] = np.sqrt(np.mean(np.abs(field[shell_mask])**2))
    
    # Find local maxima
    maxima = find_local_maxima(A_r, r_points)
    
    return RadialProfiles(r_points, A_r, maxima)
```

#### Анализ дрейфа ячеек
```python
def analyze_cell_drift(field_evolution: np.ndarray, time_points: np.ndarray) -> DriftAnalysis:
    """
    Analyze cell drift from field evolution.
    
    Physical Meaning:
        Computes the drift velocity of field patterns by analyzing
        cross-correlation of effective intensity I_eff(x,t) over time,
        revealing how patterns move and whether they are pinned.
    """
    # Compute effective intensity
    I_eff = np.abs(field_evolution)**2
    
    # Apply moving average
    window_size = int(0.1 * len(time_points))
    I_eff_smooth = moving_average(I_eff, window_size)
    
    # Compute cross-correlation
    dt = time_points[1] - time_points[0]
    correlation_shifts = []
    
    for i in range(len(time_points) - 1):
        # Cross-correlation between consecutive time steps
        corr = cross_correlation_2d(I_eff_smooth[i], I_eff_smooth[i+1])
        
        # Find peak shift
        shift = find_peak_shift(corr)
        correlation_shifts.append(shift)
    
    # Compute drift velocity
    drift_velocity = np.mean(correlation_shifts) / dt
    
    # Compute Jaccard index for pattern stability
    jaccard_index = compute_jaccard_index(I_eff_smooth)
    
    return DriftAnalysis(drift_velocity, correlation_shifts, jaccard_index)
```

### 3. Конфигурационные файлы

#### Конфигурация C1: `configs/level_c/C1_single_wall.json`
```json
{
  "case": "C1_single_wall",
  "description": "Single spherical boundary with admittance contrast",
  "domain": {
    "L": 100.5309649,
    "N": 384,
    "dimensions": 3
  },
  "physics": {
    "nu": 1.0,
    "beta": 1.0,
    "lambda": 0.0,
    "precision": "float64"
  },
  "boundary": {
    "type": "spherical",
    "center": [50.26548245, 50.26548245, 50.26548245],
    "radius": 18.84955592,
    "thickness": 3,
    "contrast_range": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
  },
  "frequency_sweep": {
    "omega_min": 0.05,
    "omega_max": 5.0,
    "points_per_decade": 60,
    "refine_peaks": true,
    "peak_refinement_tolerance": 0.001
  },
  "analysis": {
    "find_resonance_modes": true,
    "compute_radial_profiles": true,
    "analyze_admittance_spectrum": true,
    "check_passivity": true
  },
  "output": {
    "directory": "output/C1",
    "save_profiles": true,
    "save_admittance": true,
    "save_field_snapshots": true,
    "format": "hdf5"
  }
}
```

#### Конфигурация C2: `configs/level_c/C2_chain_abcd.json`
```json
{
  "case": "C2_chain_abcd",
  "description": "Chain of 5 nested resonators with ABCD analysis",
  "domain": {
    "L": 125.6637061,
    "N": 384,
    "dimensions": 3
  },
  "physics": {
    "nu": 1.0,
    "beta": 1.2,
    "lambda": 0.0,
    "precision": "float64"
  },
  "resonators": {
    "shells": [
      {
        "R_over_pi": 6.0,
        "thickness_in_cells": 3,
        "memory": [{"gamma": 0.3, "tau": 0.7}]
      },
      {
        "R_over_pi": 9.0,
        "thickness_in_cells": 3,
        "memory": [{"gamma": 0.2, "tau": 1.0}]
      },
      {
        "R_over_pi": 12.0,
        "thickness_in_cells": 3,
        "memory": [{"gamma": 0.3, "tau": 1.5}]
      },
      {
        "R_over_pi": 15.0,
        "thickness_in_cells": 3,
        "memory": [{"gamma": 0.25, "tau": 1.0}]
      },
      {
        "R_over_pi": 18.0,
        "thickness_in_cells": 3,
        "memory": [{"gamma": 0.35, "tau": 0.8}]
      }
    ]
  },
  "frequency_sweep": {
    "omega_min": 0.05,
    "omega_max": 6.0,
    "points_per_decade": 60,
    "refine_peaks": true
  },
  "abcd_analysis": {
    "use_two_runs": true,
    "epsilon_dipole": 0.08,
    "tolerance": 0.05
  },
  "analysis": {
    "abcd_comparison": true,
    "system_mode_analysis": true,
    "mode_coupling_analysis": true
  },
  "output": {
    "directory": "output/C2",
    "save_transmission_matrices": true,
    "save_abcd_validation": true,
    "save_system_modes": true
  }
}
```

#### Конфигурация C3/C4: `configs/level_c/C3C4_beating_pinning.json`
```json
{
  "case": "C3C4_beating_pinning",
  "description": "Mode beating and pinning analysis",
  "domain": {
    "L": 100.5309649,
    "N": 384,
    "dimensions": 3
  },
  "physics": {
    "nu": 1.0,
    "beta": 1.0,
    "lambda": 0.0,
    "precision": "float64"
  },
  "layers": [
    {
      "R_over_pi": 6.0,
      "thickness_in_cells": 3,
      "memory": [{"gamma": 0.0, "tau": 1.0}]
    }
  ],
  "time_integration": {
    "dt": 0.005,
    "T": 400.0,
    "avg_window": 0.8,
    "scheme": "exponential"
  },
  "forcing": {
    "type": "dual_mode",
    "omega_0": 1.0,
    "delta_omega_ratios": [0.02, 0.05],
    "amplitude": 1.0,
    "profile": "gaussian"
  },
  "memory_sweep": {
    "gamma_list": [0.0, 0.2, 0.4, 0.6, 0.8],
    "tau_list": [0.5, 1.0, 2.0]
  },
  "analysis": {
    "compute_drift_velocity": true,
    "compute_cross_correlation": true,
    "compute_jaccard_index": true,
    "find_freezing_threshold": true
  },
  "output": {
    "directory": "output/C3C4",
    "save_intensity_series": true,
    "save_k_spectra": true,
    "save_drift_analysis": true
  }
}
```

### 4. Критерии приёмки и валидация

#### Детализированные критерии

**C1 - Одна стенка:**
- При η = 0: Нет пиков ≥ 8 дБ в спектре Y(ω)
- При η ≥ 0.1: 
  - Появляется ≥ 1 пик в Y(ω)
  - Локальные максимумы A(r) между ядром и оболочкой
  - Порог η* для первой моды определен с точностью ±5%
- Пассивность: Re Y(ω) ≥ 0 для всех ω
- Сходимость: Ошибки ω_n ≤ 3%, Q_n ≤ 10% при N → 512

**C2 - Цепочка резонаторов:**
- ≥ 3 пика в Y(ω) ≥ 8 дБ
- Ошибки ABCD-прогноза:
  - Общие ошибки ≤ 10%
  - Ошибки на пиках ≤ 5%
  - Частоты пиков ≤ 5%
  - Добротности Q ≤ 10%
- Пассивность: Re Y(ω) ≥ 0
- Согласованность мод между ABCD и численными результатами

**C3 - Квенч-память:**
- При γ = 0: v_cell ≈ Δω/|Δk|, отклонение ≤ 10%
- При γ ≥ γ*: v_cell ≤ 10⁻³ L/T₀ (заморожено)
- Жаккар-индекс ≥ 0.95 на длинных окнах
- Порог γ* определен с точностью ±10%

**C4 - Биения мод:**
- Без пиннинга: |v_cell^num - v_cell^pred|/v_cell^pred ≤ 10%
- С пиннингом: v_cell^num/v_cell^pred ≤ 0.1 (подавление ≥10×)
- Согласованность с результатами C3

#### Валидационные тесты

```python
def validate_level_c_results():
    """Validate all Level C test results."""
    
    # C1 validation
    c1_results = load_test_results("C1")
    assert c1_results.no_peaks_at_zero_contrast()
    assert c1_results.resonance_birth_threshold() <= 0.1
    assert c1_results.passivity_check()
    assert c1_results.convergence_check()
    
    # C2 validation
    c2_results = load_test_results("C2")
    assert c2_results.minimum_peaks() >= 3
    assert c2_results.abcd_errors() <= 0.1
    assert c2_results.peak_errors() <= 0.05
    assert c2_results.passivity_check()
    
    # C3 validation
    c3_results = load_test_results("C3")
    assert c3_results.drift_velocity_at_zero_memory() <= 0.1
    assert c3_results.freezing_threshold() <= 0.1
    assert c3_results.jaccard_index() >= 0.95
    
    # C4 validation
    c4_results = load_test_results("C4")
    assert c4_results.beating_error_without_pinning() <= 0.1
    assert c4_results.suppression_with_pinning() >= 10.0
    
    return True
```

### 5. Выходные данные и отчётность

#### Структура выходных данных
```
output/
├── C1_single_wall/
│   ├── admittance_spectrum.h5
│   ├── radial_profiles.h5
│   ├── resonance_modes.json
│   ├── field_snapshots.h5
│   └── analysis_report.json
├── C2_chain_abcd/
│   ├── system_admittance.h5
│   ├── transmission_matrices/
│   ├── abcd_validation.csv
│   ├── system_modes.json
│   └── comparison_report.json
├── C3C4_beating_pinning/
│   ├── intensity_evolution.h5
│   ├── drift_analysis.json
│   ├── cross_correlation.h5
│   ├── pinning_analysis.json
│   └── beating_analysis.json
└── level_c_summary/
    ├── test_results.json
    ├── validation_report.json
    └── visualization/
        ├── admittance_spectra.png
        ├── radial_profiles.png
        ├── abcd_comparison.png
        ├── drift_velocity.png
        └── beating_patterns.png
```

#### Формат отчёта
```json
{
  "test_suite": "Level C - Boundaries and Cells",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "results": {
    "C1_single_wall": {
      "status": "PASS",
      "resonance_birth_threshold": 0.12,
      "peak_count": 3,
      "passivity_violations": 0,
      "convergence_errors": {
        "frequency": 0.02,
        "quality_factor": 0.08
      }
    },
    "C2_chain_abcd": {
      "status": "PASS",
      "system_peaks": 5,
      "abcd_errors": {
        "overall": 0.08,
        "peaks": 0.04,
        "frequencies": 0.03,
        "quality_factors": 0.07
      }
    },
    "C3_quench_memory": {
      "status": "PASS",
      "freezing_threshold": 0.35,
      "drift_velocity_at_zero": 0.05,
      "jaccard_index": 0.97
    },
    "C4_mode_beating": {
      "status": "PASS",
      "beating_error_without_pinning": 0.08,
      "suppression_with_pinning": 12.5
    }
  },
  "validation": {
    "all_tests_passed": true,
    "criteria_met": true,
    "performance_metrics": {
      "total_runtime": "2.5 hours",
      "memory_peak": "8.2 GB",
      "accuracy": "high"
    }
  }
}
```

## Заключение

Детализированное задание шага 06 представляет собой комплексную систему тестов для изучения границ и ячеек в 7D фазовом поле. Ключевые аспекты:

1. **Физическая основа**: Переход от фундаментальных свойств к сложным резонансным структурам
2. **Математические модели**: От простых границ к сложным цепочкам резонаторов
3. **Экспериментальная валидация**: Четыре критически важных теста с жёсткими критериями
4. **Практическая реализация**: Детальные алгоритмы и конфигурации

Успешное выполнение уровня C создаёт основу для перехода к многомодовым моделям уровня D и демонстрирует, как границы и контрасты в фазовом поле создают структуры, предшествующие частицам.
