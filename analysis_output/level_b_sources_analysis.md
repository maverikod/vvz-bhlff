# Анализ использования источников в уровне B

## Сравнение теории и реализации

### 📋 Требования теории (из документа 7d-32)

**Нейтрализованная гауссиана**:
```
g_σ(x) = exp(-|x-x₀|²/(2σ²))
ḡ = (1/L³) ∫_Ω g_σ dx
s(x) = g_σ(x) - ḡ
```

**Ключевые требования**:
- σ ∈ [1.5Δ, 3Δ] (чтобы избежать одиночного узла сетки)
- Центр x₀ = (L/2, L/2, L/2)
- **Критично**: ŝ(0) = 0 автоматически (нейтрализация)

### 🔍 Реализация в коде

#### 1. BVPSource в тестах уровня B

**Файл**: `tests/unit/test_level_b/test_fundamental_properties_basic.py`

```python
# Конфигурация BVPSource
config = {
    "carrier_frequency": 1.85e43,
    "envelope_amplitude": 1.0,
    "base_source_type": "gaussian"  # ← Использует гауссов тип
}
self.source = BVPSource(self.domain, config)
```

**Генерация источника**:
```python
def _generate_test_field(self) -> np.ndarray:
    # Создание источника через BVPSource
    source_field = self.source.generate()  # ← BVP-модулированный источник
    field = self._solve_field_equation(source_field)
    return field
```

#### 2. BVPSourceGenerators - типы источников

**Файл**: `bhlff/core/sources/bvp_source_generators.py`

**Доступные типы**:
- `"gaussian"` - гауссов источник
- `"point"` - точечный источник  
- `"distributed"` - распределенный источник

**Гауссов источник**:
```python
def generate_gaussian_source(self) -> np.ndarray:
    # Параметры из конфига
    amplitude = self.config.get("gaussian_amplitude", 1.0)
    center = self.config.get("gaussian_center", [0.5, 0.5, 0.5])
    width = self.config.get("gaussian_width", 0.1)
    
    # Создание гауссианы
    source = amplitude * self._step_resonator_source(r_squared, width)
    return source
```

## 🔶 Отклонения от теории

### 1. **Отсутствие нейтрализации**

**Проблема**: BVPSource не реализует нейтрализацию ḡ = (1/L³) ∫_Ω g_σ dx

**Теория требует**:
```python
s(x) = g_σ(x) - ḡ  # Нейтрализация
```

**Реализация**:
```python
source = amplitude * self._step_resonator_source(r_squared, width)  # Без нейтрализации
```

### 2. **BVP-модуляция вместо чистого источника**

**Проблема**: BVPSource создает BVP-модулированный источник:
```python
s(x) = s₀(x) * A(x) * exp(iω₀t)  # BVP-модулированный
```

**Теория требует**: Чистый нейтрализованный источник для тестов уровня B

### 3. **Отсутствие контроля σ ∈ [1.5Δ, 3Δ]**

**Проблема**: Ширина источника не контролируется согласно теории

**Теория**: σ ∈ [1.5Δ, 3Δ]
**Реализация**: width = 0.1 (фиксированное значение)

## 🎯 Где используется BVPSource

### 1. **Тесты уровня B**
- `test_fundamental_properties_basic.py` - B1, B2
- `test_fundamental_properties_advanced.py` - B3, B4
- `test_simple_level_b.py` - простые тесты

### 2. **Интеграционные тесты**
- `test_bvp_level_b_integration.py` - BVP интеграция

### 3. **Конфигурации**
- `configs/level_b_tests.json` - параметры тестов

## 🔧 Рекомендации по исправлению

### 1. **Добавить нейтрализованный гауссов источник**

```python
def generate_neutralized_gaussian_source(self) -> np.ndarray:
    """Generate neutralized Gaussian source according to theory."""
    # Параметры согласно теории
    sigma = self.config.get("gaussian_sigma", 2.0 * self.domain.L / self.domain.N)
    center = self.config.get("gaussian_center", [0.5, 0.5, 0.5])
    
    # Создание гауссианы
    g_sigma = self._create_gaussian(sigma, center)
    
    # Нейтрализация: s(x) = g_σ(x) - ḡ
    g_bar = np.mean(g_sigma)
    neutralized_source = g_sigma - g_bar
    
    return neutralized_source
```

### 2. **Добавить контроль σ ∈ [1.5Δ, 3Δ]**

```python
def _validate_sigma(self, sigma: float) -> float:
    """Validate sigma is in required range [1.5Δ, 3Δ]."""
    delta = self.domain.L / self.domain.N
    min_sigma = 1.5 * delta
    max_sigma = 3.0 * delta
    
    if sigma < min_sigma or sigma > max_sigma:
        raise ValueError(f"Sigma {sigma} must be in [{min_sigma}, {max_sigma}]")
    
    return sigma
```

### 3. **Создать чистый источник для уровня B**

```python
class LevelBSource:
    """Pure source for Level B tests (no BVP modulation)."""
    
    def generate_neutralized_gaussian(self, sigma: float, center: List[float]) -> np.ndarray:
        """Generate neutralized Gaussian source according to theory."""
        # Реализация согласно теории
        pass
```

## 📊 Влияние на результаты

### ✅ **Положительные аспекты**
- BVPSource обеспечивает BVP-интеграцию
- Модульная архитектура позволяет легко добавить нейтрализацию
- Тесты проходят с текущей реализацией

### ⚠️ **Потенциальные проблемы**
- Отсутствие нейтрализации может влиять на точность тестов
- BVP-модуляция добавляет сложность, не требуемую для уровня B
- Критерии приёмки могут быть неточными из-за неправильных источников

## 🎯 Заключение

**BVPSource используется в уровне B для**:
1. **Генерации источников** в тестах B1-B4
2. **BVP-интеграции** с фреймворком
3. **Модуляции источников** высокочастотным носителем

**Основная проблема**: Отсутствие нейтрализации согласно теории, что может влиять на точность тестов уровня B.

**Рекомендация**: Добавить нейтрализованный гауссов источник для точного соответствия теории.
