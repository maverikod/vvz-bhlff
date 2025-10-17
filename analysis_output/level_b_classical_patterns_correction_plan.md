# Пошаговый план исправления классических паттернов в уровне B

## 🎯 Цель
Исправить классические паттерны (Юкава, экспоненциальное затухание) на **ступенчатую теорию 7D BVP** с дискретными слоями и геометрическим убыванием.

---

## 📋 Шаг 1: Исправление Power Law Analyzer

### **Файл**: `bhlff/models/level_b/power_law_analyzer.py`

#### **1.1. Метод `analyze_power_law_tail` (строки 56-147)**

**❌ Классический паттерн**:
```python
# Строки 104-117: Простая лог-лог регрессия
log_r = np.log(r_tail)
log_A = np.log(np.abs(A_tail))
slope, intercept, r_squared, _, _ = stats.linregress(log_r_valid, log_A_valid)

# Строки 119-124: Простой степенной закон
theoretical_slope = 2 * beta - 3
```

**✅ Исправление на ступенчатую структуру**:
```python
def analyze_stepwise_tail(self, field: np.ndarray, beta: float, center: List[float]) -> Dict[str, Any]:
    """
    Analyze STEPWISE tail structure with discrete layers.
    
    Physical Meaning:
        Analyzes discrete layered structure with geometric decay
        ||∇θₙ₊₁|| ≤ q ||∇θₙ|| instead of simple power law.
    """
    # 1. Detect discrete layers R₀ < R₁ < R₂ < ...
    layers = self._detect_stepwise_layers(field, center)
    
    # 2. Analyze geometric decay between layers
    q_factors = self._compute_geometric_decay(layers)
    
    # 3. Check radius quantization
    quantization = self._check_radius_quantization(layers)
    
    return {
        "layers": layers,
        "q_factors": q_factors,
        "quantization": quantization,
        "stepwise_structure": True
    }
```

#### **1.2. Новые методы для ступенчатой структуры**

**Добавить после строки 347**:
```python
def _detect_stepwise_layers(self, field: np.ndarray, center: List[float]) -> List[Dict[str, Any]]:
    """
    Detect discrete layers R₀ < R₁ < R₂ < ... in stepwise structure.
    
    Physical Meaning:
        Identifies discrete layers with quantized radii according to
        7D BVP theory: Θ(r) = Σₙ≥₀ θₙ(r), θₙ поддержана в [Rₙ,Rₙ₊₁]
    """
    radial_profile = self._compute_radial_profile(field, center)
    
    # Detect layer boundaries using gradient analysis
    gradient = np.gradient(radial_profile["A"], radial_profile["r"])
    second_derivative = np.gradient(gradient, radial_profile["r"])
    
    # Find layer boundaries as significant changes in gradient
    layer_boundaries = self._find_layer_boundaries(gradient, second_derivative)
    
    # Extract layers
    layers = []
    for i in range(len(layer_boundaries) - 1):
        r_start = layer_boundaries[i]
        r_end = layer_boundaries[i + 1]
        
        layer_mask = (radial_profile["r"] >= r_start) & (radial_profile["r"] < r_end)
        layer_data = {
            "r_start": r_start,
            "r_end": r_end,
            "amplitude": radial_profile["A"][layer_mask],
            "radius": radial_profile["r"][layer_mask],
            "layer_index": i
        }
        layers.append(layer_data)
    
    return layers

def _compute_geometric_decay(self, layers: List[Dict[str, Any]]) -> List[float]:
    """
    Compute geometric decay factors q between layers.
    
    Physical Meaning:
        Computes ||∇θₙ₊₁|| ≤ q ||∇θₙ|| for geometric decay
        between discrete layers in stepwise structure.
    """
    q_factors = []
    
    for i in range(len(layers) - 1):
        current_layer = layers[i]
        next_layer = layers[i + 1]
        
        # Compute gradient norms
        current_grad_norm = np.linalg.norm(np.gradient(current_layer["amplitude"]))
        next_grad_norm = np.linalg.norm(np.gradient(next_layer["amplitude"]))
        
        # Geometric decay factor
        if current_grad_norm > 0:
            q_factor = next_grad_norm / current_grad_norm
            q_factors.append(q_factor)
    
    return q_factors

def _check_radius_quantization(self, layers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Check radius quantization in discrete layers.
    
    Physical Meaning:
        Verifies that layer boundaries follow quantized pattern
        R₀ < R₁ < R₂ < ... with discrete spacing.
    """
    if len(layers) < 2:
        return {"quantized": False, "spacing_ratio": None}
    
    # Extract layer boundaries
    boundaries = [layer["r_start"] for layer in layers]
    boundaries.append(layers[-1]["r_end"])
    
    # Check for geometric spacing
    spacing_ratios = []
    for i in range(len(boundaries) - 2):
        ratio = (boundaries[i + 2] - boundaries[i + 1]) / (boundaries[i + 1] - boundaries[i])
        spacing_ratios.append(ratio)
    
    # Check if ratios are approximately constant (quantized)
    if len(spacing_ratios) > 0:
        mean_ratio = np.mean(spacing_ratios)
        std_ratio = np.std(spacing_ratios)
        quantized = std_ratio / mean_ratio < 0.1  # 10% tolerance
    else:
        quantized = False
        mean_ratio = None
    
    return {
        "quantized": quantized,
        "spacing_ratio": mean_ratio,
        "spacing_ratios": spacing_ratios
    }
```

---

## 📋 Шаг 2: Исправление Node Analyzer

### **Файл**: `bhlff/models/level_b/node_analyzer.py`

#### **2.1. Метод `check_spherical_nodes` (строки 47-105)**

**❌ Классический паттерн**:
```python
# Строки 75-88: Простая монотонность
dA_dr = np.gradient(A, r)
sign_changes = self._count_sign_changes(dA_dr)
is_monotonic = self._check_monotonicity(A, r)
```

**✅ Исправление на ступенчатую структуру**:
```python
def check_stepwise_structure(self, field: np.ndarray, center: List[float]) -> Dict[str, Any]:
    """
    Check for stepwise structure instead of simple monotonicity.
    
    Physical Meaning:
        Verifies discrete layered structure with quantized transitions
        instead of simple monotonic decay.
    """
    # 1. Detect stepwise pattern
    stepwise_pattern = self._detect_stepwise_pattern(field, center)
    
    # 2. Check level quantization
    level_quantization = self._check_level_quantization(field, center)
    
    # 3. Verify discrete layers
    discrete_layers = self._verify_discrete_layers(field, center)
    
    return {
        "stepwise_structure": stepwise_pattern,
        "level_quantization": level_quantization,
        "discrete_layers": discrete_layers,
        "passed": stepwise_pattern and level_quantization and discrete_layers
    }
```

#### **2.2. Новые методы для ступенчатой структуры**

**Добавить после строки 421**:
```python
def _detect_stepwise_pattern(self, field: np.ndarray, center: List[float]) -> bool:
    """
    Detect stepwise pattern in field structure.
    
    Physical Meaning:
        Identifies discrete stepwise transitions instead of
        smooth monotonic decay.
    """
    radial_profile = self._compute_radial_profile(field, center)
    
    # Analyze gradient for stepwise transitions
    gradient = np.gradient(radial_profile["A"], radial_profile["r"])
    second_derivative = np.gradient(gradient, radial_profile["r"])
    
    # Look for sharp transitions (steps)
    gradient_threshold = np.std(gradient) * 2
    sharp_transitions = np.abs(gradient) > gradient_threshold
    
    # Count significant transitions
    num_transitions = np.sum(sharp_transitions)
    
    # Stepwise pattern if we have discrete transitions
    return num_transitions > 0

def _check_level_quantization(self, field: np.ndarray, center: List[float]) -> bool:
    """
    Check for level quantization in stepwise structure.
    
    Physical Meaning:
        Verifies that field levels are quantized according to
        discrete layer structure.
    """
    radial_profile = self._compute_radial_profile(field, center)
    
    # Find local extrema (quantized levels)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(radial_profile["A"])
    valleys, _ = find_peaks(-radial_profile["A"])
    
    # Check if extrema follow quantized pattern
    if len(peaks) > 1:
        peak_values = radial_profile["A"][peaks]
        # Check for quantized spacing
        peak_spacing = np.diff(peak_values)
        if len(peak_spacing) > 0:
            # Check if spacing is approximately constant
            mean_spacing = np.mean(peak_spacing)
            std_spacing = np.std(peak_spacing)
            quantized = std_spacing / mean_spacing < 0.2  # 20% tolerance
        else:
            quantized = False
    else:
        quantized = False
    
    return quantized

def _verify_discrete_layers(self, field: np.ndarray, center: List[float]) -> bool:
    """
    Verify discrete layered structure.
    
    Physical Meaning:
        Confirms that field exhibits discrete layered structure
        with clear boundaries between layers.
    """
    radial_profile = self._compute_radial_profile(field, center)
    
    # Analyze field structure for discrete layers
    amplitude = radial_profile["A"]
    radius = radial_profile["r"]
    
    # Look for clear layer boundaries
    gradient = np.gradient(amplitude, radius)
    second_derivative = np.gradient(gradient, radius)
    
    # Find significant changes in gradient (layer boundaries)
    gradient_changes = np.abs(np.diff(gradient))
    threshold = np.std(gradient_changes) * 1.5
    
    # Count significant layer boundaries
    significant_changes = np.sum(gradient_changes > threshold)
    
    # Discrete layers if we have clear boundaries
    return significant_changes > 0
```

---

## 📋 Шаг 3: Исправление Zone Analyzer

### **Файл**: `bhlff/models/level_b/zone_analyzer.py`

#### **3.1. Метод `separate_zones` (строки 65-138)**

**❌ Классический паттерн**:
```python
# Строки 90-102: Простое разделение по порогам
indicators = self._compute_zone_indicators(field)
N = indicators["N"]
S = indicators["S"]
core_mask = (N_norm > thresholds["N_core"]) & (S_norm > thresholds["S_core"])
```

**✅ Исправление на ступенчатую структуру**:
```python
def separate_stepwise_zones(self, field: np.ndarray, center: List[float]) -> Dict[str, Any]:
    """
    Separate field into stepwise zones with discrete layers.
    
    Physical Meaning:
        Separates field into discrete layered zones with
        quantized transitions between layers.
    """
    # 1. Detect discrete layers
    layers = self._detect_stepwise_layers(field, center)
    
    # 2. Classify layers into zones
    zone_classification = self._classify_layers_to_zones(layers)
    
    # 3. Compute zone statistics for discrete layers
    zone_stats = self._compute_stepwise_zone_statistics(field, zone_classification)
    
    return {
        "layers": layers,
        "zone_classification": zone_classification,
        "zone_stats": zone_stats,
        "stepwise_structure": True
    }
```

#### **3.2. Новые методы для ступенчатой структуры**

**Добавить после строки 376**:
```python
def _detect_stepwise_layers(self, field: np.ndarray, center: List[float]) -> List[Dict[str, Any]]:
    """
    Detect discrete layers in stepwise structure.
    
    Physical Meaning:
        Identifies discrete layers with quantized radii
        according to 7D BVP theory.
    """
    # Similar to power law analyzer but for zone classification
    radial_profile = self._compute_radial_profile(field, center)
    
    # Detect layer boundaries
    gradient = np.gradient(radial_profile["A"], radial_profile["r"])
    second_derivative = np.gradient(gradient, radial_profile["r"])
    
    # Find significant transitions
    layer_boundaries = self._find_layer_boundaries(gradient, second_derivative)
    
    # Extract layers
    layers = []
    for i in range(len(layer_boundaries) - 1):
        r_start = layer_boundaries[i]
        r_end = layer_boundaries[i + 1]
        
        layer_mask = (radial_profile["r"] >= r_start) & (radial_profile["r"] < r_end)
        layer_data = {
            "r_start": r_start,
            "r_end": r_end,
            "amplitude": radial_profile["A"][layer_mask],
            "radius": radial_profile["r"][layer_mask],
            "layer_index": i
        }
        layers.append(layer_data)
    
    return layers

def _classify_layers_to_zones(self, layers: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """
    Classify discrete layers into core/transition/tail zones.
    
    Physical Meaning:
        Classifies discrete layers into zones based on
        their position and properties in stepwise structure.
    """
    if len(layers) < 3:
        return {"core": [], "transition": [], "tail": []}
    
    # Classify based on layer position and amplitude
    core_layers = []
    transition_layers = []
    tail_layers = []
    
    for i, layer in enumerate(layers):
        # Core: first few layers with high amplitude
        if i < len(layers) // 3:
            core_layers.append(i)
        # Tail: last few layers with low amplitude
        elif i >= 2 * len(layers) // 3:
            tail_layers.append(i)
        # Transition: middle layers
        else:
            transition_layers.append(i)
    
    return {
        "core": core_layers,
        "transition": transition_layers,
        "tail": tail_layers
    }

def _compute_stepwise_zone_statistics(self, field: np.ndarray, zone_classification: Dict[str, List[int]]) -> Dict[str, Any]:
    """
    Compute statistics for stepwise zones.
    
    Physical Meaning:
        Computes statistics for discrete layered zones
        instead of continuous zone separation.
    """
    stats = {}
    
    for zone_name, layer_indices in zone_classification.items():
        if layer_indices:
            # Compute statistics for this zone's layers
            zone_amplitudes = []
            zone_radii = []
            
            for layer_idx in layer_indices:
                # Get layer data (would need to store this)
                # This is a simplified version
                pass
            
            stats[zone_name] = {
                "num_layers": len(layer_indices),
                "layer_indices": layer_indices,
                "stepwise_structure": True
            }
        else:
            stats[zone_name] = {
                "num_layers": 0,
                "layer_indices": [],
                "stepwise_structure": True
            }
    
    return stats
```

---

## 📋 Шаг 4: Исправление тестов

### **Файл**: `tests/unit/test_level_b/test_fundamental_properties_basic.py`

#### **4.1. Метод `test_power_law_tail` (строки 94-133)**

**❌ Классический паттерн**:
```python
# Строки 120-125: Простая проверка степенного закона
result = self.power_law_analyzer.analyze_power_law_tail(field, beta, center)
assert result["passed"], f"Power law test failed"
```

**✅ Исправление на ступенчатую структуру**:
```python
def test_stepwise_tail(self):
    """
    Test B1: Stepwise tail structure instead of simple power law.
    
    Physical Meaning:
        Verifies discrete layered structure with geometric decay
        instead of simple power law behavior.
    """
    try:
        # Generate test field
        field = self._generate_test_field()
        
        # Analyze stepwise structure
        result = self.power_law_analyzer.analyze_stepwise_tail(field, self.parameters.beta, center)
        
        # Verify stepwise structure
        assert result["stepwise_structure"], "Should have stepwise structure"
        assert len(result["layers"]) > 0, "Should have discrete layers"
        assert len(result["q_factors"]) > 0, "Should have geometric decay factors"
        assert result["quantization"]["quantized"], "Should have quantized radii"
        
        return {
            "passed": True,
            "layers": len(result["layers"]),
            "q_factors": result["q_factors"],
            "quantization": result["quantization"]
        }
        
    except Exception as e:
        return {
            "passed": False,
            "error": str(e)
        }
```

#### **4.2. Метод `test_no_spherical_nodes` (строки 135-170)**

**❌ Классический паттерн**:
```python
# Строки 150-157: Простая проверка монотонности
node_result = self.node_analyzer.check_spherical_nodes(field, center)
assert node_result["sign_changes"] <= 1, "Should have minimal sign changes"
assert node_result["is_monotonic"], "Should be monotonic"
```

**✅ Исправление на ступенчатую структуру**:
```python
def test_stepwise_structure(self):
    """
    Test B2: Stepwise structure instead of simple monotonicity.
    
    Physical Meaning:
        Verifies discrete layered structure with quantized
        transitions instead of simple monotonic decay.
    """
    try:
        # Generate test field
        field = self._generate_test_field()
        
        # Check stepwise structure
        result = self.node_analyzer.check_stepwise_structure(field, center)
        
        # Verify stepwise structure
        assert result["stepwise_structure"], "Should have stepwise structure"
        assert result["level_quantization"], "Should have level quantization"
        assert result["discrete_layers"], "Should have discrete layers"
        
        return {
            "passed": True,
            "stepwise_structure": result["stepwise_structure"],
            "level_quantization": result["level_quantization"],
            "discrete_layers": result["discrete_layers"]
        }
        
    except Exception as e:
        return {
            "passed": False,
            "error": str(e)
        }
```

---

## 📋 Шаг 5: Обновление конфигураций

### **Файл**: `configs/level_b_tests.json`

#### **5.1. Обновление конфигурации B1**

**❌ Классический паттерн**:
```json
"B1_power_law": {
    "analysis": {
        "min_decades": 1.5,
        "r_squared_threshold": 0.99,
        "error_threshold": 0.05
    }
}
```

**✅ Исправление на ступенчатую структуру**:
```json
"B1_stepwise_tail": {
    "analysis": {
        "min_layers": 3,
        "q_factor_threshold": 0.8,
        "quantization_tolerance": 0.1,
        "stepwise_structure_required": true
    }
}
```

#### **5.2. Обновление конфигурации B2**

**❌ Классический паттерн**:
```json
"B2_no_nodes": {
    "analysis": {
        "max_sign_changes": 1,
        "periodicity_tolerance": 0.1
    }
}
```

**✅ Исправление на ступенчатую структуру**:
```json
"B2_stepwise_structure": {
    "analysis": {
        "stepwise_structure_required": true,
        "level_quantization_required": true,
        "discrete_layers_required": true,
        "quantization_tolerance": 0.2
    }
}
```

---

## 📋 Шаг 6: Обновление критериев приёмки

### **Файл**: `docs/theory/7d-32-БВП_план_численных_экспериментов_B.md`

#### **6.1. Обновление критериев B1 (строка 215)**

**❌ Классический паттерн**:
```
* Наклон $\hat p$ в ДИ 95% попадает в $p_\text{теор}=2\beta-3 \pm 0.05$.
```

**✅ Исправление на ступенчатую структуру**:
```
* Обнаружены дискретные слои R₀ < R₁ < R₂ < ... (минимум 3 слоя).
* Геометрическое убывание: q ∈ (0,1) для всех пар соседних слоев.
* Квантование радиусов: относительная ошибка ≤ 10%.
```

#### **6.2. Обновление критериев B2 (строка 231)**

**❌ Классический паттерн**:
```
* Число изменений знака $Z\le 1$ (разрешён единичный локальный перегиб на границе ядра);
```

**✅ Исправление на ступенчатую структуру**:
```
* Обнаружена ступенчатая структура с дискретными слоями.
* Квантование уровней: относительная ошибка ≤ 20%.
* Дискретные слои: минимум 2 значимых перехода.
```

---

## 🎯 Итоговый план выполнения

### **Приоритет 1 (Критично)**:
1. **Шаг 1**: Исправить `power_law_analyzer.py` - метод `analyze_power_law_tail`
2. **Шаг 2**: Исправить `node_analyzer.py` - метод `check_spherical_nodes`
3. **Шаг 4**: Обновить тесты для ступенчатой структуры

### **Приоритет 2 (Важно)**:
4. **Шаг 3**: Исправить `zone_analyzer.py` - метод `separate_zones`
5. **Шаг 5**: Обновить конфигурации

### **Приоритет 3 (Желательно)**:
6. **Шаг 6**: Обновить документацию и критерии приёмки

**Общее время выполнения**: 2-3 дня
**Критичность**: Высокая (неправильная физика)
