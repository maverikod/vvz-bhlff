# НЕМЕДЛЕННЫЕ ДЕЙСТВИЯ для гипотезы А

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com  
**Date:** 2024-12-19

## 🚨 КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ (СЕГОДНЯ)

### 1. FFTSolver7DBasic - добавить solve() метод
**Файл:** `bhlff/core/fft/fft_solver_7d_basic.py`
**Строка:** После метода `solve_stationary()`

```python
def solve(self, source: np.ndarray) -> np.ndarray:
    """
    Main solve method - alias for solve_stationary.
    
    Physical Meaning:
        Solves the fractional Laplacian equation L_β a = s
        using spectral methods in 7D space-time.
    """
    return self.solve_stationary(source)

def get_spectral_coefficients(self) -> np.ndarray:
    """
    Get spectral coefficients for the fractional Laplacian.
    
    Returns:
        np.ndarray: Spectral coefficients |k|^(2β) + λ
    """
    return self.spectral_coefficients
```

### 2. FractionalLaplacian - исправить инициализацию
**Файл:** `bhlff/core/operators/fractional_laplacian.py`
**Строка:** 69 (заменить проверку beta)

```python
def __init__(self, domain: "Domain", beta: Union[float, "Parameters"], lambda_param: float = 0.0):
    """
    Initialize fractional Laplacian operator.
    
    Args:
        domain (Domain): Computational domain.
        beta (Union[float, Parameters]): Fractional order or parameters object.
        lambda_param (float): Damping parameter.
    """
    # Handle both float and Parameters object
    if hasattr(beta, 'beta'):
        # Parameters object
        self.beta = beta.beta
        self.lambda_param = getattr(beta, 'lambda_param', lambda_param)
    else:
        # Direct float value
        self.beta = beta
        self.lambda_param = lambda_param
    
    # Validate parameters
    if not (0 < self.beta < 2):
        raise ValueError("Fractional order beta must be in (0,2)")
    
    # Rest of initialization...
```

### 3. BVP Core - добавить solve_envelope()
**Файл:** `bhlff/core/bvp/bvp_core/bvp_core_facade_impl.py`
**Строка:** После метода `__init__()`

```python
def solve_envelope(self, source: np.ndarray) -> np.ndarray:
    """
    Solve BVP envelope equation.
    
    Physical Meaning:
        Solves the BVP envelope equation:
        ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    """
    return self._operations.solve_envelope(source)

def detect_quenches(self, envelope: np.ndarray) -> Dict[str, Any]:
    """
    Detect quench events in BVP envelope.
    
    Physical Meaning:
        Detects threshold events (amplitude/detuning/gradient)
        in the BVP envelope field.
    """
    return self._operations.detect_quenches(envelope)
```

### 4. BVP Operations - добавить методы
**Файл:** `bhlff/core/bvp/bvp_core/bvp_operations.py`
**Строка:** После метода `_setup_parameter_access()`

```python
def solve_envelope(self, source: np.ndarray) -> np.ndarray:
    """
    Solve BVP envelope equation.
    
    Physical Meaning:
        Solves the BVP envelope equation using the envelope solver.
    """
    return self._envelope_solver.solve(source)

def detect_quenches(self, envelope: np.ndarray) -> Dict[str, Any]:
    """
    Detect quench events in BVP envelope.
    
    Physical Meaning:
        Detects threshold events using the quench detector.
    """
    return self._quench_detector.detect(envelope)
```

## 🧪 ТЕСТИРОВАНИЕ

После внесения изменений запустить:

```bash
cd /home/vasilyvz/Desktop/Инерция/7d/progs/bhlff
python -m pytest tests/unit/test_level_a/ -v --tb=short
```

**Ожидаемый результат:** Прохождение тестов должно увеличиться с 50 до 80+

## 📊 ПРОВЕРКА РЕЗУЛЬТАТОВ

1. **Количество проходящих тестов:** должно быть ≥80
2. **Количество падающих тестов:** должно быть ≤15
3. **Количество ошибок:** должно быть ≤5

## 🎯 СЛЕДУЮЩИЕ ШАГИ

После исправления критических ошибок:
1. Реализовать QuenchDetector
2. Реализовать U(1)³ Phase Vector
3. Реализовать BVP Impedance Calculator
4. Добавить интеграционные тесты

**ЦЕЛЬ:** Полная готовность к проверке гипотезы А через 2 недели
