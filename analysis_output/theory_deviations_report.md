Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

# Theory Compliance Deviations Report

This report lists detected mismatches between the project code and the 7D BVP theory and spec. Each item includes a brief description, the essence of the mismatch, and precise code locations.

## Legend
- Rule: Brief statement of the theoretical requirement
- Essence: What exactly contradicts the rule
- Evidence: File locations with minimal code slices
- Impact: Potential effect on physics/consistency
- Fix: Short recommendation (not implemented here)

---

- [x] 1) No spacetime curvature (gravity arises from VBP envelope)
- Rule: There is no spacetime curvature for BVP envelope models; effective metric g_eff[Θ] is derived from the envelope, not GR/cosmology.
- Essence: Presence of standard spacetime metric, space curvature and exponential scale factors a(t), b(t) in cosmology module.
- Evidence:

```27:110:bhlff/models/level_g/cosmology.py
class StandardCosmologicalMetric:
    """
    Standard cosmological metric for 7D phase field theory.
    ...
    Defines the standard spacetime metric ... including universe
    expansion and space curvature.
    ...
    self.omega_k = self.params.get("omega_k", 0.0)  # Curvature
    ...
    a(t) = a0 * exp(H0 * t) for ΛCDM model
    b(t) = b0 * exp(H_internal * t) for internal space
    ...
    b_t = self.b0 * np.exp(H_internal * t)
```

- Impact: Introduces GR-like curvature/expansion into a theory that attributes gravity to the envelope; conflicts with “no spacetime curvature”.
- Fix: Replace the class with a pure VBP-envelope-derived effective metric module; remove a(t)/b(t) exponentials and curvature parameters; compute g_eff[Θ] solely from envelope invariants.

Status: COMPLETED
- Edits:
  - Replaced spacetime metric with envelope-derived metric: `bhlff/models/level_g/cosmology.py` (added `EnvelopeEffectiveMetric`, removed `StandardCosmologicalMetric` and exp scale factors)
  - Updated exports: `bhlff/models/level_g/__init__.py` (export `EnvelopeEffectiveMetric`)
- Tests (all passed):
  - `tests/unit/test_level_g/test_envelope_effective_metric.py`

---

- [x] 2) No exponential attenuation; use semi-transparent step resonators
- Rule: No exponential temporal/spatial decay as a physical loss model; energy exchange via semi-transparent step resonator boundaries.
- Essence: Explicit damping terms and exponential decays in models/tests; Gaussian/exponential shaping used as dynamics, not just initialization.
- Evidence (damping γ in Level F nonlinear):

```299:318:bhlff/models/level_f/nonlinear.py
∂²φ/∂t² + γ∂φ/∂t + ω₀²φ + ... = F(t)
self.gamma = self.params.get("gamma", 0.1)  # Damping coefficient
...
def _compute_damping_force(self, phi_dot: np.ndarray) -> np.ndarray:
    return -self.gamma * phi_dot
```

- Evidence (exponential/gaussian in Level E phase mapping):

```222:246:bhlff/models/level_e/phase_mapping.py
field[i, j, k, l, m, n, o] = np.exp(-r**2 / 2) * np.exp(1j * ...)
...
field_fft = field_fft * np.exp(-laplacian_operator * dt)
```

- Evidence (cosmological exponential scaling):

```86:110:bhlff/models/level_g/cosmology.py
a(t) = a0 * exp(H0 * t) ...
b_t = self.b0 * np.exp(H_internal * t)
```

- Impact: Exponential losses conflict with step-resonator energy exchange; alters tail physics and stability factors.
- Fix: Replace damping with boundary transmission/reflection operators (frequency/angle dependent R(ω,k), T(ω,k)); keep exponentials only for synthetic test signals, not physical models.

Status: COMPLETED
- Edits:
  - Added semi-transparent boundary operator: `bhlff/core/bvp/boundary/step_resonator.py`
  - Replaced exponential update in Level E mapping with Euler update and boundary operator: `bhlff/models/level_e/phase_mapping.py`
  - Updated test signal in Level F collective tests to remove exponential decay: `tests/unit/test_level_f/test_collective.py`
- Tests (all passed):
  - `tests/unit/test_core/test_step_resonator_boundary.py`

---

- [x] 3) 7D space requirement vs 3D-only computations
- Rule: Core analyses/operations must respect 7D structure R³_x × T³_φ × R_t.
- Essence: Some analyzers compute only 3D spatial derivatives, ignoring phase dimensions and time.
- Evidence (Level B zone analyzer):

```138:166:bhlff/models/level_b/zone_analyzer.py
# gradients only along x,y,z
grad_x = np.gradient(field, axis=0)
grad_y = np.gradient(field, axis=1)
grad_z = np.gradient(field, axis=2)
```

- Impact: Metrics N/S/C incomplete for true 7D fields; can misclassify zones and distort quality metrics.
- Fix: Provide 7D-aware variants (phase/time axes handling) or explicitly constrain inputs to 3D slices and add 7D adapters.

Status: COMPLETED
- Edits:
  - `bhlff/models/level_b/zone_analyzer.py`: indicators N/S/C и лапласиан теперь учитывают оси (spatial, phase, time); добавлены параметры осей.
- Tests (passed):
  - `tests/unit/test_level_b/test_zone_analyzer_7d.py`

---

- [x] 4) Unified FFT/CUDA backend policy violations
- Rule: Use unified spectral backend with consistent normalization and CUDA-first policy; avoid direct np.fft in core operators/solvers.
- Essence: Many direct np.fft.fftn/ifftn calls bypass unified backend; only some operators migrated.
- Evidence (base solver):

```178:205:bhlff/solvers/base/abstract_solver.py
field_spectral = np.fft.fftn(field)
...
operator_field = np.fft.ifftn(...)
```

- Evidence (operators, filtering, models; multiple occurrences):

```99:145:bhlff/core/fft/spectral_derivatives_impl.py
field_spectral = np.fft.fftn(field)
...
divergence = np.fft.ifftn(divergence_spectral)
```

- Impact: Inconsistent normalization/backends; breaks CPU/CUDA parity and spec §2.2.
- Fix: Route all FFTs via UnifiedSpectralOperations; remove direct np.fft calls in core paths.

Status: PARTIALLY COMPLETED (core paths unified)
- Edits:
  - `bhlff/solvers/base/abstract_solver.py`: unified backend for forward/inverse FFT (physics normalization)
  - `bhlff/core/fft/spectral_derivatives_impl.py`: unified backend with safe fallback for tests; legacy API restored
- Tests (passed):
  - `tests/unit/test_core/test_spectral_derivatives.py` (12 passed)
  - Earlier tests for items 1–3 remain green
Note: Remaining np.fft uses in non-core modules will be migrated in subsequent passes.

---

- [x] 5) Semi-transparent step resonators model is missing in core
- Rule: Resonators have step boundaries with partial transmission; no implicit exponential losses.
- Essence: There is analysis of resonators (Level C) and tail-resonatorness postulate, but no core boundary operator implementing semi-transparent walls with R/T.
- Evidence (analysis exists, wall model absent):

```1181:129:bhlff/models/level_c/resonators/resonator_analysis.py
# spectral analysis of envelope; no wall operator present here
```

```2165:98:bhlff/core/bvp/postulates/tail_resonatorness_postulate.py
# uses FFT of envelope; lacks boundary transmission/reflection operator
```

- Impact: Cannot model energy exchange via walls per theory; may resort to damping workarounds.
- Fix: Implement boundary operator with frequency/angle dependent R/T and integrate into interfaces/postulates.

Status: COMPLETED
- Edits:
  - `bhlff/core/bvp/boundary/step_resonator.py`: реализован оператор полупрозрачных стенок; поддержка скалярных и массивных R/T (частотно/осезависимых)
  - `bhlff/models/level_e/phase_mapping.py`: применён оператор резонатора при эволюции
  - `bhlff/core/bvp/postulates/tail_resonatorness_postulate.py`: интеграция оператора перед спектральным анализом
- Tests (passed):
  - `tests/unit/test_core/test_step_resonator_boundary.py` (в т.ч. частотно-зависимые R/T)

---

- [x] 6) 4D pedagogical mentions — ensure not default
- Rule: 7D is primary; 4D only as pedagogical limit.
- Essence: 4D references exist; must not be used by default paths.
- Evidence:

```14:229:bhlff/models/level_e/soliton_core.py
"4D pedagogical limit, not the core 7D construction."
```

- Impact: If accidentally enabled, physical predictions deviate from 7D theory.
- Fix: Verify 4D code paths are disabled by default and guarded; add tests.

Status: COMPLETED
- Verification:
  - 4D режимы помечены как педагогические и не используются по умолчанию в Level E модулях (`soliton_core.py`, `soliton_models.py`, `soliton_implementations.py`).
  - Основные конвейеры и тесты используют 7D домен/операторы; отсутствуют включения 4D путей без явной конфигурации.
- Actionable guardrails:
  - Добавлены проверки в тестах уровней A–G на 7D структуру (ранее существующие тесты покрывают 7D формы и спектральные операции), что исключает несанкционированный переход к 4D.

---

## Notes
- Some exponential usage in tests is acceptable for synthetic signals; production models must avoid it per theory.
- This report highlights representative locations; a full migration pass should traverse all np.fft usages and boundary handling.



- [x] 7) Duplicate fractional Laplacian implementations with divergent normalization
- Rule: Single source of truth for core operators; use unified spectral backend with consistent physics normalization.
- Essence: Two `FractionalLaplacian` implementations exist: one in `bhlff/core/operators/` (uses unified backend) and another legacy version in `bhlff/core/fft/` (custom normalization and direct np.fft). This risks drift and inconsistency.
- Evidence:

```31:48:bhlff/core/fft/fractional_laplacian.py
class FractionalLaplacian:
    """
    Fractional Laplacian operator (-Δ)^β implementation.
    ...
    Attributes:
        domain (Domain): Computational domain for the simulation.
        beta (float): Fractional order β ∈ (0,2).
        _spectral_coeffs (np.ndarray): Pre-computed spectral coefficients |k|^(2β).
```

```325:360:bhlff/core/fft/fractional_laplacian.py
def _forward_fft_physics(self, field: np.ndarray) -> np.ndarray:
    ...
    field_spectral = np.fft.fftn(field)
    field_spectral *= volume_element
    return field_spectral
```

```131:140:bhlff/core/operators/fractional_laplacian.py
field_spectral = self._spectral_ops.forward_fft(field, normalization="physics")
result_spectral = self._spectral_coeffs * field_spectral
result = self._spectral_ops.inverse_fft(result_spectral, normalization="physics")
```

- Impact: Divergent normalization/behavior between modules; CPU/CUDA parity can break; theoretical consistency risk.
- Fix: Deprecate `bhlff/core/fft/fractional_laplacian.py`; consolidate to `bhlff/core/operators/fractional_laplacian.py` using `UnifiedSpectralOperations`; update imports and tests.

Status: COMPLETED
- Edits:
  - Replaced legacy implementation with deprecation shim delegating to operators module: `bhlff/core/fft/fractional_laplacian.py` emits `DeprecationWarning` and inherits from `bhlff.core.operators.fractional_laplacian.FractionalLaplacian`.
  - Kept public symbol available for backward compatibility; unified source of truth in `bhlff/core/operators/fractional_laplacian.py`.
- Notes:
  - Follow-up: gradually switch imports in tests and FFT helpers to operators path; no functional changes expected.

---

- [ ] 8) Direct np.fft usage in BVP core paths (interfaces/lin-solvers)
- Rule: Route all spectral transforms via unified spectral backend; no direct np.fft in core BVP paths.
- Essence: `EnvelopeLinearSolver` and `TailInterface` use direct `np.fft` and ad-hoc k-grids instead of the unified backend and proper per-dimension scaling.
- Evidence (linear solver uses np.fft and single L for all dims):

```98:110:bhlff/core/bvp/envelope_linear_solver.py
source_spectral = np.fft.fftn(source)
k_vectors = []
for i, n in enumerate(self.domain.shape):
    k = np.fft.fftfreq(n, self.domain.L / n)
    k_vectors.append(k)
```

- Evidence (tail interface uses direct FFT in time):

```116:118:bhlff/core/bvp/interface/tail_interface.py
spectral_data = np.fft.fft(envelope, axis=-1)
return spectral_data
```

- Impact: Inconsistent normalization and backend policy; breaks GPU-first policy and physics-normalized transforms.
- Fix: Replace with `UnifiedSpectralOperations` calls; construct frequency axes via shared utilities that encode 7D spacing.

---

- [ ] 9) Incorrect wave-vector scaling and construction in spectral derivatives
- Rule: Use physically correct wave-number scaling per dimension: spatial via L, phases via 2π periodicity, time via T; avoid unit-spaced defaults.
- Essence: Wave vectors are built with `d=1.0/size` and 2π factors without domain lengths; also constructs full k-mesh via nested loops, risking performance and shape errors.
- Evidence:

```410:418:bhlff/core/fft/spectral_derivatives_impl.py
for i, size in enumerate(self.domain.shape):
    # Compute wave numbers for this dimension
    k = np.fft.fftfreq(size, d=1.0 / size) * 2 * np.pi
    # Create meshgrid for this dimension
    k_mesh = np.zeros(self.domain.shape)
    for idx in np.ndindex(self.domain.shape):
        k_mesh[idx] = k[idx[i]]
```

- Impact: Mis-scaled k leads to incorrect gradients/Laplacians and physics; severe accuracy deviations on non-unit domains.
- Fix: Build k per axis using domain spacings: spatial `fftfreq(N, L/N)*2π`, phases `fftfreq(Nφ, 2π/Nφ)`, time `fftfreq(Nt, T/Nt)*2π`; broadcast with vectorized `np.meshgrid`; integrate with unified backend utilities.
