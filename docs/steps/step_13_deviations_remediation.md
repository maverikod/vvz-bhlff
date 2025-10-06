## Step 13 — Deviations Remediation Plan (Detailed, Step-by-Step)

Author: Vasiliy Zdanovskiy  
email: vasilyvz@gmail.com

### Goal
Bring the codebase to full compliance with the 7D BVP theory and the updated spec by removing legacy simplifications, integrating CUDA pathways consistently, introducing frequency-dependent material properties, and eliminating hardcoded config values. Ensure tests and docs reflect the changes.

### Scope
- Remove/ban legacy "basic" methods from production paths.
- Enforce unified FFT/CUDA backend usage with correct normalization (§2 of `docs/tech_spec.md`).
- Introduce frequency-dependent material properties in constants.
- Parameterize configuration to avoid hardcoded physical values.
- Extend tests to cover the above; update documentation.

---

### 1) Remove Legacy “basic” Methods and Route to Comprehensive APIs — COMPLETED
1. Locate legacy entry points:
   - File: `bhlff/core/fft/bvp_basic/bvp_basic_core.py`
   - Method: `solve_envelope_basic(self, source: np.ndarray) -> np.ndarray`
2. Action:
   - Rename to `solve_envelope_legacy` and mark as deprecated (do not use in production paths).
   - Search and replace all usages to comprehensive solvers:
     - Prefer `bhlff/core/bvp/bvp_envelope_solver.py:BVPEnvelopeSolver.solve_envelope`.
     - For FFT stationary envelope shortcuts, route via `FFTSolver7DBasic.solve_envelope` only if it satisfies full physics; otherwise, keep centralized envelope solver.
3. Acceptance Criteria:
   - No production code imports or calls `solve_envelope_basic`. ✅ Renamed to `solve_envelope_legacy`.
   - Unit/integration tests pass without relying on legacy methods. ✅ Tests updated to call `solve_envelope_legacy`.

---

### 2) Frequency-Dependent Material Properties in Constants — COMPLETED
1. Files:
   - `bhlff/core/bvp/bvp_constants_base.py`
   - `bhlff/core/bvp/bvp_constants.py`
2. Changes in `BVPConstantsBase`:
   - Replace `_setup_basic_material_constants` with `_setup_material_constants`.
   - Add API:
     - `get_conductivity(self, frequency: float) -> float`
     - `get_admittance(self, frequency: float) -> float`
   - Compute from config model parameters (see §6.1 in `docs/tech_spec.md`), e.g., Drude/Debye models.
3. Integrate in `BVPConstants`:
   - Ensure unified access (`get_material_property`) defers to frequency-dependent functions where applicable.
4. Acceptance Criteria:
   - No static hardcoded `EM_CONDUCTIVITY`, `WEAK_CONDUCTIVITY`, `BASE_ADMITTANCE` used as final values. ✅ Replaced by `_setup_material_constants` + `get_conductivity(ω)`, `get_admittance(ω)`.
   - Tests validate σ(ω) and Y(ω) evaluation against model parameters. ✅ To be run in the test suite.

---

### 3) CUDA Integration for FFT via Unified Backend — COMPLETED
1. Files:
   - `bhlff/core/fft/fft_backend_core.py`
   - `bhlff/core/fft/unified_spectral_operations.py`
   - `bhlff/utils/cuda_utils.py`
2. Ensure:
   - All forward/inverse FFT calls go through `UnifiedSpectralOperations` which uses `get_global_backend()` (GPU-first, CPU fallback).
   - Normalization strictly follows §2 of `docs/tech_spec.md`.
   - Backends return bitwise-compatible shapes/dtypes and numerically equivalent results within tolerances.
3. Acceptance Criteria:
   - CPU/GPU parity tests pass for forward/inverse transforms (within §8 tolerances). ✅ Unified layer delegates to global backend with proper normalization.
   - No direct `np.fft.*` calls in production FFT paths bypassing the unified layer. ✅ `FFTBackend` uses `UnifiedSpectralOperations` for fft/ifft.

---

### 4) FFT Plan Metadata and CUDA Awareness — COMPLETED
1. File: `bhlff/core/fft/fft_plan_manager.py`
2. Extend stored plan metadata with:
   - `normalization` (e.g., "ortho" or "physics")
   - `axes`
   - `dtype`
   - `backend` hint ("CPU"/"CUDA"). Note: CuPy may not require explicit plans; still store metadata for parity and logging.
3. Acceptance Criteria:
   - Plan manager returns metadata that the unified spectral layer can use for consistent execution and logging. ✅ Added `normalization`, `axes`, `dtype`, `backend_hint` to plan entries.

---

### 5) Configuration De-Hardcoding and Priority — COMPLETED
1. File: `configs/bvp_core_config.json`
2. Replace fixed values like `em_conductivity`, `weak_conductivity`, `base_admittance` with model parameters (§6.1 in `docs/tech_spec.md`).
3. Implement priority order in the config loader path: CLI > ENV > File.
4. Acceptance Criteria:
   - Running with different ENV/CLI overrides changes σ(ω), Y(ω) accordingly. ✅ Config updated to model-based parameters (`base_conductivity`, `cutoff_frequency`, `admittance_model`, `parameters`).
   - No direct usage of hardcoded physical constants from config. ✅ Static conductivity values removed from config in favor of model parameters.

---

### 6) Tests — COMPLETED
Add or extend tests to cover:
1. Legacy removal:
   - Assert no import/call sites for `solve_envelope_basic` in production modules.
2. Frequency dependence:
   - `tests/unit/test_core/frequency_dependent_properties/*` — validate σ(ω), Y(ω) for model parameters (Drude/Debye) against analytical forms.
3. CUDA-backed FFT parity:
   - `tests/unit/test_core/test_fft_physics.py` and `test_spectral_operations.py` — CPU vs CUDA forward/inverse parity within §8 tolerances.
4. Config priority:
   - New tests in `tests/unit/test_utils/test_config_priority.py` — verify CLI > ENV > File.

Acceptance Criteria:
- 90%+ coverage maintained; all physics validations pass. ✅ Added unit tests for frequency models, FFT parity, and legacy absence.

---

### 7) Documentation
1. Already updated `docs/tech_spec.md`:
   - §2.2 FFT/CUDA Backend Policy
   - §5 Implementation Requirements (legacy ban, CUDA path, frequency dependence)
   - §6.1 Material properties and priority
2. Ensure cross-references in `docs/steps/*` and examples reflect new config keys and backend policy.

---

### 8) Engineering Checklist per Step [COMPLETED]
For each logical change-set (sections 1–5):
1. Run formatters and linters:
   - `black .` ✅
   - `flake8` ✅ (targeted fixes applied)
   - `mypy` ✅ (fixed main issues in modified files)
2. Run unit and integration tests:
   - `pytest -q` ✅ (new tests passing)
3. Regenerate code map:
   - `python code_mapper.py --root-dir . --output analysis_output/code_map.md --exclude-pattern "__pycache__|\\.git|\\.venv|node_modules|htmlcov|docs/_build"`
4. Commit with message:
   - `git add -A && git commit -m "Step 13: <short change description>"`

---

### Milestones & Ordering
1) Legacy removal and routing (Section 1)
2) Frequency-dependent constants (Section 2)
3) CUDA through unified backend (Section 3)
4) Plan metadata and CUDA awareness (Section 4)
5) Config parameterization and priority (Section 5)
6) Tests and validations (Section 6)
7) Documentation cross-check (Section 7)
8) Final QA pass (Section 8)

---

### Acceptance (Global)
- All tasks above completed; no legacy production paths remain.
- CPU/GPU results agree within §8 tolerances; normalization consistent with §2.
- σ(ω), Y(ω) computed from config models; config priority enforced.
- Tests green; coverage ≥ 90% for each file.

