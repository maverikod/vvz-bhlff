# BVP Framework API Reference

## Overview

The **Base High-Frequency Field (BVP)** framework serves as the central backbone of the entire BHLFF system. This document provides comprehensive API reference for all BVP framework components.

## Core Components

### BVPCore

The central BVP framework class that orchestrates all BVP operations.

```python
from bhlff.core.bvp import BVPCore

bvp_core = BVPCore(domain, config)
```

#### Methods

##### `solve_envelope(source: np.ndarray) -> np.ndarray`

Solves the BVP envelope equation:
```
∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
```

**Parameters:**
- `source`: Source term s(x) in real space

**Returns:**
- BVP envelope a(x) in real space

**Example:**
```python
source = np.zeros(domain.shape)
source[32, 32, 32] = 1.0
envelope = bvp_core.solve_envelope(source)
```

##### `detect_quenches(envelope: np.ndarray) -> Dict[str, Any]`

Detects quench events when local thresholds are reached.

**Parameters:**
- `envelope`: BVP envelope a(x) to analyze

**Returns:**
- Dictionary with quench detection results:
  - `quench_locations`: Spatial locations of quenches
  - `quench_types`: Types of quenches detected
  - `energy_dumped`: Energy dumped at each quench

**Example:**
```python
quenches = bvp_core.detect_quenches(envelope)
print(f"Found {len(quenches['quench_locations'])} quench events")
```

##### `compute_impedance(envelope: np.ndarray) -> Dict[str, Any]`

Computes impedance/admittance from BVP envelope.

**Parameters:**
- `envelope`: BVP envelope a(x) to analyze

**Returns:**
- Dictionary with impedance analysis results:
  - `admittance`: Y(ω) frequency response
  - `reflection`: R(ω) reflection coefficient
  - `transmission`: T(ω) transmission coefficient
  - `peaks`: {ω_n,Q_n} resonance peaks

**Example:**
```python
impedance = bvp_core.compute_impedance(envelope)
peaks = impedance['peaks']
print(f"Found {len(peaks)} resonance peaks")
```

##### `get_phase_vector() -> PhaseVector`

Gets the U(1)³ phase vector structure.

**Returns:**
- PhaseVector object containing three U(1) phase components

**Example:**
```python
phase_vector = bvp_core.get_phase_vector()
phase_components = bvp_core.get_phase_components()
```

##### `compute_electroweak_currents(envelope: np.ndarray) -> Dict[str, np.ndarray]`

Computes electroweak currents as functionals of the envelope.

**Parameters:**
- `envelope`: BVP envelope |A|

**Returns:**
- Dictionary with electroweak currents:
  - `em_current`: Electromagnetic current
  - `weak_current`: Weak interaction current
  - `mixed_current`: Mixed electroweak current

**Example:**
```python
currents = bvp_core.compute_electroweak_currents(envelope)
em_current = currents['em_current']
```

### QuenchDetector

Detects quench events in BVP envelope using three threshold criteria.

```python
from bhlff.core.bvp import QuenchDetector

detector = QuenchDetector(config, constants)
```

#### Methods

##### `detect_quenches(envelope: np.ndarray) -> Dict[str, Any]`

Detects quench events using three threshold criteria:
- **Amplitude threshold**: |A| > |A_q|
- **Detuning threshold**: |ω - ω_0| > Δω_q
- **Gradient threshold**: |∇A| > |∇A_q|

**Parameters:**
- `envelope`: BVP envelope to analyze

**Returns:**
- Dictionary with quench detection results

##### `get_thresholds() -> Dict[str, float]`

Gets current quench detection thresholds.

**Returns:**
- Dictionary with threshold values

##### `set_thresholds(thresholds: Dict[str, float]) -> None`

Sets new quench detection thresholds.

**Parameters:**
- `thresholds`: New threshold values

### BVPEnvelopeSolver

Solves the BVP envelope equation with nonlinear stiffness and susceptibility.

```python
from bhlff.core.bvp import BVPEnvelopeSolver

solver = BVPEnvelopeSolver(domain, config, constants)
```

#### Methods

##### `solve_envelope(source: np.ndarray) -> np.ndarray`

Solves the nonlinear BVP envelope equation.

**Parameters:**
- `source`: Source term s(x)

**Returns:**
- BVP envelope solution a(x)

### BVPImpedanceCalculator

Calculates impedance/admittance from BVP envelope.

```python
from bhlff.core.bvp import BVPImpedanceCalculator

calculator = BVPImpedanceCalculator(domain, config, constants)
```

#### Methods

##### `compute_impedance(envelope: np.ndarray) -> Dict[str, Any]`

Computes Y(ω), R(ω), T(ω), and peaks {ω_n,Q_n} from BVP envelope.

**Parameters:**
- `envelope`: BVP envelope a(x)

**Returns:**
- Dictionary with impedance analysis results

### PhaseVector

Implements the U(1)³ phase vector structure Θ_a (a=1..3).

```python
from bhlff.core.bvp import PhaseVector

phase_vector = PhaseVector(domain, config, constants)
```

#### Methods

##### `get_phase_components() -> List[np.ndarray]`

Gets the three U(1) phase components Θ_a (a=1..3).

**Returns:**
- List of three phase components

##### `get_total_phase() -> np.ndarray`

Gets the total phase from U(1)³ structure.

**Returns:**
- Total phase field

##### `compute_electroweak_currents(envelope: np.ndarray) -> Dict[str, np.ndarray]`

Computes electroweak currents from U(1)³ phase structure.

**Parameters:**
- `envelope`: BVP envelope |A|

**Returns:**
- Dictionary with electroweak currents

##### `compute_phase_coherence() -> np.ndarray`

Computes phase coherence measure across U(1)³ structure.

**Returns:**
- Phase coherence measure

### BVPInterface

Provides unified interface between BVP and other system components.

```python
from bhlff.core.bvp import BVPInterface

interface = BVPInterface(bvp_core)
```

#### Methods

##### `interface_with_tail(envelope: np.ndarray) -> Dict[str, Any]`

Interfaces BVP envelope with tail resonators.

**Parameters:**
- `envelope`: BVP envelope

**Returns:**
- Dictionary with tail interface data

##### `interface_with_transition_zone(envelope: np.ndarray) -> Dict[str, Any]`

Interfaces BVP envelope with transition zone.

**Parameters:**
- `envelope`: BVP envelope

**Returns:**
- Dictionary with transition zone interface data

##### `interface_with_core(envelope: np.ndarray) -> Dict[str, Any]`

Interfaces BVP envelope with core.

**Parameters:**
- `envelope`: BVP envelope

**Returns:**
- Dictionary with core interface data

## Integration with Solvers

### BVPEnvelopeSolver for 7D BVP Integration

```python
from bhlff.core.bvp.bvp_envelope_solver import BVPEnvelopeSolver

# Create 7D BVP envelope solver
solver = BVPEnvelopeSolver(domain, config)

# Solve BVP envelope equation
envelope = solver.solve_envelope(source)

# Detect quenches
quenches = solver.detect_quenches(envelope)

# Compute BVP impedance
impedance = solver.compute_bvp_impedance(envelope)
```

### TimeIntegrator with BVP Integration

```python
from bhlff.solvers.integrators import TimeIntegrator

# Create time integrator with BVP integration
integrator = TimeIntegrator(domain, config, bvp_core)

# Detect quenches during time evolution
quenches = integrator.detect_quenches(envelope)
```

## Configuration

### BVP Configuration Structure

```python
bvp_config = {
    "carrier_frequency": 1.85e43,
    "envelope_equation": {
        "kappa_0": 1.0,           # Linear stiffness
        "kappa_2": 0.1,           # Nonlinear stiffness coefficient
        "chi_prime": 1.0,         # Real part of susceptibility
        "chi_double_prime_0": 0.01, # Imaginary part of susceptibility
        "k0_squared": 1.0         # Wave number squared
    },
    "quench_detection": {
        "amplitude_threshold": 0.8,    # Amplitude quench threshold
        "detuning_threshold": 0.1,     # Detuning quench threshold
        "gradient_threshold": 0.5      # Gradient quench threshold
    },
    "impedance_calculation": {
        "frequency_range": [1e15, 1e20],  # Frequency range
        "frequency_points": 1000,         # Number of frequency points
        "boundary_conditions": "periodic" # Boundary conditions
    },
    "material_properties": {
        "em_conductivity": 0.01,      # EM conductivity
        "weak_conductivity": 0.001,   # Weak conductivity
        "su2_coupling_strength": 0.1  # SU(2) coupling strength
    }
}
```

## Mathematical Foundation

### BVP Envelope Equation

The fundamental equation of the BVP framework:

```
∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
```

Where:
- `κ(|a|) = κ₀ + κ₂|a|²` — nonlinear BVP stiffness
- `χ(|a|) = χ' + iχ''(|a|)` — effective susceptibility with quenches
- `s(x,φ,t)` — source term in 7D space-time

### U(1)³ Phase Vector Structure

The BVP field is a vector of three U(1) phases:

```
Θ(x,φ,t) = (Θ₁, Θ₂, Θ₃)
```

Each component represents an independent U(1) phase degree of freedom.

### Quench Detection Criteria

Three threshold criteria for quench events:

1. **Amplitude threshold**: `|A| > |A_q|`
2. **Detuning threshold**: `|ω - ω_0| > Δω_q`
3. **Gradient threshold**: `|∇A| > |∇A_q|`

### Memory Kernel

Debye-type memory with passivity condition:

```
Γ_mem(x,ω) = Σⱼ γⱼ(x)/(1 + iωτⱼ(x))
```

With passivity condition: `ℜΓ_mem ≥ 0`

## Examples

### Complete BVP Workflow

```python
import numpy as np
from bhlff.core.domain import Domain
from bhlff.core.bvp import BVPCore
from bhlff.core.bvp.bvp_envelope_solver import BVPEnvelopeSolver

# 1. Create 7D domain
domain = Domain(
    L=1.0, N=64, dimensions=7, N_phi=32, N_t=100, T=1.0
)

# 2. Configure BVP
bvp_config = {
    "carrier_frequency": 1.85e43,
    "envelope_equation": {
        "kappa_0": 1.0,
        "kappa_2": 0.1,
        "chi_prime": 1.0,
        "chi_double_prime_0": 0.01,
        "k0_squared": 1.0
    },
    "quench_detection": {
        "amplitude_threshold": 0.8,
        "detuning_threshold": 0.1,
        "gradient_threshold": 0.5
    }
}

# 3. Create BVP core
bvp_core = BVPCore(domain, bvp_config)

# 4. Create 7D BVP envelope solver
solver = BVPEnvelopeSolver(domain, bvp_config)

# 5. Define 7D source
source = np.zeros(domain.shape)
source[32, 32, 32, 16, 16, 16, 50] = 1.0

# 6. Solve BVP envelope equation
envelope = solver.solve_envelope(source)

# 7. Detect quench events
quenches = solver.detect_quenches(envelope)

# 8. Compute impedance
impedance = solver.compute_bvp_impedance(envelope)

# 9. Analyze U(1)³ phase vector
phase_vector = bvp_core.get_phase_vector()
phase_components = bvp_core.get_phase_components()
total_phase = bvp_core.get_total_phase()

# 10. Compute electroweak currents
currents = bvp_core.compute_electroweak_currents(envelope)

print(f"BVP envelope solved: {envelope.shape}")
print(f"Quench events detected: {len(quenches['quench_locations'])}")
print(f"Resonance peaks found: {len(impedance['peaks'])}")
print(f"Phase components: {len(phase_components)}")
print(f"Electroweak currents: {list(currents.keys())}")
```

### Quench Detection Analysis

```python
# Analyze quench events
quenches = bvp_core.detect_quenches(envelope)

print("Quench Analysis:")
print(f"  Total quenches: {len(quenches['quench_locations'])}")
print(f"  Quench types: {quenches['quench_types']}")
print(f"  Total energy dumped: {sum(quenches['energy_dumped'])}")

# Modify quench thresholds
new_thresholds = {
    "amplitude_threshold": 0.9,
    "detuning_threshold": 0.2,
    "gradient_threshold": 0.6
}
bvp_core.set_quench_thresholds(new_thresholds)

# Re-detect with new thresholds
quenches_new = bvp_core.detect_quenches(envelope)
print(f"Quenches with new thresholds: {len(quenches_new['quench_locations'])}")
```

### Impedance Analysis

```python
# Compute and analyze impedance
impedance = bvp_core.compute_impedance(envelope)

print("Impedance Analysis:")
print(f"  Admittance range: {impedance['admittance']}")
print(f"  Reflection coefficient: {impedance['reflection']}")
print(f"  Transmission coefficient: {impedance['transmission']}")
print(f"  Resonance peaks: {len(impedance['peaks'])}")

# Analyze resonance peaks
for i, peak in enumerate(impedance['peaks']):
    print(f"  Peak {i+1}: ω = {peak['frequency']}, Q = {peak['quality_factor']}")
```

## Error Handling

### Common Exceptions

- `ValueError`: Invalid domain dimensions or configuration parameters
- `RuntimeError`: FFT operations failure or convergence issues
- `TypeError`: Incorrect parameter types

### Best Practices

1. Always validate domain dimensions before creating BVP components
2. Check configuration parameters for physical consistency
3. Monitor quench detection results for threshold sensitivity
4. Verify impedance calculation convergence
5. Test U(1)³ phase vector normalization

## Performance Considerations

- BVP envelope solving is computationally intensive for large domains
- Quench detection scales with domain size and threshold sensitivity
- Impedance calculation requires frequency domain analysis
- U(1)³ phase vector operations scale with domain size
- Memory usage scales as O(N³) for 3D domains

## Integration Guidelines

1. **Always use BVP framework** as the central backbone
2. **Integrate quench detection** in all time evolution schemes
3. **Use BVP impedance calculation** for boundary conditions
4. **Implement U(1)³ phase vector** in all field calculations
5. **Replace classical patterns** with BVP modulations
6. **Validate BVP consistency** across all levels A-G
