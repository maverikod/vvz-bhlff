# Author: Vasiliy Zdanovskiy
# email: vasilyvz@gmail.com

## Soliton Optimization Algorithms Documentation

## Overview

This document describes the complete soliton optimization algorithms implemented according to 7D BVP (Boundary Value Problem) theory. The algorithms provide full functionality for finding and analyzing soliton solutions in 7D phase field theory without classical simplifications.

## Physical Meaning

The soliton optimization algorithms implement the complete 7D phase field theory for soliton solutions, including:

- **Single soliton solutions** with complete optimization using 7D fractional Laplacian equations
- **Multi-soliton solutions** with full interaction analysis and stability properties
- **Step resonator theory** replacing classical exponential decay patterns
- **Complete energy computation** including kinetic and potential energy contributions
- **Topological charge analysis** for soliton characterization

## Mathematical Foundation

### 7D Soliton Equation

The algorithms solve the 7D fractional Laplacian equation:

```math
L_β a = μ(-Δ)^β a + λa = s(x,t)
```

Where:
- `β ∈ (0,2)` is the fractional order
- `μ > 0` is the diffusion coefficient
- `λ ≥ 0` is the damping parameter
- `s(x,t)` is the source term

### Step Resonator Theory

Instead of classical exponential decay, the algorithms use step resonator theory:

```math
f(x) = 1 if |x - pos| < width, 0 if |x - pos| ≥ width
```

This provides sharp cutoffs at interaction ranges, following 7D BVP principles.

## Algorithm Components

### 1. Single Soliton Optimization

**Class:** `SingleSolitonSolver`

**Key Methods:**
- `find_single_soliton()`: Complete optimization using 7D BVP theory
- `_step_resonator_profile()`: Step resonator profile implementation
- `_step_resonator_source()`: Step resonator source terms
- `compute_7d_soliton_ode()`: 7D fractional Laplacian ODE system
- `compute_soliton_energy()`: Complete energy computation

**Physical Meaning:**
Finds single soliton solutions through complete optimization using 7D fractional Laplacian equations and boundary value problem solving.

**Mathematical Foundation:**
Solves the 7D soliton equation with soliton boundary conditions and energy minimization.

### 2. Multi-Soliton Optimization

**Class:** `MultiSolitonSolver`

**Key Methods:**
- `find_multi_soliton_solutions()`: Complete multi-soliton optimization
- `find_two_soliton_solutions()`: Two-soliton optimization with interactions
- `find_three_soliton_solutions()`: Three-soliton optimization with all interactions
- `_step_resonator_interaction()`: Step resonator interaction implementation

**Physical Meaning:**
Finds multi-soliton solutions through complete optimization using 7D fractional Laplacian equations and soliton-soliton interaction potentials.

**Mathematical Foundation:**
Solves the multi-soliton system:
```math
L_β a = μ(-Δ)^β a + λa + V_int(a₁, a₂, ...) = s(x,t)
```
where `V_int` represents soliton-soliton interactions.

### 3. Core Multi-Soliton Operations

**Class:** `MultiSolitonCore`

**Key Methods:**
- `compute_7d_two_soliton_ode()`: Two-soliton ODE system with interactions
- `compute_7d_three_soliton_ode()`: Three-soliton ODE system with all interactions
- `compute_two_soliton_energy()`: Total energy including interactions
- `compute_three_soliton_energy()`: Total energy including all interactions
- `_step_resonator_source()`: Step resonator source terms

**Physical Meaning:**
Implements core multi-soliton physics including ODE systems, energy calculations, and interaction potentials for multi-soliton configurations.

## Key Features

### 1. Complete Optimization

All algorithms use full optimization without simplifications:

- **L-BFGS-B optimization** with proper bounds and convergence criteria
- **Boundary value problem solving** using `scipy.integrate.solve_bvp`
- **Complete energy minimization** including all interaction terms
- **Full convergence analysis** with iteration counts and gradient norms

### 2. Step Resonator Theory

Replaces classical exponential decay with step resonator theory:

- **Sharp cutoffs** at interaction ranges
- **Step function profiles** instead of exponential decay
- **Step function sources** for soliton generation
- **Step function interactions** between solitons

### 3. 7D Fractional Laplacian

Implements the complete fractional Laplacian operator:

- **Spectral representation** using FFT
- **Proper handling** of k=0 mode
- **Full 7D BVP theory** implementation
- **Energy conservation** in optimization

### 4. Complete Energy Computation

Includes all energy contributions:

- **Kinetic energy** from field gradients
- **Potential energy** from field values
- **Interaction energy** between solitons
- **Three-body interactions** for multi-soliton systems

## Usage Examples

### Single Soliton Optimization

```python
from bhlff.models.level_f.nonlinear.soliton_analysis.single_soliton import SingleSolitonSolver

# Initialize solver
solver = SingleSolitonSolver(system, nonlinear_params)

# Find single soliton solution
solution = solver.find_single_soliton()

if solution:
    print(f"Amplitude: {solution['amplitude']}")
    print(f"Width: {solution['width']}")
    print(f"Position: {solution['position']}")
    print(f"Energy: {solution['energy']}")
    print(f"Optimization Success: {solution['optimization_success']}")
```

### Multi-Soliton Optimization

```python
from bhlff.models.level_f.nonlinear.soliton_analysis.multi_soliton import MultiSolitonSolver

# Initialize solver
solver = MultiSolitonSolver(system, nonlinear_params)

# Find multi-soliton solutions
solutions = solver.find_multi_soliton_solutions()

for solution in solutions:
    print(f"Type: {solution['type']}")
    print(f"Number of solitons: {solution['num_solitons']}")
    print(f"Energy: {solution['energy']}")
    print(f"Optimization Success: {solution['optimization_success']}")
```

### Step Resonator Profile

```python
# Generate step resonator profile
x = np.linspace(-5.0, 5.0, 100)
position = 0.0
width = 2.0

profile = solver._step_resonator_profile(x, position, width)

# Profile is 1.0 where |x - position| < width, 0.0 otherwise
```

## Testing

The algorithms are thoroughly tested with comprehensive test suites:

- **Complete optimization testing** with convergence verification
- **Step resonator theory testing** with comparison to exponential profiles
- **7D fractional Laplacian testing** with linearity verification
- **Energy computation testing** with scaling verification
- **Topological charge testing** with different field configurations

## Performance Considerations

### Optimization Parameters

- **L-BFGS-B method** for efficient optimization
- **Proper bounds** for physical parameters
- **Convergence criteria** with `ftol=1e-9`
- **Maximum iterations** appropriate for problem complexity

### Memory Usage

- **Efficient FFT operations** for fractional Laplacian
- **Sparse matrix operations** where possible
- **Optimized array operations** for large systems

### Computational Complexity

- **O(N log N)** for FFT operations
- **O(N²)** for interaction calculations
- **O(N³)** for three-body interactions

## Differences from Classical Approaches

### 1. Step Resonator vs Exponential Decay

**Classical:** `f(x) = exp(-x²/σ²)`
**7D BVP:** `f(x) = 1 if |x| < σ, 0 if |x| ≥ σ`

### 2. Complete vs Simplified Optimization

**Classical:** Simplified ML predictions, placeholder implementations
**7D BVP:** Full optimization with complete algorithms

### 3. Energy Computation

**Classical:** Simplified energy calculations
**7D BVP:** Complete energy including all interaction terms

### 4. Interaction Theory

**Classical:** Exponential interaction potentials
**7D BVP:** Step resonator interaction theory

## Future Enhancements

1. **GPU acceleration** for large-scale optimizations
2. **Parallel multi-soliton optimization** for complex systems
3. **Adaptive mesh refinement** for better resolution
4. **Machine learning integration** for parameter prediction
5. **Real-time visualization** of optimization progress

## References

1. 7D Phase Field Theory - Complete Implementation
2. Fractional Laplacian in Spectral Space - Mathematical Foundation
3. Step Resonator Theory - 7D BVP Principles
4. Soliton-Soliton Interactions - Multi-body Physics
5. Boundary Value Problem Solving - Numerical Methods
