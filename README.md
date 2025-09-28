# BHLFF: 7D Phase Field Theory Implementation

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/vasilyvz/bhlff/workflows/Tests/badge.svg)](https://github.com/vasilyvz/bhlff/actions)
[![Documentation Status](https://readthedocs.org/projects/bhlff/badge/?version=latest)](https://bhlff.readthedocs.io/en/latest/?badge=latest)

**BHLFF** (Base High-Frequency Field Framework) is a comprehensive implementation of the 7D phase field theory for elementary particles. This framework provides numerical tools for simulating BVP envelope dynamics in 7-dimensional space-time, including topological defects, solitons, and collective phenomena.

## 🎯 Overview

BHLFF implements the theoretical framework where the **Base High-Frequency Field (BVP)** serves as the central backbone of the entire system. All observed "modes" are envelope modulations and beatings of the high-frequency carrier, with a three-level structure:

- **Core**: High coherence region with BVP envelope topological defects
- **Transition Zone**: Nonlinear interface between core and tail with BVP quench events
- **Tail**: Pure wave nature described by BVP envelope modulations

## 🚀 Key Features

- **BVP Framework**: Base High-Frequency Field as central backbone with envelope equation `∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)`
- **7D Space-Time**: Full 7D implementation M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ with U(1)³ phase vector structure
- **Quench Detection**: Three-threshold system for amplitude, detuning, and gradient quench events
- **Memory Kernel**: Debye-type memory with passivity condition for BVP envelope evolution
- **Multi-level Architecture**: Hierarchical implementation from BVP validation (Level A) to cosmological models (Level G)
- **BVP Impedance**: Y(ω), R(ω), T(ω) calculation and resonance peaks {ω_n,Q_n} analysis
- **Electroweak Currents**: EM and weak current generation from BVP envelope functionals
- **Comprehensive Testing**: Automated BVP framework validation across all levels A-G

## 📦 Installation

### From PyPI (recommended)

```bash
pip install bhlff
```

### Development Installation

```bash
git clone https://github.com/vasilyvz/bhlff.git
cd bhlff
pip install -e .[dev]
```

### With Optional Dependencies

```bash
# For visualization
pip install bhlff[visualization]

# For GPU acceleration
pip install bhlff[performance]

# For documentation
pip install bhlff[docs]
```

## 🏗️ Architecture

The package is organized in hierarchical levels:

```
bhlff/
├── core/           # Level A: Basic solvers and validation
├── models/         # Level B-G: Physics models
│   ├── level_a/    # Solver validation and scaling
│   ├── level_b/    # Fundamental field properties
│   ├── level_c/    # Boundaries and resonators
│   ├── level_d/    # Multi-modal superposition
│   ├── level_e/    # Solitons and defects
│   ├── level_f/    # Collective effects
│   └── level_g/    # Cosmological models
├── utils/          # Utilities and analysis tools
└── cli/            # Command-line interface
```

## 🧮 Quick Start

### Basic BVP Framework Usage

```python
import numpy as np
from bhlff.core.domain import Domain
from bhlff.core.bvp import BVPCore
from bhlff.solvers.spectral import FFTSolver3D

# Create computational domain
domain = Domain(
    dimensions=3,
    size=(10.0, 10.0, 10.0),
    resolution=(256, 256, 256),
    boundary_conditions="periodic"
)

# Set up BVP configuration
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

# Create BVP core
bvp_core = BVPCore(domain, bvp_config)

# Create FFT solver with BVP integration
solver = FFTSolver3D(domain, bvp_config, bvp_core)

# Define source term
source = np.zeros(domain.shape)
source[128, 128, 128] = 1.0  # Point source

# Solve BVP envelope equation
envelope = solver.solve_bvp_envelope(source)

# Detect quench events
quenches = solver.detect_quenches(envelope)

# Compute BVP impedance
impedance = solver.compute_bvp_impedance(envelope)

# Get U(1)³ phase vector
phase_vector = bvp_core.get_phase_vector()
phase_components = bvp_core.get_phase_components()

# Compute electroweak currents
currents = bvp_core.compute_electroweak_currents(envelope)
```

### Command Line Interface

```bash
# Run BVP framework validation tests
bhlff run --level A --bvp

# Analyze BVP envelope power law tails
bhlff analyze --test B1 --bvp --beta 1.0

# Test BVP quench detection
bhlff test --quench-detection --thresholds 0.8,0.1,0.5

# Generate BVP framework report
bhlff report --bvp --output results/
```

## 🧪 Testing

Run the complete BVP framework test suite:

```bash
# All BVP framework tests
pytest tests/test_bvp_framework.py -v

# BVP integration tests for all levels A-G
pytest tests/test_bvp_levels_integration.py -v

# Specific BVP level tests
pytest -m level_a --bvp
pytest -m level_b --bvp

# BVP performance tests
pytest -m slow --bvp
```

## 📚 Documentation

- [Full Documentation](https://bhlff.readthedocs.io)
- [API Reference](https://bhlff.readthedocs.io/en/latest/api/)
- [Theory Guide](https://bhlff.readthedocs.io/en/latest/theory/)
- [Examples](https://bhlff.readthedocs.io/en/latest/examples/)

## 🔬 Theoretical Background

BHLFF implements the 7D phase field theory where:

- **7D Space-time**: M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ (3 spatial + 3 phase + 1 time)
- **Phase Field**: θ(x,φ,t) with topological defects
- **Fractional Dynamics**: Governed by Riesz operator with β ∈ (0,2)
- **Power Law Tails**: A(r) ∝ r^(2β-3) in homogeneous media
- **BVP Framework**: High-frequency carrier with slow envelope modulation

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Vasiliy Zdanovskiy**
- Email: vasilyvz@gmail.com
- GitHub: [@vasilyvz](https://github.com/vasilyvz)

## 🙏 Acknowledgments

- Theoretical framework based on 7D phase field theory
- Numerical methods from fractional calculus
- Community feedback and contributions

## 📊 Citation

If you use BHLFF in your research, please cite:

```bibtex
@software{bhlff2024,
  title={BHLFF: 7D Phase Field Theory Implementation},
  author={Zdanovskiy, Vasiliy},
  year={2024},
  url={https://github.com/vasilyvz/bhlff},
  license={MIT}
}
```

---

**Note**: This is an active research project. The API may change between versions as the theoretical framework evolves.
