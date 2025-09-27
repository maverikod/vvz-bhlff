# BHLFF: 7D Phase Field Theory Implementation

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/vasilyvz/bhlff/workflows/Tests/badge.svg)](https://github.com/vasilyvz/bhlff/actions)
[![Documentation Status](https://readthedocs.org/projects/bhlff/badge/?version=latest)](https://bhlff.readthedocs.io/en/latest/?badge=latest)

**BHLFF** (Base High-Frequency Field Framework) is a comprehensive implementation of the 7D phase field theory for elementary particles. This framework provides numerical tools for simulating phase field dynamics in 7-dimensional space-time, including topological defects, solitons, and collective phenomena.

## 🎯 Overview

BHLFF implements the theoretical framework where elementary particles are represented as stable phase field configurations with a three-level structure:

- **Core**: High coherence region with topological defects
- **Transition Zone**: Nonlinear interface between core and tail
- **Tail**: Pure wave nature described by wave functions

## 🚀 Key Features

- **Fractional Riesz Operator**: High-precision spectral solver for `L_β a = μ(-Δ)^β a + λa`
- **Multi-level Architecture**: Hierarchical implementation from basic solvers (Level A) to cosmological models (Level G)
- **BVP Framework**: Base High-Frequency Field as central backbone
- **Topological Analysis**: Defect detection, charge computation, and stability analysis
- **Multi-modal Superposition**: Field projections and resonance analysis
- **Comprehensive Testing**: Automated validation across all theoretical predictions

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

### Basic Usage

```python
import numpy as np
from bhlff.core.fft import FFTSolver3D
from bhlff.core.domain import Domain

# Create computational domain
domain = Domain(L=10.0, N=256, dimensions=3)

# Set up physics parameters
params = {
    "mu": 1.0,      # Diffusion coefficient
    "beta": 1.0,    # Fractional order
    "lambda": 0.0   # Damping parameter
}

# Create solver
solver = FFTSolver3D(domain, params)

# Define source term
source = np.zeros((256, 256, 256))
source[128, 128, 128] = 1.0  # Point source

# Solve the equation
solution = solver.solve(source)
```

### Command Line Interface

```bash
# Run basic validation tests
bhlff run --level A

# Analyze power law tails
bhlff analyze --test B1 --beta 1.0

# Generate comprehensive report
bhlff report --output results/
```

## 🧪 Testing

Run the complete test suite:

```bash
# All tests
pytest

# Specific level tests
pytest -m level_a
pytest -m level_b

# Performance tests
pytest -m slow
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
