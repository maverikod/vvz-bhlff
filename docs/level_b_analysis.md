# Level B Fundamental Properties Analysis

## Overview

Level B analysis validates the fundamental properties of the phase field in homogeneous "interval-free" medium, confirming the core theoretical predictions of the 7D phase field theory.

## Theoretical Background

### Physical Meaning

Level B represents the analysis of fundamental properties of the BVP field including:

1. **Power Law Tails**: A(r) ∝ r^(2β-3) in homogeneous medium
2. **Absence of Spherical Nodes**: No standing wave patterns in pure fractional regime
3. **Topological Charge**: Integer quantization for defect stability
4. **Zone Separation**: Quantitative separation into core/transition/tail regions

### Mathematical Foundation

The analysis is based on the Riesz operator:
```
L_β a = μ(-Δ)^β a + λa
```

In spectral space:
```
â(k) = ŝ(k)/(μ|k|^(2β) + λ)
```

For λ=0 (pure fractional regime):
- Symbol D(k) = μk^(2β) has no poles
- Leads to power law tails with exponent 2β-3
- Prevents formation of standing nodes
- Ensures monotonic decay

## Test Structure

### B1: Power Law Tail Analysis

**Purpose**: Validate A(r) ∝ r^(2β-3) behavior

**Algorithm**:
1. Compute radial profile A(r)
2. Filter tail region (exclude core)
3. Linear regression in log-log coordinates
4. Compare with theoretical slope 2β-3

**Criteria**:
- R² ≥ 0.99 on ≥1.5 decades
- Error ≤5% relative to theoretical slope
- Monotonic decay in tail region

### B2: Node Analysis

**Purpose**: Confirm absence of spherical standing nodes

**Algorithm**:
1. Compute radial derivative dA/dr
2. Count sign changes in derivative
3. Find amplitude zeros
4. Check for periodicity in zeros

**Criteria**:
- Sign changes ≤ 1
- No periodic zeros
- Monotonic decay

### B3: Topological Charge

**Purpose**: Validate topological stability

**Algorithm**:
1. Compute phase field φ = arg(a)
2. Create spherical integration contour
3. Integrate ∇φ·dl around contour
4. Normalize to 2π

**Criteria**:
- Error ≤1% from integer value
- Stable under smooth perturbations

### B4: Zone Separation

**Purpose**: Quantitatively separate field into zones

**Algorithm**:
1. Compute indicators N, S, C
2. Apply threshold criteria
3. Separate into core/transition/tail
4. Compute zone statistics

**Criteria**:
- Clear zone boundaries
- Proper amplitude ordering
- Balanced zone fractions

## Usage

### Basic Usage

```python
from bhlff.models.level_b import (
    LevelBPowerLawAnalyzer,
    LevelBNodeAnalyzer, 
    LevelBZoneAnalyzer
)

# Initialize analyzers
power_law_analyzer = LevelBPowerLawAnalyzer()
node_analyzer = LevelBNodeAnalyzer()
zone_analyzer = LevelBZoneAnalyzer()

# Analyze power law tail
result = power_law_analyzer.analyze_power_law_tail(field, beta, center)

# Check for nodes
node_result = node_analyzer.check_spherical_nodes(field, center)

# Separate zones
zone_result = zone_analyzer.separate_zones(field, center, thresholds)
```

### Comprehensive Testing

```python
from bhlff.tests.unit.test_level_b.test_fundamental_properties import LevelBFundamentalPropertiesTests

# Run all Level B tests
test_suite = LevelBFundamentalPropertiesTests()
results = test_suite.run_all_tests()
```

### Visualization

```python
from bhlff.models.level_b import LevelBVisualizer

# Create comprehensive report
visualizer = LevelBVisualizer()
visualizer.create_comprehensive_report(results, "output_directory")
```

## Configuration

### Test Parameters

```json
{
    "B1_power_law": {
        "domain": {"L": 10.0, "N": 512, "dimensions": 3},
        "physics": {"mu": 1.0, "beta": 1.0, "lambda": 0.0},
        "source": {"type": "point_source", "center": [5.0, 5.0, 5.0]},
        "analysis": {"min_decades": 1.5, "r_squared_threshold": 0.99}
    }
}
```

### Threshold Values

- **N_core**: 3.0 (core density threshold)
- **S_core**: 1.0 (core curvature threshold)  
- **N_tail**: 0.3 (tail density threshold)
- **S_tail**: 0.3 (tail curvature threshold)

## Output Data

### Analysis Results

- `power_law_analysis.json`: Power law fit parameters and quality metrics
- `nodes_analysis.json`: Node detection results and quality assessment
- `topological_charge.json`: Charge computation and error analysis
- `zone_separation.json`: Zone boundaries and statistics

### Visualizations

- `power_law_analysis.png`: Radial profile and log-log fit
- `node_analysis.png`: Derivative analysis and node detection
- `topological_analysis.png`: Charge computation and contour visualization
- `zone_analysis.png`: Zone maps and indicator visualization
- `summary_dashboard.png`: Overall test results and quality metrics

## Quality Metrics

### Power Law Analysis
- **R²**: Correlation coefficient (≥0.99)
- **Relative Error**: Slope error (≤5%)
- **Log Range**: Decade coverage (≥1.5)

### Node Analysis
- **Sign Changes**: Derivative sign changes (≤1)
- **Periodic Zeros**: Absence of periodic zeros
- **Monotonicity**: Monotonic decay verification

### Topological Charge
- **Charge Error**: Deviation from integer (≤1%)
- **Integration Quality**: Contour integration accuracy
- **Stability**: Robustness to perturbations

### Zone Separation
- **Zone Balance**: Balanced zone fractions
- **Amplitude Ordering**: Core > Tail amplitude
- **Boundary Clarity**: Clear zone boundaries

## Examples

See `examples/level_b_example.py` for comprehensive usage examples including:

- Basic analysis workflow
- Parameter variation studies
- Visualization generation
- Comprehensive testing

## Troubleshooting

### Common Issues

1. **Insufficient Range**: Increase domain size or resolution
2. **Poor Power Law Fit**: Check β value and λ=0 condition
3. **Node Detection**: Verify homogeneous medium conditions
4. **Zone Separation**: Adjust threshold values

### Performance Tips

- Use appropriate domain size for analysis range
- Ensure sufficient resolution for accurate derivatives
- Check boundary conditions for periodic systems
- Validate source placement and amplitude

## References

- 7D Phase Field Theory Documentation
- Riesz Operator Mathematical Foundation
- Level B Test Specifications
- BVP Framework Integration Guide
