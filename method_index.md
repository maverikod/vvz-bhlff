# Method Index Report
Generated for: .

## ./bhlff/core/bvp/abstract_bvp_facade.py
Methods: 11

### __init__(self, domain, config, domain_7d)
**Описание:** Initialize abstract BVP facade.

Physical Meaning:
    Sets up the base interface for the BVP framework with computational
    domains and configuration parameters.

Args:
    domain (Domain): Standar...

### solve_envelope(self, source)
**Описание:** Solve BVP envelope equation for U(1)³ phase structure.

Physical Meaning:
    Computes the envelope a(x,φ,t) of the Base High-Frequency Field
    in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ that modulates t...
**Декораторы:** abstractmethod

### detect_quenches(self, envelope)
**Описание:** Detect quench events when local thresholds are reached.

Physical Meaning:
    Identifies when BVP dissipatively "dumps" energy into
    the medium at local thresholds (amplitude/detuning/gradient).

...
**Декораторы:** abstractmethod

### compute_impedance(self, envelope)
**Описание:** Compute impedance/admittance from BVP envelope.

Physical Meaning:
    Calculates Y(ω), R(ω), T(ω), and peaks {ω_n,Q_n}
    from the BVP envelope at boundaries.

Args:
    envelope (np.ndarray): BVP e...
**Декораторы:** abstractmethod

### get_phase_vector(self)
**Описание:** Get U(1)³ phase vector structure.

Physical Meaning:
    Retrieves the U(1)³ phase vector Θ = (Θ₁, Θ₂, Θ₃) representing
    the three independent U(1) phase degrees of freedom.

Returns:
    Optional[...

### validate_configuration(self) -> bool
**Описание:** Validate BVP configuration parameters.

Physical Meaning:
    Ensures that the BVP configuration parameters are physically
    meaningful and mathematically consistent.

Returns:
    bool: True if con...

### is_7d_available(self) -> bool
**Описание:** Check if 7D domain is available.

Physical Meaning:
    Determines whether the 7D computational domain is available
    for full space-time operations.

Returns:
    bool: True if 7D domain is availab...

### get_7d_domain(self)
**Описание:** Get 7D domain if available.

Physical Meaning:
    Retrieves the 7D computational domain for full space-time
    operations if available.

Returns:
    Optional[Domain7D]: 7D domain or None if not ava...

### get_domain_info(self)
**Описание:** Get domain information.

Physical Meaning:
    Returns comprehensive information about the computational
    domains used by the BVP facade.

Returns:
    Dict[str, Any]: Domain information including ...

### get_configuration_info(self)
**Описание:** Get configuration information.

Physical Meaning:
    Returns information about the BVP configuration parameters
    for monitoring and analysis purposes.

Returns:
    Dict[str, Any]: Configuration i...

### __repr__(self) -> str
**Описание:** String representation of BVP facade.

## ./bhlff/core/bvp/abstract_solver_core.py
Methods: 8

### __init__(self, domain, config)
**Описание:** Initialize abstract solver core.

Physical Meaning:
    Sets up the base solver core with the computational domain
    and configuration parameters for solving the BVP equation.

Args:
    domain (Dom...

### compute_residual(self, envelope, source)
**Описание:** Compute residual of the BVP envelope equation.

Physical Meaning:
    Computes the residual r = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s(x,φ,t)
    for the Newton-Raphson method in 7D space-time.

Mathematical F...

### compute_jacobian(self, envelope)
**Описание:** Compute Jacobian matrix for Newton-Raphson method.

Physical Meaning:
    Computes the Jacobian matrix J = ∂r/∂a of the residual
    with respect to the envelope field.

Mathematical Foundation:
    J...

### solve_linear_system(self, jacobian, residual)
**Описание:** Solve linear system for Newton-Raphson update.

Physical Meaning:
    Solves the linear system J·δa = -r for the Newton-Raphson
    update δa, where J is the Jacobian and R is the residual.

Mathemati...

### solve_envelope(self, source, initial_guess)
**Описание:** Solve BVP envelope equation using Newton-Raphson method.

Physical Meaning:
    Solves the full BVP envelope equation for the envelope field
    in space-time using iterative Newton-Raphson method for...

### validate_solution(self, solution, source, tolerance)
**Описание:** Validate envelope equation solution.

Physical Meaning:
    Validates that the solution satisfies the envelope equation
    within the specified tolerance by computing the residual.

Mathematical Foun...

### get_solver_parameters(self)
**Описание:** Get solver parameters.

Physical Meaning:
    Returns the current values of all solver parameters for
    monitoring and analysis purposes.

Returns:
    Dict[str, Any]: Dictionary containing solver p...

### __repr__(self) -> str
**Описание:** String representation of solver core.

## ./bhlff/core/bvp/analysis/resonance_optimization.py
Methods: 5

### __init__(self, constants)
**Описание:** Initialize resonance optimizer.

Physical Meaning:
    Sets up the optimizer with BVP constants for
    resonance quality factor optimization.

Args:
    constants (BVPConstants): BVP constants instan...

### optimize_quality_factors(self, frequencies, magnitude, peak_indices)
**Описание:** Optimize quality factors using advanced fitting techniques.

Physical Meaning:
    Optimizes quality factors using advanced fitting techniques
    to improve accuracy and reliability of resonance anal...

### _extract_peak_region(self, frequencies, magnitude, peak_idx)
**Описание:** Extract region around a resonance peak.

Physical Meaning:
    Extracts a localized region around a resonance peak for
    detailed analysis, ensuring sufficient data for accurate
    optimization whi...

### _advanced_lorentzian_fitting(self, peak_region)
**Описание:** Perform advanced Lorentzian fitting.

Physical Meaning:
    Performs advanced Lorentzian fitting to extract accurate
    resonance parameters, including amplitude, center frequency,
    and full width...

### _calculate_optimized_quality_factor(self, params) -> float
**Описание:** Calculate optimized quality factor.

Physical Meaning:
    Calculates the optimized quality factor from advanced
    fitting parameters, applying corrections for noise
    and other systematic effects...

## ./bhlff/core/bvp/analysis/resonance_quality_analysis.py
Methods: 11

### __init__(self, constants)
**Описание:** Initialize advanced quality analyzer.

Physical Meaning:
    Sets up the analyzer with BVP constants and initializes
    optimization and statistics modules for comprehensive analysis.

Args:
    cons...

### analyze_resonance_characteristics(self, frequencies, magnitude, peak_indices)
**Описание:** Analyze comprehensive resonance characteristics.

Physical Meaning:
    Performs comprehensive analysis of resonance characteristics,
    including quality factors, resonance shapes, and
    frequency...

### compare_resonance_quality(self, quality_factors_1, quality_factors_2)
**Описание:** Compare quality factors between two sets of resonances.

Physical Meaning:
    Compares quality factors between two sets of resonances
    to analyze differences in resonance characteristics and
    i...

### _extract_peak_region(self, frequencies, magnitude, peak_idx)
**Описание:** Extract region around a resonance peak.

Physical Meaning:
    Extracts a localized region around a resonance peak for
    detailed analysis, ensuring sufficient data for accurate
    characterization...

### _analyze_resonance_shape(self, peak_region)
**Описание:** Analyze resonance shape characteristics.

Physical Meaning:
    Analyzes the shape characteristics of a resonance peak,
    including amplitude, width, and symmetry, which are
    crucial for understa...

### _analyze_frequency_properties(self, peak_region)
**Описание:** Analyze frequency properties.

Physical Meaning:
    Analyzes frequency-domain properties of the resonance,
    including center frequency, frequency span, and resolution,
    which are essential for ...

### _analyze_amplitude_properties(self, peak_region)
**Описание:** Analyze amplitude properties.

Physical Meaning:
    Analyzes amplitude characteristics of the resonance,
    including maximum, minimum, mean, and standard deviation,
    which provide insights into ...

### _classify_resonance_type(self, peak_region) -> str
**Описание:** Classify resonance type.

Physical Meaning:
    Classifies the resonance type based on its characteristics,
    which helps in understanding the nature of BVP impedance
    variations and resonance be...

### _calculate_quality_factor_from_characteristics(self, resonance_shape, frequency_properties) -> float
**Описание:** Calculate quality factor from resonance characteristics.

Physical Meaning:
    Calculates the quality factor from resonance characteristics,
    which is a key parameter for characterizing BVP impeda...

### _calculate_peak_width(self, magnitude, peak_idx) -> float
**Описание:** Calculate peak width.

Physical Meaning:
    Calculates the width of the resonance peak, which is
    essential for quality factor determination and
    resonance characterization.

Args:
    magnitud...

### _calculate_peak_symmetry(self, magnitude, peak_idx) -> float
**Описание:** Calculate peak symmetry.

Physical Meaning:
    Calculates the symmetry of the resonance peak, which
    provides insights into the linearity and quality of
    BVP impedance characteristics.

Mathema...

## ./bhlff/core/bvp/analysis/resonance_statistics.py
Methods: 6

### __init__(self, constants)
**Описание:** Initialize resonance statistics analyzer.

Physical Meaning:
    Sets up the statistics analyzer with BVP constants for
    statistical analysis of resonance properties.

Args:
    constants (BVPConst...

### compare_quality_factors(self, quality_factors_1, quality_factors_2)
**Описание:** Compare quality factors between two sets of resonances.

Physical Meaning:
    Compares quality factors between two sets of resonances
    to analyze differences in resonance characteristics and
    i...

### analyze_quality_factor_distribution(self, quality_factors)
**Описание:** Analyze quality factor distribution.

Physical Meaning:
    Analyzes the statistical distribution of quality factors
    to understand the variability and characteristics of
    BVP impedance resonanc...

### _calculate_correlation(self, qf1, qf2) -> float
**Описание:** Calculate correlation between two quality factor sets.

Physical Meaning:
    Calculates the correlation coefficient between two sets
    of quality factors to assess their relationship.

Mathematical...

### _calculate_skewness(self, data, mean, std) -> float
**Описание:** Calculate skewness of the distribution.

Physical Meaning:
    Calculates the skewness (third moment) of the quality factor
    distribution to assess asymmetry.

Mathematical Foundation:
    Skewness...

### _calculate_kurtosis(self, data, mean, std) -> float
**Описание:** Calculate kurtosis of the distribution.

Physical Meaning:
    Calculates the kurtosis (fourth moment) of the quality factor
    distribution to assess tail heaviness.

Mathematical Foundation:
    Ku...

## ./bhlff/core/bvp/boundary/step_resonator.py
Methods: 1

### apply_step_resonator(field, axes, R, T)
**Описание:** Apply semi-transparent step resonator boundary conditions in-place.

Args:
    field: N-dimensional complex or real field (supports 7D arrays)
    axes: axes along which to apply boundary mixing (e.g....

## ./bhlff/core/bvp/bvp_constants.py
Methods: 4

### __init__(self, config)
**Описание:** Initialize unified BVP constants.

Physical Meaning:
    Sets up all BVP constants by initializing base, advanced, and
    numerical constant modules with the provided configuration.

Args:
    config...

### get_material_property(self, property_name) -> float
**Описание:** Get material property constant (unified interface).

Physical Meaning:
    Provides unified access to both basic and advanced material
    properties through a single interface.

Args:
    property_na...

### get_all_constants(self)
**Описание:** Get all BVP constants as a dictionary.

Physical Meaning:
    Returns all BVP constants for monitoring and analysis purposes.

Returns:
    Dict[str, Any]: Dictionary containing all constants.

### __repr__(self) -> str
**Описание:** String representation of unified BVP constants.

## ./bhlff/core/bvp/bvp_constants_base.py
Methods: 13

### __init__(self, config)
**Описание:** Initialize base BVP constants with optional configuration override.

Physical Meaning:
    Sets up fundamental physical constants with values from configuration
    or uses scientifically accurate def...

### _setup_envelope_constants(self)
**Описание:** Setup envelope equation constants.

### _setup_material_constants(self)
**Описание:** Setup material property constants with frequency-dependent models.

### get_conductivity(self, frequency) -> float
**Описание:** Compute frequency-dependent conductivity σ(ω).

### get_admittance(self, frequency) -> float
**Описание:** Compute frequency-dependent base admittance Y(ω).

### _setup_physical_constants(self)
**Описание:** Setup fundamental physical constants.

### get_envelope_parameter(self, parameter_name) -> float
**Описание:** Get envelope equation parameter.

Args:
    parameter_name (str): Name of the parameter.

Returns:
    float: Parameter value.

### get_basic_material_property(self, property_name) -> float
**Описание:** Get basic material property constant.

Args:
    property_name (str): Name of the material property.

Returns:
    float: Property value.

### get_physical_constant(self, constant_name) -> float
**Описание:** Get fundamental physical constant.

Args:
    constant_name (str): Name of the physical constant.

Returns:
    float: Constant value.

### get_physical_parameter(self, parameter_name) -> float
**Описание:** Get physical parameter value.

Physical Meaning:
    Retrieves physical parameters used in BVP postulates and calculations.

Args:
    parameter_name (str): Name of the physical parameter.

Returns:
 ...

### get_carrier_frequency(self) -> float
**Описание:** Get BVP carrier frequency.

Physical Meaning:
    Returns the high-frequency carrier frequency ω₀ of the BVP field,
    which is the fundamental frequency that all envelope modulations
    and beating...

### get_quench_parameter(self, parameter_name) -> float
**Описание:** Get quench detection parameter value.

Physical Meaning:
    Retrieves parameters used for quench detection in BVP postulates.

Args:
    parameter_name (str): Name of the quench parameter.

Returns:
...

### __repr__(self) -> str
**Описание:** String representation of base BVP constants.

## ./bhlff/core/bvp/bvp_constants_numerical.py
Methods: 8

### __init__(self, config)
**Описание:** Initialize numerical BVP constants.

Physical Meaning:
    Sets up numerical solver parameters, quench detection thresholds,
    and impedance calculation parameters.

Args:
    config (Dict[str, Any]...

### _setup_numerical_constants(self)
**Описание:** Setup numerical solver constants.

### _setup_quench_constants(self)
**Описание:** Setup quench detection constants.

### _setup_impedance_constants(self)
**Описание:** Setup impedance calculation constants.

### get_numerical_parameter(self, parameter_name) -> float
**Описание:** Get numerical solver parameter.

Args:
    parameter_name (str): Name of the numerical parameter.

Returns:
    float: Parameter value.

### get_quench_threshold(self, threshold_name) -> float
**Описание:** Get quench detection threshold.

Args:
    threshold_name (str): Name of the threshold.

Returns:
    float: Threshold value.

### get_impedance_parameter(self, parameter_name) -> Any
**Описание:** Get impedance calculation parameter.

Args:
    parameter_name (str): Name of the impedance parameter.

Returns:
    Any: Parameter value.

### __repr__(self) -> str
**Описание:** String representation of numerical BVP constants.

## ./bhlff/core/bvp/bvp_core/bvp_7d_interface.py
Methods: 8

### __init__(self, domain_7d, config)
**Описание:** Initialize BVP 7D interface.

Physical Meaning:
    Sets up the 7D interface with the 7D computational domain
    and configuration parameters, initializing the 7D envelope
    equation solver and pos...

### _setup_7d_components(self)
**Описание:** Setup 7D components for envelope equation and postulates.

Physical Meaning:
    Initializes the 7D envelope equation solver and postulates
    validator for working with the full 7D space-time struct...

### solve_envelope_7d(self, source_7d)
**Описание:** Solve 7D BVP envelope equation.

Physical Meaning:
    Solves the full 7D envelope equation in space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ
    using the 7D envelope equation solver. This provides the complete
   ...

### validate_postulates_7d(self, envelope_7d)
**Описание:** Validate all 9 BVP postulates for 7D field.

Physical Meaning:
    Validates all 9 BVP postulates to ensure the 7D field
    satisfies the fundamental properties of the BVP framework.
    This compreh...

### get_7d_domain(self) -> Domain7D
**Описание:** Get the 7D domain.

Physical Meaning:
    Returns the 7D computational domain M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ
    for accessing spatial, phase, and temporal configurations.

Returns:
    Domain7D: The 7D space-t...

### get_7d_envelope_equation(self) -> BVPEnvelopeEquation7D
**Описание:** Get the 7D envelope equation solver.

Physical Meaning:
    Returns the 7D envelope equation solver for direct access
    to 7D envelope equation operations and parameters.

Returns:
    BVPEnvelopeEq...

### get_7d_postulates(self) -> BVPPostulates7D
**Описание:** Get the 7D postulates validator.

Physical Meaning:
    Returns the 7D postulates validator for direct access
    to individual postulate validation and analysis.

Returns:
    BVPPostulates7D: The 7D...

### get_7d_parameters(self)
**Описание:** Get 7D interface parameters.

Physical Meaning:
    Returns the current parameters of the 7D interface
    including envelope equation and postulates parameters.

Returns:
    Dict[str, Any]: Dictiona...

## ./bhlff/core/bvp/bvp_core/bvp_core_facade_impl.py
Methods: 12

### __init__(self, domain, config, domain_7d)
**Описание:** Initialize BVP core facade implementation.

Physical Meaning:
    Sets up the concrete implementation for the BVP framework with
    computational domains and configuration parameters, initializing
  ...

### solve_envelope(self, source)
**Описание:** Solve BVP envelope equation for U(1)³ phase structure.

Physical Meaning:
    Computes the envelope a(x,φ,t) of the Base High-Frequency Field
    in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ that modulates t...

### solve_envelope_7d(self, source_7d)
**Описание:** Solve 7D BVP envelope equation.

Physical Meaning:
    Solves the full 7D envelope equation in space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ
    using the 7D envelope equation solver. This provides the complete
   ...

### validate_postulates_7d(self, envelope_7d)
**Описание:** Validate all 9 BVP postulates in 7D space-time.

Physical Meaning:
    Validates all 9 BVP postulates against the 7D envelope solution,
    ensuring physical consistency and theoretical correctness.

...

### detect_quenches(self, envelope)
**Описание:** Detect quench events when local thresholds are reached.

Physical Meaning:
    Identifies when BVP dissipatively "dumps" energy into
    the medium at local thresholds (amplitude/detuning/gradient).

...

### compute_impedance(self, envelope)
**Описание:** Compute impedance/admittance from BVP envelope.

Physical Meaning:
    Calculates Y(ω), R(ω), T(ω), and peaks {ω_n,Q_n}
    from the BVP envelope at boundaries.

Mathematical Foundation:
    Computes ...

### get_phase_vector(self)
**Описание:** Get U(1)³ phase vector structure.

Physical Meaning:
    Retrieves the U(1)³ phase vector Θ = (Θ₁, Θ₂, Θ₃) representing
    the three independent U(1) phase degrees of freedom.

Returns:
    Optional[...

### get_bvp_constants(self) -> BVPConstants
**Описание:** Get BVP constants and configuration.

Physical Meaning:
    Retrieves the BVP constants and configuration parameters
    used in the framework.

Returns:
    BVPConstants: BVP constants instance.

### get_7d_interface(self)
**Описание:** Get 7D interface if available.

Physical Meaning:
    Retrieves the 7D interface for full space-time operations
    if a 7D domain was provided during initialization.

Returns:
    Optional[BVPCore7DI...

### get_phase_operations(self)
**Описание:** Get phase operations interface.

Physical Meaning:
    Retrieves the phase operations interface for U(1)³ phase
    structure analysis and manipulation.

Returns:
    Phase operations interface.

### get_parameter_access(self)
**Описание:** Get parameter access interface.

Physical Meaning:
    Retrieves the parameter access interface for BVP constants
    and configuration management.

Returns:
    Parameter access interface.

### __repr__(self) -> str
**Описание:** String representation of BVP core facade.

## ./bhlff/core/bvp/bvp_core/bvp_operations.py
Methods: 13

### __init__(self, domain, config, domain_7d)
**Описание:** Initialize BVP core operations.

Physical Meaning:
    Sets up the core operations with the computational domains
    and configuration parameters, initializing all necessary
    components for BVP op...

### _setup_phase_vector(self)
**Описание:** Setup phase vector for U(1)³ phase structure.

### _setup_envelope_solver(self)
**Описание:** Setup envelope solver for BVP equation.

### _setup_quench_detector(self)
**Описание:** Setup quench detector for threshold events.

### _setup_impedance_calculator(self)
**Описание:** Setup impedance calculator for boundary analysis.

### _setup_phase_operations(self)
**Описание:** Setup phase operations for U(1)³ structure.

### _setup_parameter_access(self)
**Описание:** Setup parameter access for configuration management.

### solve_envelope(self, source)
**Описание:** Solve BVP envelope equation for U(1)³ phase structure.

Physical Meaning:
    Computes the envelope a(x,φ,t) of the Base High-Frequency Field
    in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ that modulates t...

### detect_quenches(self, envelope)
**Описание:** Detect quench events when local thresholds are reached.

Physical Meaning:
    Identifies when BVP dissipatively "dumps" energy into
    the medium at local thresholds (amplitude/detuning/gradient).

...

### compute_impedance(self, envelope)
**Описание:** Compute impedance/admittance from BVP envelope.

Physical Meaning:
    Calculates Y(ω), R(ω), T(ω), and peaks {ω_n,Q_n}
    from the BVP envelope at boundaries.

Mathematical Foundation:
    Computes ...

### get_phase_vector(self) -> PhaseVector
**Описание:** Get phase vector for U(1)³ phase structure.

Physical Meaning:
    Returns the phase vector containing the three U(1) phase
    components Θ_a (a=1..3) that represent the BVP field
    structure in 7D...

### get_phase_operations(self) -> BVPPhaseOperations
**Описание:** Get phase operations for U(1)³ structure.

Physical Meaning:
    Returns the phase operations object for working with
    the U(1)³ phase structure of the BVP field.

Returns:
    BVPPhaseOperations: ...

### get_parameter_access(self) -> BVPParameterAccess
**Описание:** Get parameter access for configuration management.

Physical Meaning:
    Returns the parameter access object for managing
    BVP configuration parameters and settings.

Returns:
    BVPParameterAcce...

## ./bhlff/core/bvp/bvp_envelope_solver.py
Methods: 10

### __init__(self, domain, config, constants)
**Описание:** Initialize envelope equation solver.

Physical Meaning:
    Sets up the solver with parameters for the nonlinear
    envelope equation including stiffness and susceptibility
    coefficients.

Args:
 ...

### _setup_parameters(self)
**Описание:** Setup envelope equation parameters.

Physical Meaning:
    Initializes the base physical parameters for the envelope equation
    from the constants instance. These are used as base values for
    com...

### _setup_solver_components(self)
**Описание:** Setup solver components.

### solve_envelope(self, source)
**Описание:** Solve 7D BVP envelope equation.

Physical Meaning:
    Computes the envelope a(x,φ,t) of the Base High-Frequency Field
    in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ that modulates the high-frequency carri...
**Декораторы:** <ast.Call object at 0x753e6f6038d0>

### solve_envelope_linearized(self, source)
**Описание:** Solve linearized 7D BVP envelope equation.

Physical Meaning:
    Solves the linearized version of the envelope equation
    ∇·(κ₀∇a) + k₀²χ'a = s(x,φ,t) for initial guess generation.

Mathematical Fo...
**Декораторы:** <ast.Call object at 0x753e6f66e3d0>

### get_parameters(self)
**Описание:** Get envelope equation parameters.

Physical Meaning:
    Returns the current base parameters for the envelope equation.
    Note: Actual coefficients are computed dynamically as functions
    of field...

### get_nonlinear_coefficients(self, envelope)
**Описание:** Get nonlinear coefficients for given envelope.

Physical Meaning:
    Computes and returns the nonlinear coefficients κ(|a|) and χ(|a|)
    for the given envelope field, showing how they depend on
   ...

### get_memory_info(self)
**Описание:** Get memory information and usage statistics.

Physical Meaning:
    Returns current memory usage statistics for monitoring
    and debugging purposes.

Returns:
    Dict[str, Any]: Memory information ...

### validate_solution(self, solution, source, tolerance)
**Описание:** Validate envelope equation solution.

Physical Meaning:
    Validates that the solution satisfies the envelope equation
    within the specified tolerance by computing the residual.

Mathematical Foun...

### __repr__(self) -> str
**Описание:** String representation of envelope solver.

## ./bhlff/core/bvp/bvp_impedance_calculator.py
Methods: 8

### __init__(self, domain, config, constants)
**Описание:** Initialize impedance calculator.

Physical Meaning:
    Sets up the calculator with configuration for frequency
    response analysis and resonance detection.

Args:
    domain (Domain): Computational...

### _setup_components(self)
**Описание:** Setup impedance calculation components.

Physical Meaning:
    Initializes the core impedance calculation and resonance
    detection components.

### compute_impedance(self, envelope)
**Описание:** Compute impedance/admittance from BVP envelope.

Physical Meaning:
    Calculates Y(ω), R(ω), T(ω), and peaks {ω_n,Q_n}
    from the BVP envelope at boundaries.

Mathematical Foundation:
    Computes ...

### get_parameters(self)
**Описание:** Get impedance calculation parameters.

Physical Meaning:
    Returns the current parameters for impedance calculation.

Returns:
    Dict[str, Any]: Impedance calculation parameters.

### set_quality_factor_threshold(self, threshold)
**Описание:** Set quality factor threshold.

Physical Meaning:
    Updates the threshold for quality factor filtering
    in resonance detection.

Args:
    threshold (float): New quality factor threshold.

### get_impedance_core(self) -> ImpedanceCore
**Описание:** Get impedance core component.

Physical Meaning:
    Returns the core impedance calculation component
    for advanced operations.

Returns:
    ImpedanceCore: Core impedance calculation component.

### get_resonance_detector(self) -> ResonanceDetector
**Описание:** Get resonance detector component.

Physical Meaning:
    Returns the resonance detection component
    for advanced peak analysis.

Returns:
    ResonanceDetector: Resonance detection component.

### __repr__(self) -> str
**Описание:** String representation of impedance calculator.

## ./bhlff/core/bvp/bvp_level_integration.py
Methods: 10

### __init__(self, bvp_core)
**Описание:** Initialize BVP level integration.

Physical Meaning:
    Sets up integration interfaces for all levels A-G with
    the BVP core framework.

### get_level_a_data(self, envelope)
**Описание:** Get Level A data from BVP envelope.

### get_level_b_data(self, envelope)
**Описание:** Get Level B data from BVP envelope.

### get_level_c_data(self, envelope)
**Описание:** Get Level C data from BVP envelope.

### get_level_d_data(self, envelope)
**Описание:** Get Level D data from BVP envelope.

### get_level_e_data(self, envelope)
**Описание:** Get Level E data from BVP envelope.

### get_level_f_data(self, envelope)
**Описание:** Get Level F data from BVP envelope.

### get_level_g_data(self, envelope)
**Описание:** Get Level G data from BVP envelope.

### get_all_levels_data(self, envelope)
**Описание:** Get data for all levels A-G from BVP envelope.

### validate_bvp_integration(self, envelope) -> bool
**Описание:** Validate BVP integration with all levels.

Physical Meaning:
    Ensures that BVP envelope data is properly integrated
    with all levels A-G and maintains framework compliance.

## ./bhlff/core/bvp/bvp_level_interface_base.py
Methods: 1

### process_bvp_data(self, envelope)
**Описание:** Process BVP envelope data for this level.

Physical Meaning:
    Transforms BVP envelope data according to level-specific
    requirements while maintaining BVP framework compliance.

Args:
    envelo...
**Декораторы:** abstractmethod

## ./bhlff/core/bvp/bvp_level_interfaces_g.py
Methods: 6

### __init__(self, bvp_core)
**Описание:** Нет докстринга

### process_bvp_data(self, envelope)
**Описание:** Process BVP data for Level G operations.

Physical Meaning:
    Analyzes cosmological evolution, large-scale structure,
    astrophysical objects, and gravitational effects in BVP envelope.

### _analyze_cosmological_evolution(self, envelope)
**Описание:** Analyze cosmological evolution.

### _analyze_large_scale_structure(self, envelope)
**Описание:** Analyze large-scale structure.

### _analyze_astrophysical_objects(self, envelope)
**Описание:** Analyze astrophysical objects.

### _analyze_gravitational_effects(self, envelope)
**Описание:** Analyze gravitational effects.

## ./bhlff/core/bvp/bvp_parameter_access.py
Methods: 6

### __init__(self, constants, envelope_solver, quench_detector, impedance_calculator)
**Описание:** Initialize parameter access.

Physical Meaning:
    Sets up parameter access with references to BVP components
    that contain the configuration parameters.

Args:
    constants (BVPConstants): BVP p...

### get_carrier_frequency(self) -> float
**Описание:** Get the high-frequency carrier frequency.

Physical Meaning:
    Returns the frequency ω₀ of the high-frequency carrier
    that is modulated by the envelope.

Returns:
    float: Carrier frequency ω₀...

### get_envelope_parameters(self)
**Описание:** Get envelope equation parameters.

Physical Meaning:
    Returns the parameters κ₀, κ₂, χ', χ'' for the
    envelope equation.

Returns:
    Dict[str, float]: Envelope equation parameters.

### get_quench_thresholds(self)
**Описание:** Get quench detection thresholds.

Physical Meaning:
    Returns the current threshold values used for quench detection.

Returns:
    Dict[str, float]: Quench detection thresholds.

### set_quench_thresholds(self, thresholds)
**Описание:** Set new quench detection thresholds.

Physical Meaning:
    Updates the threshold values used for quench detection.

Args:
    thresholds (Dict[str, float]): New threshold values.

### get_impedance_parameters(self)
**Описание:** Get impedance calculation parameters.

Physical Meaning:
    Returns the current parameters for impedance calculation.

Returns:
    Dict[str, Any]: Impedance calculation parameters.

## ./bhlff/core/bvp/bvp_phase_operations.py
Methods: 7

### __init__(self, phase_vector)
**Описание:** Initialize phase operations.

Physical Meaning:
    Sets up phase operations with access to the U(1)³
    phase vector structure.

Args:
    phase_vector (PhaseVector): U(1)³ phase vector structure.

### get_phase_components(self)
**Описание:** Get the three U(1) phase components Θ_a (a=1..3).

Physical Meaning:
    Returns the three independent U(1) phase components
    that form the U(1)³ structure of the BVP field.

Returns:
    List[np.n...

### get_total_phase(self)
**Описание:** Get the total phase from U(1)³ structure.

Physical Meaning:
    Computes the total phase by combining the three
    U(1) components with proper SU(2) coupling.

Returns:
    np.ndarray: Total phase f...

### compute_electroweak_currents(self, envelope)
**Описание:** Compute electroweak currents as functionals of the envelope.

Physical Meaning:
    Computes electromagnetic and weak currents that are
    generated as functionals of the BVP envelope through
    the...

### compute_phase_coherence(self)
**Описание:** Compute phase coherence measure.

Physical Meaning:
    Computes a measure of phase coherence across the
    U(1)³ structure, indicating the degree of
    synchronization between the three phase compo...

### get_su2_coupling_strength(self) -> float
**Описание:** Get the SU(2) coupling strength.

Physical Meaning:
    Returns the strength of the weak hierarchical
    coupling to SU(2)/core.

Returns:
    float: SU(2) coupling strength.

### set_su2_coupling_strength(self, strength)
**Описание:** Set the SU(2) coupling strength.

Physical Meaning:
    Updates the strength of the weak hierarchical
    coupling to SU(2)/core.

Args:
    strength (float): New SU(2) coupling strength.

## ./bhlff/core/bvp/bvp_postulate_base.py
Methods: 1

### apply(self, envelope)
**Описание:** Apply the postulate to the envelope.

Physical Meaning:
    Performs the mathematical operation specific to this
    postulate to validate BVP field properties.

Mathematical Foundation:
    Each post...
**Декораторы:** abstractmethod

## ./bhlff/core/bvp/bvp_postulates.py
Methods: 7

### __init__(self, domain, constants)
**Описание:** Initialize BVP postulates interface.

Physical Meaning:
    Sets up all 9 BVP postulates with domain and constants
    for comprehensive field validation.

Args:
    domain (Domain): Computational dom...

### apply_all_postulates(self, envelope)
**Описание:** Apply all BVP postulates to the envelope.

Physical Meaning:
    Applies all 9 BVP postulates to verify that the envelope
    satisfies the BVP framework requirements.

Args:
    envelope (np.ndarray)...

### validate_bvp_framework(self, envelope) -> bool
**Описание:** Validate BVP framework compliance.

Physical Meaning:
    Checks if the envelope satisfies all BVP postulates,
    indicating proper BVP framework compliance.

Args:
    envelope (np.ndarray): BVP env...

### get_postulate_summary(self, envelope)
**Описание:** Get summary of postulate satisfaction.

Physical Meaning:
    Provides a quick overview of which postulates are
    satisfied and which need attention.

Args:
    envelope (np.ndarray): BVP envelope t...

### get_failed_postulates(self, envelope)
**Описание:** Get list of failed postulates.

Physical Meaning:
    Identifies which postulates are not satisfied,
    helping to diagnose field issues.

Args:
    envelope (np.ndarray): BVP envelope to analyze.

R...

### get_postulate_quality_scores(self, envelope)
**Описание:** Get quality scores for each postulate.

Physical Meaning:
    Provides quantitative measures of how well each
    postulate is satisfied.

Args:
    envelope (np.ndarray): BVP envelope to analyze.

Re...

### _extract_quality_score(self, result) -> float
**Описание:** Extract quality score from postulate result.

Physical Meaning:
    Computes a normalized quality score (0.0 to 1.0)
    from postulate analysis results.

Args:
    result (Dict[str, Any]): Postulate ...

## ./bhlff/core/bvp/bvp_rigidity_postulate.py
Methods: 8

### __init__(self, domain, constants)
**Описание:** Initialize BVP rigidity postulate.

Physical Meaning:
    Sets up the postulate with domain and constants for
    analyzing BVP field rigidity properties.

Args:
    domain (Domain): Computational dom...

### apply(self, envelope)
**Описание:** Apply BVP rigidity postulate.

Physical Meaning:
    Verifies that BVP field exhibits high stiffness and
    short correlation length, indicating rigidity.

Mathematical Foundation:
    Analyzes field...

### _analyze_field_stiffness(self, envelope)
**Описание:** Analyze field stiffness from spatial gradients.

Physical Meaning:
    Computes field stiffness from second derivatives and
    gradient magnitudes, indicating resistance to deformation.

Mathematical...

### _analyze_correlation_length(self, envelope)
**Описание:** Analyze correlation length of the field.

Physical Meaning:
    Computes correlation length from spatial correlation
    function, indicating field coherence length.

Mathematical Foundation:
    Corr...

### _compute_autocorrelation(self, amplitude, axis)
**Описание:** Compute autocorrelation function along specified axis.

Physical Meaning:
    Calculates spatial autocorrelation to determine
    field coherence properties.

Args:
    amplitude (np.ndarray): Field a...

### _extract_correlation_length(self, autocorr, axis) -> float
**Описание:** Extract correlation length from autocorrelation function.

Physical Meaning:
    Finds characteristic length where autocorrelation
    drops to 1/e of its maximum value.

Args:
    autocorr (np.ndarra...

### _check_rigidity_properties(self, stiffness_analysis, correlation_analysis)
**Описание:** Check rigidity properties of the BVP field.

Physical Meaning:
    Verifies that field exhibits high stiffness and
    short correlation length, indicating rigidity.

Args:
    stiffness_analysis (Dic...

### _validate_bvp_rigidity(self, rigidity_properties) -> bool
**Описание:** Validate BVP rigidity postulate.

Physical Meaning:
    Checks that field exhibits sufficient rigidity
    properties for BVP framework validity.

Args:
    rigidity_properties (Dict[str, Any]): Rigid...

## ./bhlff/core/bvp/carrier_primacy_postulate.py
Methods: 8

### __init__(self, domain, constants)
**Описание:** Initialize carrier primacy postulate.

Physical Meaning:
    Sets up the postulate with domain and constants for
    analyzing carrier-envelope scale separation.

Args:
    domain (Domain): Computatio...

### apply(self, envelope)
**Описание:** Apply carrier primacy postulate.

Physical Meaning:
    Verifies that carrier frequency dominates over envelope
    frequencies and envelope represents small modulation.

Mathematical Foundation:
    ...

### _analyze_frequency_spectrum(self, envelope)
**Описание:** Analyze frequency spectrum of the envelope.

Physical Meaning:
    Performs FFT analysis to identify dominant frequencies
    and their relative magnitudes.

Args:
    envelope (np.ndarray): BVP envel...

### _find_dominant_frequencies(self, spectrum, freq_axis)
**Описание:** Find dominant frequencies in the spectrum.

Physical Meaning:
    Identifies peaks in frequency spectrum that correspond
    to significant frequency components.

Args:
    spectrum (np.ndarray): Freq...

### _compute_frequency_statistics(self, spectrum, freq_axis)
**Описание:** Compute frequency statistics.

Physical Meaning:
    Calculates statistical measures of frequency distribution
    to characterize spectrum properties.

Args:
    spectrum (np.ndarray): Frequency spec...

### _check_scale_separation(self, frequency_analysis)
**Описание:** Check scale separation between carrier and envelope.

Physical Meaning:
    Verifies that carrier frequency is much higher than
    envelope frequencies, ensuring proper scale separation.

Args:
    f...

### _analyze_modulation_strength(self, envelope)
**Описание:** Analyze modulation strength of the envelope.

Physical Meaning:
    Quantifies how small the envelope modulation is
    compared to the carrier amplitude.

Args:
    envelope (np.ndarray): BVP envelop...

### _validate_carrier_primacy(self, scale_separation, modulation_analysis) -> bool
**Описание:** Validate carrier primacy postulate.

Physical Meaning:
    Checks that both scale separation and small modulation
    conditions are satisfied.

Args:
    scale_separation (Dict[str, Any]): Scale sepa...

## ./bhlff/core/bvp/constants/bvp_constants_advanced.py
Methods: 9

### __init__(self, config)
**Описание:** Initialize advanced BVP constants.

Physical Meaning:
    Sets up advanced material properties and frequency-dependent
    calculation parameters.

Args:
    config (Dict[str, Any], optional): Configu...

### _setup_advanced_material_constants(self)
**Описание:** Setup advanced material property constants.

### get_advanced_material_property(self, property_name) -> float
**Описание:** Get advanced material property constant.

Args:
    property_name (str): Name of the material property.

Returns:
    float: Property value.

### compute_frequency_dependent_conductivity(self, frequency) -> float
**Описание:** Compute frequency-dependent conductivity using advanced Drude-Lorentz model.

Physical Meaning:
    Computes conductivity using the Drude-Lorentz model for free electrons
    with frequency-dependent ...

### compute_frequency_dependent_capacitance(self, frequency) -> float
**Описание:** Compute frequency-dependent capacitance using advanced Debye-Cole model.

Physical Meaning:
    Computes capacitance using the Debye-Cole model for dielectric
    relaxation with frequency-dependent p...

### compute_frequency_dependent_inductance(self, frequency) -> float
**Описание:** Compute frequency-dependent inductance using advanced skin effect and proximity models.

Physical Meaning:
    Computes inductance considering skin effect, proximity effect,
    and frequency-dependen...

### compute_nonlinear_admittance_coefficients(self, frequency, amplitude)
**Описание:** Compute nonlinear admittance coefficients using advanced field theory.

Physical Meaning:
    Computes frequency and amplitude dependent coefficients for
    nonlinear admittance using full electromag...

### compute_renormalized_coefficients(self, amplitude, gradient_magnitude_squared)
**Описание:** Compute renormalized coefficients using advanced field theory.

Physical Meaning:
    Computes amplitude and gradient dependent coefficients
    using full quantum field theory with renormalization gr...

### __repr__(self) -> str
**Описание:** String representation of advanced BVP constants.

## ./bhlff/core/bvp/constants/frequency_dependent_properties.py
Methods: 12

### __init__(self, constants, domain)
**Описание:** Initialize frequency-dependent properties calculator.

Physical Meaning:
    Sets up the frequency-dependent property calculations
    with access to BVP constants.

Args:
    constants: BVP constants...

### _setup_frequency_arrays(self, domain)
**Описание:** Setup frequency arrays from domain.

### compute_conductivity(self, frequency) -> float
**Описание:** Compute frequency-dependent conductivity using advanced Drude-Lorentz model.

Physical Meaning:
    Computes conductivity using the Drude-Lorentz model for free electrons
    with frequency-dependent ...

### compute_capacitance(self, frequency) -> float
**Описание:** Compute frequency-dependent capacitance using advanced Debye-Cole model.

Physical Meaning:
    Computes capacitance using the Debye-Cole model for dielectric
    relaxation with frequency-dependent p...

### compute_inductance(self, frequency) -> float
**Описание:** Compute frequency-dependent inductance using advanced skin effect and proximity models.

Physical Meaning:
    Computes inductance considering skin effect, proximity effect,
    and frequency-dependen...

### compute_susceptibility(self, frequency) -> complex
**Описание:** Compute frequency-dependent susceptibility.

Physical Meaning:
    Computes the complex susceptibility χ(ω) = χ'(ω) + iχ''(ω)
    representing the material's response to electromagnetic fields.
    
A...

### compute_dispersion_relation(self, frequency) -> float
**Описание:** Compute dispersion relation k(ω).

Physical Meaning:
    Computes the wave number k as a function of frequency ω
    based on the dispersion relation.
    
Args:
    frequency (float): Frequency in ra...

### compute_phase_velocity(self, frequency) -> float
**Описание:** Compute phase velocity v_phase = ω/k.

Physical Meaning:
    Computes the phase velocity of electromagnetic waves
    in the material.
    
Args:
    frequency (float): Frequency in rad/s.
    
Return...

### compute_group_velocity(self, frequency) -> float
**Описание:** Compute group velocity v_group = dω/dk.

Physical Meaning:
    Computes the group velocity of wave packets
    in the material.
    
Args:
    frequency (float): Frequency in rad/s.
    
Returns:
    ...

### compute_absorption_coefficient(self, frequency) -> float
**Описание:** Compute absorption coefficient α(ω).

Physical Meaning:
    Computes the absorption coefficient representing
    energy loss in the material.
    
Args:
    frequency (float): Frequency in rad/s.
    ...

### compute_refractive_index(self, frequency) -> complex
**Описание:** Compute complex refractive index n(ω).

Physical Meaning:
    Computes the complex refractive index n = n' + in''
    representing the material's optical properties.
    
Args:
    frequency (float): ...

### __repr__(self) -> str
**Описание:** String representation of frequency-dependent properties.

## ./bhlff/core/bvp/constants/nonlinear_coefficients.py
Methods: 3

### __init__(self, constants)
**Описание:** Initialize nonlinear coefficients calculator.

Physical Meaning:
    Sets up the nonlinear coefficient calculations
    with access to BVP constants.

Args:
    constants: BVP constants instance.

### compute_admittance_coefficients(self, frequency, amplitude)
**Описание:** Compute nonlinear admittance coefficients using advanced field theory.

Physical Meaning:
    Computes frequency and amplitude dependent coefficients for
    nonlinear admittance using full electromag...

### __repr__(self) -> str
**Описание:** String representation of nonlinear coefficients.

## ./bhlff/core/bvp/constants/renormalized_coefficients.py
Methods: 3

### __init__(self, constants)
**Описание:** Initialize renormalized coefficients calculator.

Physical Meaning:
    Sets up the renormalized coefficient calculations
    with access to BVP constants.

Args:
    constants: BVP constants instance...

### compute_renormalized_coefficients(self, amplitude, gradient_magnitude_squared)
**Описание:** Compute renormalized coefficients using advanced field theory.

Physical Meaning:
    Computes amplitude and gradient dependent coefficients
    using full quantum field theory with renormalization gr...

### __repr__(self) -> str
**Описание:** String representation of renormalized coefficients.

## ./bhlff/core/bvp/core_region_analyzer.py
Methods: 6

### __init__(self, domain, constants)
**Описание:** Initialize core region analyzer.

Args:
    domain (Domain): Computational domain for analysis.
    constants (BVPConstants): BVP physical constants.

### identify_core_region(self, envelope)
**Описание:** Identify the core region of the envelope.

Physical Meaning:
    Finds the central region where envelope amplitude is highest
    and defines core boundaries based on amplitude decay.

Args:
    envel...

### _find_center_of_mass(self, amplitude)
**Описание:** Find center of mass of the amplitude distribution.

Physical Meaning:
    Computes center of mass as weighted average of coordinates
    with amplitude as weight.

Args:
    amplitude (np.ndarray): En...

### _compute_core_radius(self, amplitude, center) -> float
**Описание:** Compute effective core radius.

Physical Meaning:
    Finds radius where amplitude drops to 1/e of maximum,
    defining effective core boundary.

Args:
    amplitude (np.ndarray): Envelope amplitude....

### _compute_distances_from_center(self, amplitude, center)
**Описание:** Compute distances from center for each point.

Physical Meaning:
    Calculates Euclidean distance from center for each
    point in the domain.

Args:
    amplitude (np.ndarray): Envelope amplitude.
...

### _create_core_mask(self, amplitude, center, radius)
**Описание:** Create mask for core region.

Physical Meaning:
    Creates binary mask identifying points within
    the core region boundary.

Args:
    amplitude (np.ndarray): Envelope amplitude.
    center (List[...

## ./bhlff/core/bvp/core_renormalization_analyzer.py
Methods: 7

### __init__(self, domain, constants)
**Описание:** Initialize core renormalization analyzer.

Args:
    domain (Domain): Computational domain for analysis.
    constants (BVPConstants): BVP physical constants.

### compute_renormalized_coefficients(self, envelope, core_region)
**Описание:** Compute renormalized coefficients c_i^eff(|A|,|∇A|).

Physical Meaning:
    Calculates effective coefficients that depend on envelope
    amplitude and gradient, representing BVP renormalization.

Mat...

### analyze_core_energy_minimization(self, envelope, core_region)
**Описание:** Analyze core energy minimization.

Physical Meaning:
    Computes energy components and checks if core is at
    energy minimum.

Args:
    envelope (np.ndarray): BVP envelope.
    core_region (Dict[s...

### compute_boundary_conditions(self, envelope, core_region)
**Описание:** Compute boundary pressure/stiffness conditions.

Physical Meaning:
    Calculates boundary pressure and stiffness from amplitude
    gradients and second derivatives at core boundary.

Args:
    envel...

### _compute_boundary_pressure(self, amplitude, core_mask) -> float
**Описание:** Compute boundary pressure from amplitude gradients.

Physical Meaning:
    Calculates pressure at core boundary from amplitude
    gradient magnitude.

Args:
    amplitude (np.ndarray): Envelope ampli...

### _compute_boundary_stiffness(self, amplitude, core_mask) -> float
**Описание:** Compute boundary stiffness from second derivatives.

Physical Meaning:
    Calculates stiffness at core boundary from second
    derivative of amplitude.

Args:
    amplitude (np.ndarray): Envelope am...

### _find_boundary_points(self, core_mask)
**Описание:** Find boundary points of core region.

Physical Meaning:
    Identifies points at the boundary of the core region
    for boundary condition analysis.

Args:
    core_mask (np.ndarray): Core region mas...

## ./bhlff/core/bvp/core_renormalization_postulate.py
Methods: 3

### __init__(self, domain, constants)
**Описание:** Initialize core renormalization postulate.

Physical Meaning:
    Sets up the postulate for analyzing core energy minimization
    and coefficient renormalization.

Args:
    domain (Domain): Computat...

### apply(self, envelope)
**Описание:** Apply core renormalization postulate.

Physical Meaning:
    Verifies that the core represents a minimum of
    ω₀-averaged energy with renormalized coefficients
    and proper boundary conditions.

M...

### _validate_core_renormalization(self, renormalized_coefficients, energy_analysis) -> bool
**Описание:** Validate core renormalization postulate.

Physical Meaning:
    Checks that the core exhibits proper renormalization
    of coefficients and energy minimization.

Args:
    renormalized_coefficients (...

## ./bhlff/core/bvp/envelope_equation/advanced/solver_adaptive.py
Methods: 9

### __init__(self, domain, config)
**Описание:** Initialize adaptive solver.

### solve_adaptive(self, source)
**Описание:** Solve using adaptive methods.

Physical Meaning:
    Solves the envelope equation using adaptive step size control
    and preconditioning for improved convergence.

Args:
    source (np.ndarray): Sou...

### initialize_solution(self, source)
**Описание:** Initialize solution for adaptive solving.

### compute_residual(self, solution, source)
**Описание:** Compute residual for adaptive solving.

### compute_jacobian(self, solution) -> csc_matrix
**Описание:** Compute Jacobian for adaptive solving.

### solve_linear_system(self, jacobian, residual)
**Описание:** Solve linear system for adaptive solving.

### compute_step_size(self, solution, update, residual) -> float
**Описание:** Compute adaptive step size.

### smooth_field(self, field)
**Описание:** Apply adaptive smoothing to field.

### scale_field(self, field)
**Описание:** Apply adaptive scaling to field.

## ./bhlff/core/bvp/envelope_equation/advanced/solver_advanced_core.py
Methods: 19

### __init__(self, domain, config)
**Описание:** Initialize advanced 7D envelope solver core.

Args:
    domain (Domain7D): 7D computational domain.
    config (Dict[str, Any]): Solver configuration parameters.

### solve_envelope_adaptive(self, source)
**Описание:** Solve envelope equation using adaptive methods.

Physical Meaning:
    Solves the 7D envelope equation using adaptive step size control
    and preconditioning for improved convergence and stability.
...

### solve_envelope_optimized(self, source)
**Описание:** Solve envelope equation using optimization techniques.

Physical Meaning:
    Solves the 7D envelope equation using optimization techniques
    for improved efficiency and convergence.

Mathematical F...

### _initialize_solution_adaptive(self, source)
**Описание:** Initialize solution for adaptive solving.

Physical Meaning:
    Initializes the solution field for adaptive solving methods
    using intelligent initial guess based on source characteristics.

Args:...

### _initialize_solution_optimized(self, source)
**Описание:** Initialize solution for optimized solving.

Physical Meaning:
    Initializes the solution field for optimized solving methods
    using efficient initialization techniques.

Args:
    source (np.ndar...

### _compute_residual_advanced(self, solution, source)
**Описание:** Compute residual for advanced solving.

Physical Meaning:
    Computes the residual of the envelope equation for advanced
    solving methods with enhanced accuracy and efficiency.

Args:
    solution...

### _compute_residual_optimized(self, solution, source)
**Описание:** Compute residual for optimized solving.

Physical Meaning:
    Computes the residual of the envelope equation for optimized
    solving methods with computational efficiency.

Args:
    solution (np.n...

### _compute_jacobian_advanced(self, solution) -> csc_matrix
**Описание:** Compute Jacobian for advanced solving.

Physical Meaning:
    Computes the Jacobian matrix for advanced solving methods
    with enhanced accuracy and preconditioning support.

Args:
    solution (np....

### _compute_jacobian_optimized(self, solution) -> csc_matrix
**Описание:** Compute Jacobian for optimized solving.

Physical Meaning:
    Computes the Jacobian matrix for optimized solving methods
    with computational efficiency and memory optimization.

Args:
    solution...

### _solve_linear_system_advanced(self, jacobian, residual)
**Описание:** Solve linear system for advanced solving.

Physical Meaning:
    Solves the linear system for advanced solving methods
    with preconditioning and adaptive techniques.

Args:
    jacobian (csc_matrix...

### _solve_linear_system_optimized(self, jacobian, residual)
**Описание:** Solve linear system for optimized solving.

Physical Meaning:
    Solves the linear system for optimized solving methods
    with computational efficiency and memory optimization.

Args:
    jacobian ...

### _apply_preconditioning(self, jacobian, residual)
**Описание:** Apply preconditioning to linear system.

Physical Meaning:
    Applies preconditioning to improve the conditioning of the
    linear system for better convergence.

Args:
    jacobian (csc_matrix): Ja...

### _compute_preconditioner(self, jacobian) -> csc_matrix
**Описание:** Compute preconditioner matrix.

Physical Meaning:
    Computes a preconditioner matrix to improve the conditioning
    of the linear system.

Args:
    jacobian (csc_matrix): Jacobian matrix.

Returns...

### _compute_adaptive_step_size(self, solution, update, residual) -> float
**Описание:** Compute adaptive step size.

Physical Meaning:
    Computes an adaptive step size based on the current solution,
    update vector, and residual for improved convergence.

Args:
    solution (np.ndarr...

### _compute_optimized_step_size(self, solution, update, residual) -> float
**Описание:** Compute optimized step size.

Physical Meaning:
    Computes an optimized step size for efficient convergence
    using optimization techniques.

Args:
    solution (np.ndarray): Current solution fiel...

### _adaptive_smooth_field(self, field)
**Описание:** Apply adaptive smoothing to field.

Physical Meaning:
    Applies adaptive smoothing to the field for improved
    numerical stability and convergence.

Args:
    field (np.ndarray): Field to smooth.
...

### _adaptive_scale_field(self, field)
**Описание:** Apply adaptive scaling to field.

Physical Meaning:
    Applies adaptive scaling to the field for improved
    numerical conditioning and convergence.

Args:
    field (np.ndarray): Field to scale.

R...

### _optimized_smooth_field(self, field)
**Описание:** Apply optimized smoothing to field.

Physical Meaning:
    Applies optimized smoothing to the field for improved
    efficiency and numerical stability.

Args:
    field (np.ndarray): Field to smooth....

### _optimized_scale_field(self, field)
**Описание:** Apply optimized scaling to field.

Physical Meaning:
    Applies optimized scaling to the field for improved
    efficiency and numerical conditioning.

Args:
    field (np.ndarray): Field to scale.

...

## ./bhlff/core/bvp/envelope_equation/advanced/solver_optimized.py
Methods: 9

### __init__(self, domain, config)
**Описание:** Initialize optimized solver.

### solve_optimized(self, source)
**Описание:** Solve using optimization techniques.

Physical Meaning:
    Solves the envelope equation using optimization techniques
    for improved efficiency and convergence.

Args:
    source (np.ndarray): Sour...

### initialize_solution(self, source)
**Описание:** Initialize solution for optimized solving.

### compute_residual(self, solution, source)
**Описание:** Compute residual for optimized solving.

### compute_jacobian(self, solution) -> csc_matrix
**Описание:** Compute Jacobian for optimized solving.

### solve_linear_system(self, jacobian, residual)
**Описание:** Solve linear system for optimized solving.

### compute_step_size(self, solution, update, residual) -> float
**Описание:** Compute optimized step size.

### smooth_field(self, field)
**Описание:** Apply optimized smoothing to field.

### scale_field(self, field)
**Описание:** Apply optimized scaling to field.

## ./bhlff/core/bvp/envelope_equation/advanced/solver_preconditioning.py
Methods: 5

### __init__(self, domain, config)
**Описание:** Initialize preconditioning.

### apply_preconditioning(self, jacobian, residual)
**Описание:** Apply preconditioning to linear system.

Physical Meaning:
    Applies preconditioning to improve the conditioning of the
    linear system for better convergence.

Args:
    jacobian (csc_matrix): Ja...

### compute_preconditioner(self, jacobian) -> csc_matrix
**Описание:** Compute preconditioner matrix.

Physical Meaning:
    Computes a preconditioner matrix to improve the conditioning
    of the linear system.

Args:
    jacobian (csc_matrix): Jacobian matrix.

Returns...

### _compute_jacobi_preconditioner(self, jacobian) -> csc_matrix
**Описание:** Compute Jacobi preconditioner.

### _compute_ilu_preconditioner(self, jacobian) -> csc_matrix
**Описание:** Compute ILU preconditioner (simplified).

## ./bhlff/core/bvp/envelope_equation/bvp_envelope_equation_7d_facade.py
Methods: 7

### __init__(self, domain_7d, config)
**Описание:** Initialize 7D envelope equation solver.

Physical Meaning:
    Sets up the envelope equation solver with the computational
    domain and configuration parameters, initializing all
    necessary compo...

### solve_envelope(self, source_7d, initial_guess)
**Описание:** Solve 7D envelope equation.

Physical Meaning:
    Solves the full 7D envelope equation for the BVP field
    in space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ using iterative
    Newton-Raphson method for nonlinear...

### get_parameters(self)
**Описание:** Get envelope equation parameters.

Physical Meaning:
    Returns the current values of all parameters for
    monitoring and analysis purposes.

Returns:
    Dict[str, float]: Dictionary containing al...

### analyze_solution_quality(self, envelope, source)
**Описание:** Analyze quality of the solution.

Physical Meaning:
    Analyzes the quality of the envelope solution by computing
    residual components and convergence metrics.

Args:
    envelope (np.ndarray): Co...

### __repr__(self) -> str
**Описание:** String representation of envelope equation solver.

Returns:
    str: String representation showing domain and parameters.

### residual_func(envelope, source)
**Описание:** Нет докстринга

### jacobian_func(envelope)
**Описание:** Нет докстринга

## ./bhlff/core/bvp/envelope_equation/derivative_operators/phase_operators.py
Methods: 7

### __init__(self, domain_7d)
**Описание:** Initialize phase derivative operators.

Physical Meaning:
    Sets up the phase derivative operators with the 7D
    computational domain, preparing for the computation of
    phase gradients and dive...

### setup_operators(self)
**Описание:** Setup all phase derivative operators.

Physical Meaning:
    Initializes all phase derivative operators including
    gradients and divergences for the φ₁, φ₂, and φ₃ directions
    with periodic boun...

### _setup_phase_derivatives(self, phase_shape, dphi_1, dphi_2, dphi_3)
**Описание:** Setup phase derivative operators.

Physical Meaning:
    Creates periodic derivative operators for phase coordinates
    with periodic boundary conditions appropriate for the
    toroidal phase space....

### _create_periodic_gradient_operator(self, N, dx, axis) -> csc_matrix
**Описание:** Create periodic gradient operator for phase coordinates.

Physical Meaning:
    Creates a periodic gradient operator using central differences
    with periodic boundary conditions for the toroidal ph...

### _create_periodic_divergence_operator(self, N, dx, axis) -> csc_matrix
**Описание:** Create periodic divergence operator for phase coordinates.

Physical Meaning:
    Creates a periodic divergence operator as the negative of
    the periodic gradient operator.

Args:
    N: Grid size ...

### apply_gradient(self, field, axis)
**Описание:** Apply phase gradient operator.

Physical Meaning:
    Applies the phase gradient operator to compute the gradient
    of the field along the specified phase axis with periodic
    boundary conditions....

### apply_divergence(self, field, axis)
**Описание:** Apply phase divergence operator.

Physical Meaning:
    Applies the phase divergence operator to compute the divergence
    of the field along the specified phase axis with periodic
    boundary condi...

## ./bhlff/core/bvp/envelope_equation/derivative_operators/spatial_operators.py
Methods: 7

### __init__(self, domain_7d)
**Описание:** Initialize spatial derivative operators.

Physical Meaning:
    Sets up the spatial derivative operators with the 7D
    computational domain, preparing for the computation of
    spatial gradients an...

### setup_operators(self)
**Описание:** Setup all spatial derivative operators.

Physical Meaning:
    Initializes all spatial derivative operators including
    gradients and divergences for the x, y, and z directions
    with appropriate ...

### _setup_spatial_derivatives(self, spatial_shape, dx, dy, dz)
**Описание:** Setup spatial derivative operators.

Physical Meaning:
    Creates finite difference operators for spatial derivatives
    in the x, y, and z directions with appropriate boundary conditions.

Args:
  ...

### _create_gradient_operator(self, N, dx, axis) -> csc_matrix
**Описание:** Create gradient operator for given axis.

Physical Meaning:
    Creates a finite difference gradient operator using central
    differences with appropriate boundary conditions.

Args:
    N: Grid siz...

### _create_divergence_operator(self, N, dx, axis) -> csc_matrix
**Описание:** Create divergence operator for given axis.

Physical Meaning:
    Creates a divergence operator as the negative of the gradient
    operator for conservative form of the equations.

Args:
    N: Grid ...

### apply_gradient(self, field, axis)
**Описание:** Apply spatial gradient operator.

Physical Meaning:
    Applies the spatial gradient operator to compute the gradient
    of the field along the specified spatial axis.

Args:
    field: Field to diff...

### apply_divergence(self, field, axis)
**Описание:** Apply spatial divergence operator.

Physical Meaning:
    Applies the spatial divergence operator to compute the divergence
    of the field along the specified spatial axis.

Args:
    field: Field t...

## ./bhlff/core/bvp/envelope_equation/derivative_operators/temporal_operators.py
Methods: 7

### __init__(self, domain_7d)
**Описание:** Initialize temporal derivative operators.

Physical Meaning:
    Sets up the temporal derivative operators with the 7D
    computational domain, preparing for the computation of
    temporal derivativ...

### setup_operators(self)
**Описание:** Setup temporal derivative operator.

Physical Meaning:
    Initializes the temporal derivative operator using backward
    differences for time evolution in the envelope equation.

### _setup_temporal_derivative(self)
**Описание:** Setup temporal derivative operator.

Physical Meaning:
    Creates the temporal derivative operator using backward
    differences for time evolution in the envelope equation.

### _create_temporal_derivative_operator(self, N_t, dt) -> csc_matrix
**Описание:** Create temporal derivative operator.

Physical Meaning:
    Creates a temporal derivative operator using backward
    differences for time evolution.

Args:
    N_t: Number of time steps.
    dt: Time...

### apply_derivative(self, field)
**Описание:** Apply temporal derivative operator.

Physical Meaning:
    Applies the temporal derivative operator to compute the
    time derivative of the field using backward differences.

Args:
    field: Field ...

### get_time_step(self) -> float
**Описание:** Get time step size.

Physical Meaning:
    Returns the time step size used in the temporal derivative
    operator for monitoring and analysis purposes.

Returns:
    float: Time step size dt.

### get_time_points(self) -> int
**Описание:** Get number of time points.

Physical Meaning:
    Returns the number of time points in the temporal grid
    for monitoring and analysis purposes.

Returns:
    int: Number of time points N_t.

## ./bhlff/core/bvp/envelope_equation/derivative_operators_facade.py
Methods: 10

### __init__(self, domain_7d)
**Описание:** Initialize derivative operators facade.

Physical Meaning:
    Sets up the derivative operators facade with the 7D computational
    domain, initializing all component operators for spatial, phase,
  ...

### setup_operators(self)
**Описание:** Setup all derivative operators for 7D space-time.

Physical Meaning:
    Initializes all derivative operators including spatial,
    phase, and temporal operators with appropriate boundary
    conditi...

### apply_spatial_gradient(self, field, axis)
**Описание:** Apply spatial gradient operator.

Physical Meaning:
    Applies the spatial gradient operator to compute the gradient
    of the field along the specified spatial axis.

Args:
    field: Field to diff...

### apply_spatial_divergence(self, field, axis)
**Описание:** Apply spatial divergence operator.

Physical Meaning:
    Applies the spatial divergence operator to compute the divergence
    of the field along the specified spatial axis.

Args:
    field: Field t...

### apply_phase_gradient(self, field, axis)
**Описание:** Apply phase gradient operator.

Physical Meaning:
    Applies the phase gradient operator to compute the gradient
    of the field along the specified phase axis with periodic
    boundary conditions....

### apply_phase_divergence(self, field, axis)
**Описание:** Apply phase divergence operator.

Physical Meaning:
    Applies the phase divergence operator to compute the divergence
    of the field along the specified phase axis with periodic
    boundary condi...

### apply_temporal_derivative(self, field)
**Описание:** Apply temporal derivative operator.

Physical Meaning:
    Applies the temporal derivative operator to compute the
    time derivative of the field using backward differences.

Args:
    field: Field ...

### spatial(self) -> SpatialOperators
**Описание:** Get spatial operators component.
**Декораторы:** property

### phase(self) -> PhaseOperators
**Описание:** Get phase operators component.
**Декораторы:** property

### temporal(self) -> TemporalOperators
**Описание:** Get temporal operators component.
**Декораторы:** property

## ./bhlff/core/bvp/envelope_equation/nonlinear_terms.py
Methods: 8

### __init__(self, domain_7d, config)
**Описание:** Initialize nonlinear terms.

Physical Meaning:
    Sets up the nonlinear terms with the computational domain
    and configuration parameters, including the nonlinear
    coefficients and quench param...

### setup_terms(self)
**Описание:** Setup nonlinear stiffness and susceptibility terms.

Physical Meaning:
    Initializes the nonlinear functions for stiffness and
    susceptibility based on the configuration parameters.
    These fun...

### compute_stiffness(self, amplitude)
**Описание:** Compute nonlinear stiffness κ(|a|).

Physical Meaning:
    Computes the amplitude-dependent stiffness coefficient
    κ(|a|) = κ₀ + κ₂|a|², representing the nonlinear response
    of the medium to the...

### compute_susceptibility(self, amplitude)
**Описание:** Compute effective susceptibility χ(|a|).

Physical Meaning:
    Computes the amplitude-dependent susceptibility
    χ(|a|) = χ' + iχ''(|a|), representing the complex
    response of the medium includi...

### compute_stiffness_derivative(self, amplitude)
**Описание:** Compute derivative of stiffness with respect to amplitude.

Physical Meaning:
    Computes dκ/d|a| = 2κ₂|a|, needed for the Jacobian
    matrix in Newton-Raphson iterations.

Mathematical Foundation:
...

### compute_susceptibility_derivative(self, amplitude)
**Описание:** Compute derivative of susceptibility with respect to amplitude.

Physical Meaning:
    Computes dχ/d|a| = 2iχ''₀|a|, needed for the Jacobian
    matrix in Newton-Raphson iterations.

Mathematical Foun...

### get_parameters(self)
**Описание:** Get nonlinear term parameters.

Physical Meaning:
    Returns the current values of all nonlinear parameters
    for monitoring and analysis purposes.

Returns:
    Dict[str, float]: Dictionary contai...

### update_parameters(self, new_params)
**Описание:** Update nonlinear term parameters.

Physical Meaning:
    Updates the nonlinear parameters and reinitializes
    the nonlinear functions with the new values.

Args:
    new_params (Dict[str, float]): N...

## ./bhlff/core/bvp/envelope_equation/solver_core_basic.py
Methods: 15

### __init__(self, domain, config)
**Описание:** Initialize 7D envelope solver core.

Physical Meaning:
    Sets up the solver core with the 7D computational domain
    and configuration parameters for solving the envelope equation.

Args:
    domai...

### solve_envelope(self, source)
**Описание:** Solve the 7D envelope equation.

Physical Meaning:
    Solves the 7D envelope equation for the given source term,
    representing the numerical solution of field evolution in
    7D space-time.

Math...

### _initialize_solution(self, source)
**Описание:** Initialize solution field.

Physical Meaning:
    Initializes the solution field with an appropriate initial guess,
    typically based on the source term or previous solutions.

Args:
    source (np....

### _compute_residual(self, solution, source)
**Описание:** Compute residual of the envelope equation.

Physical Meaning:
    Computes the residual R = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s,
    which measures how well the current solution satisfies
    the envelope e...

### _compute_jacobian(self, solution) -> csc_matrix
**Описание:** Compute Jacobian matrix.

Physical Meaning:
    Computes the Jacobian matrix J = ∂R/∂a, which represents
    the linearization of the residual around the current solution.

Mathematical Foundation:
  ...

### _solve_linear_system(self, jacobian, residual)
**Описание:** Solve linear system J·δa = -R.

Physical Meaning:
    Solves the linearized system to find the update δa
    for the Newton-Raphson iteration.

Args:
    jacobian (csc_matrix): Jacobian matrix.
    re...

### _smooth_field(self, field)
**Описание:** Apply smoothing to field to avoid singularities.

Args:
    field (np.ndarray): Field to smooth.

Returns:
    np.ndarray: Smoothed field.

### _compute_nonlinear_stiffness(self, solution)
**Описание:** Compute nonlinear stiffness κ(|a|).

Physical Meaning:
    Computes the nonlinear stiffness coefficient that depends
    on the field amplitude, representing the field's response
    to spatial variat...

### _compute_effective_susceptibility(self, solution)
**Описание:** Compute effective susceptibility χ(|a|).

Physical Meaning:
    Computes the effective susceptibility that depends on
    the field amplitude, representing the field's response
    to external excitat...

### _compute_divergence_term(self, solution, stiffness)
**Описание:** Compute divergence term ∇·(κ(|a|)∇a).

Physical Meaning:
    Computes the divergence of the stiffness-weighted gradient,
    representing the spatial variation of the field.

Args:
    solution (np.nd...

### _compute_susceptibility_term(self, solution, susceptibility)
**Описание:** Compute susceptibility term k₀²χ(|a|)a.

Physical Meaning:
    Computes the susceptibility-weighted field term,
    representing the field's response to external excitations.

Args:
    solution (np.n...

### _compute_gradient(self, field)
**Описание:** Compute gradient of field.

Args:
    field (np.ndarray): Field to differentiate.

Returns:
    np.ndarray: Gradient field.

### _compute_divergence(self, vector_field)
**Описание:** Compute divergence of vector field.

Args:
    vector_field (np.ndarray): Vector field to differentiate.

Returns:
    np.ndarray: Divergence field.

### _compute_jacobian_row(self, solution, idx)
**Описание:** Compute Jacobian row for a specific point.

Args:
    solution (np.ndarray): Current solution field.
    idx (Tuple[int, ...]): Multi-dimensional index.

Returns:
    Dict[int, float]: Jacobian row en...

### get_convergence_info(self)
**Описание:** Get convergence information.

Returns:
    Dict[str, Any]: Convergence information.

## ./bhlff/core/bvp/envelope_linear_solver.py
Methods: 4

### __init__(self, domain, constants)
**Описание:** Initialize linear envelope solver.

Physical Meaning:
    Sets up the linear solver with the computational domain
    and constants for solving the linearized envelope equation.

Args:
    domain (Dom...

### solve_linearized(self, source)
**Описание:** Solve linearized 7D BVP envelope equation.

Physical Meaning:
    Solves the linearized version of the envelope equation
    ∇·(κ₀∇a) + k₀²χ'a = s(x,φ,t) for initial guess generation.

Mathematical Fo...
**Декораторы:** <ast.Call object at 0x753e6f64f010>

### solve_linearized_with_coefficients(self, envelope, kappa, chi, source)
**Описание:** Solve linearized envelope equation with given coefficients.

Physical Meaning:
    Solves the linearized version of the envelope equation
    for a given nonlinear stiffness and susceptibility.

Mathe...

### _compute_div_kappa_grad(self, envelope, kappa)
**Описание:** Compute divergence of κ∇a using finite differences.

Physical Meaning:
    Computes ∇·(κ∇a) using finite difference methods
    for all 7 dimensions of the envelope field.

Mathematical Foundation:
  ...

## ./bhlff/core/bvp/envelope_nonlinear_coefficients.py
Methods: 6

### __init__(self, constants)
**Описание:** Initialize nonlinear coefficients computer.

Physical Meaning:
    Sets up the coefficients computer with the BVP constants
    to compute nonlinear coefficients based on theoretical
    framework par...

### compute_coefficients(self, envelope)
**Описание:** Compute nonlinear coefficients as functions of field amplitude.

Physical Meaning:
    Computes the nonlinear stiffness κ(|a|) and susceptibility χ(|a|)
    as functions of the local field amplitude, ...
**Декораторы:** <ast.Call object at 0x753e6f76b7d0>

### _compute_chi_real_nonlinear(self, amplitude)
**Описание:** Compute nonlinear real part of susceptibility.

Physical Meaning:
    Computes the nonlinear contribution to the real part of
    susceptibility based on field amplitude, representing
    dispersive n...

### _compute_chi_imag_nonlinear(self, amplitude)
**Описание:** Compute nonlinear imaginary part of susceptibility.

Physical Meaning:
    Computes the nonlinear contribution to the imaginary part of
    susceptibility based on field amplitude, representing
    ab...

### _compute_local_adaptive_scale(self, amplitude)
**Описание:** Compute local adaptive scaling factor.

Physical Meaning:
    Computes adaptive scaling factors based on local field properties
    to ensure proper nonlinear behavior in different regions of
    the ...

### get_base_parameters(self)
**Описание:** Get base parameters for nonlinear coefficients.

Physical Meaning:
    Returns the base parameters used for computing nonlinear
    coefficients, showing the theoretical framework parameters.

Returns...

## ./bhlff/core/bvp/envelope_solver/envelope_solver_core.py
Methods: 4

### __init__(self, domain, config, constants)
**Описание:** Initialize envelope solver core.

Physical Meaning:
    Sets up the core mathematical operations with parameters
    for the nonlinear envelope equation.

Args:
    domain (Domain): Computational doma...

### solve_newton_system(self, jacobian, residual)
**Описание:** Solve Newton-Raphson system for envelope equation.

Physical Meaning:
    Solves the linearized Newton-Raphson system J·δa = -R
    for the envelope equation update δa.

Mathematical Foundation:
    S...

### compute_gradient(self, envelope, source)
**Описание:** Compute gradient for fallback gradient descent.

Physical Meaning:
    Computes the gradient of the residual norm for use
    in gradient descent when Newton method fails.

Args:
    envelope (np.ndar...

### __repr__(self) -> str
**Описание:** String representation of envelope solver core.

## ./bhlff/core/bvp/envelope_solver/gradient_computer.py
Methods: 3

### __init__(self, domain, constants)
**Описание:** Initialize gradient computer.

Physical Meaning:
    Sets up the gradient computation with parameters
    for the nonlinear envelope equation.

Args:
    domain (Domain): Computational domain.
    con...

### compute_gradient(self, envelope, source)
**Описание:** Compute gradient for fallback gradient descent.

Physical Meaning:
    Computes the gradient of the residual norm for use
    in gradient descent when Newton method fails.

Mathematical Foundation:
  ...

### __repr__(self) -> str
**Описание:** String representation of gradient computer.

## ./bhlff/core/bvp/envelope_solver/newton_solver.py
Methods: 3

### __init__(self, domain, constants)
**Описание:** Initialize Newton solver.

Physical Meaning:
    Sets up the Newton-Raphson solver with parameters
    for the nonlinear envelope equation.

Args:
    domain (Domain): Computational domain.
    consta...

### solve_newton_system(self, jacobian, residual)
**Описание:** Solve Newton system J * δa = -r.

Physical Meaning:
    Solves the linear system for the Newton update step
    using advanced numerical methods.

Mathematical Foundation:
    Solves J * δa = -r where...

### __repr__(self) -> str
**Описание:** String representation of Newton solver.

## ./bhlff/core/bvp/envelope_solver_line_search.py
Methods: 4

### __init__(self, constants)
**Описание:** Initialize line search algorithms.

Args:
    constants (BVPConstants, optional): BVP constants instance.

### perform_line_search(self, envelope, delta_envelope, residual, source, initial_step, residual_func) -> float
**Описание:** Perform line search for optimal step size.

Physical Meaning:
    Finds the optimal step size along the Newton direction
    to minimize the residual norm.

Mathematical Foundation:
    Minimizes ||r(...

### perform_wolfe_line_search(self, envelope, delta_envelope, residual, source, initial_step, residual_func) -> float
**Описание:** Perform Wolfe line search for optimal step size.

Physical Meaning:
    Finds the optimal step size using Wolfe conditions for
    more robust convergence.

Mathematical Foundation:
    Uses both Armi...

### perform_adaptive_line_search(self, envelope, delta_envelope, residual, source, initial_step, residual_func) -> float
**Описание:** Perform adaptive line search with dynamic parameters.

Physical Meaning:
    Adapts line search parameters based on convergence history
    for optimal performance.

Mathematical Foundation:
    Dynam...

## ./bhlff/core/bvp/impedance_core.py
Methods: 7

### __init__(self, domain, config, constants)
**Описание:** Initialize impedance core.

Physical Meaning:
    Sets up the core mathematical operations with parameters
    for impedance calculation.

Args:
    domain (Domain): Computational domain.
    config (...

### _setup_parameters(self, config)
**Описание:** Setup impedance calculation parameters.

### compute_admittance_from_envelope(self, envelope, frequencies)
**Описание:** Compute admittance from envelope.

Physical Meaning:
    Computes the frequency-dependent admittance Y(ω)
    from the BVP envelope using boundary analysis.

Args:
    envelope (np.ndarray): BVP envel...

### compute_reflection_coefficient(self, admittance)
**Описание:** Compute reflection coefficient from admittance.

Physical Meaning:
    Computes the reflection coefficient R(ω) from
    the admittance Y(ω).

Args:
    admittance (np.ndarray): Admittance Y(ω).

Retu...

### compute_transmission_coefficient(self, admittance)
**Описание:** Compute transmission coefficient from admittance.

Physical Meaning:
    Computes the transmission coefficient T(ω) from
    the admittance Y(ω).

Args:
    admittance (np.ndarray): Admittance Y(ω).

...

### get_parameters(self)
**Описание:** Get impedance core parameters.

Physical Meaning:
    Returns the current parameters for impedance calculation.

Returns:
    Dict[str, Any]: Impedance core parameters.

### __repr__(self) -> str
**Описание:** String representation of impedance core.

## ./bhlff/core/bvp/interface/core_interface.py
Methods: 8

### __init__(self, bvp_core)
**Описание:** Initialize core interface.

Physical Meaning:
    Sets up the interface with the BVP core module for
    core calculations.

Args:
    bvp_core (BVPCore): BVP core module instance.

### interface_with_core(self, envelope)
**Описание:** Interface BVP with core.

Physical Meaning:
    Provides the necessary data for core calculations:
    - Renormalized coefficients c_i^eff(A,∇A) from BVP averaging
    - Boundary conditions (pressure/...

### compute_field_gradient(self, field) -> list
**Описание:** Compute field gradient in all 7 dimensions.

Physical Meaning:
    Computes the gradient of the field in all 7 dimensions
    (3 spatial + 3 phase + 1 temporal) for 7D space-time analysis.

Mathematic...

### _compute_renormalized_coefficients(self, envelope)
**Описание:** Compute renormalized coefficients c_i^eff(A,∇A).

Physical Meaning:
    Computes the renormalized coefficients that result
    from BVP averaging over the high-frequency carrier.

Returns:
    Dict[st...

### _compute_boundary_pressure(self, envelope)
**Описание:** Compute boundary pressure P_boundary.

Physical Meaning:
    Computes the boundary pressure that results from
    the BVP field at the boundaries.

Returns:
    np.ndarray: Boundary pressure.

### _compute_core_stiffness(self, envelope)
**Описание:** Compute core stiffness K_core.

Physical Meaning:
    Computes the core stiffness that results from
    the BVP field interaction with the core.

Returns:
    np.ndarray: Core stiffness.

### _compute_core_energy_density(self, envelope)
**Описание:** Compute core energy density.

Physical Meaning:
    Computes the energy density in the core region
    resulting from the BVP field.

Returns:
    np.ndarray: Core energy density.

### _compute_effective_parameters(self, envelope)
**Описание:** Compute effective core parameters.

Physical Meaning:
    Computes the effective parameters for core evolution
    that result from BVP averaging.

Returns:
    Dict[str, float]: Effective parameters.

## ./bhlff/core/bvp/interface/interface_facade.py
Methods: 7

### __init__(self, bvp_core)
**Описание:** Initialize BVP interface facade.

Physical Meaning:
    Sets up the interface facade with the BVP core module,
    establishing connections to all system components.

Args:
    bvp_core (BVPCore): BVP...

### interface_with_tail(self, envelope)
**Описание:** Interface BVP with tail resonators.

Physical Meaning:
    Provides the necessary data for tail resonator calculations
    through the tail interface component.

Args:
    envelope (np.ndarray): 7D en...

### interface_with_transition_zone(self, envelope)
**Описание:** Interface BVP with transition zone.

Physical Meaning:
    Provides the necessary data for transition zone calculations
    through the transition interface component.

Args:
    envelope (np.ndarray)...

### interface_with_core(self, envelope)
**Описание:** Interface BVP with core.

Physical Meaning:
    Provides the necessary data for core calculations
    through the core interface component.

Args:
    envelope (np.ndarray): 7D envelope field.

Return...

### get_tail_interface(self) -> TailInterface
**Описание:** Get the tail interface component.

Physical Meaning:
    Returns the tail interface component for direct access
    to tail-specific operations.

Returns:
    TailInterface: Tail interface component.

### get_transition_interface(self) -> TransitionInterface
**Описание:** Get the transition interface component.

Physical Meaning:
    Returns the transition interface component for direct access
    to transition zone-specific operations.

Returns:
    TransitionInterfac...

### get_core_interface(self) -> CoreInterface
**Описание:** Get the core interface component.

Physical Meaning:
    Returns the core interface component for direct access
    to core-specific operations.

Returns:
    CoreInterface: Core interface component.

## ./bhlff/core/bvp/interface/tail_interface.py
Methods: 3

### __init__(self, bvp_core)
**Описание:** Initialize tail interface.

Physical Meaning:
    Sets up the interface with the BVP core module for
    tail resonator calculations.

Args:
    bvp_core (BVPCore): BVP core module instance.

### interface_with_tail(self, envelope)
**Описание:** Interface BVP with tail resonators.

Physical Meaning:
    Provides the necessary data for tail resonator calculations:
    - Admittance Y(ω) for cascade resonator calculations
    - Resonance peaks {...

### _compute_spectral_data(self, envelope)
**Описание:** Compute spectral data S(ω) from BVP envelope.

Physical Meaning:
    Computes the spectral data S(ω) that represents the
    frequency content of the BVP envelope for cascade
    resonator calculation...

## ./bhlff/core/bvp/interface/transition_interface.py
Methods: 5

### __init__(self, bvp_core)
**Описание:** Initialize transition zone interface.

Physical Meaning:
    Sets up the interface with the BVP core module for
    transition zone calculations.

Args:
    bvp_core (BVPCore): BVP core module instanc...

### interface_with_transition_zone(self, envelope)
**Описание:** Interface BVP with transition zone.

Physical Meaning:
    Provides the necessary data for transition zone calculations:
    - Nonlinear admittance Y_tr(ω,|A|) for transition zone analysis
    - EM/we...

### _compute_nonlinear_admittance(self, envelope)
**Описание:** Compute nonlinear admittance Y_tr(ω,|A|).

Physical Meaning:
    Computes the nonlinear admittance that depends on both
    frequency and envelope amplitude, representing the
    transition zone respo...

### _compute_em_current_sources(self, envelope)
**Описание:** Compute EM current sources J_EM(ω;A).

Physical Meaning:
    Computes the electromagnetic current sources generated
    from the BVP envelope, representing the coupling
    between BVP and EM fields.
...

### _compute_loss_map(self, envelope)
**Описание:** Compute loss map χ''(|A|).

Physical Meaning:
    Computes the loss map that shows how losses depend
    on the envelope amplitude, including quench effects.

Returns:
    np.ndarray: Loss map χ''(|A|...

## ./bhlff/core/bvp/level_a_interface.py
Methods: 4

### __init__(self, bvp_core)
**Описание:** Нет докстринга

### process_bvp_data(self, envelope)
**Описание:** Process BVP data for Level A operations.

Physical Meaning:
    Provides BVP envelope data for validation, scaling,
    and nondimensionalization in Level A.

### _compute_scaling_parameters(self, envelope)
**Описание:** Compute scaling parameters from BVP envelope.

### _compute_nondimensionalization(self, envelope)
**Описание:** Compute nondimensionalization factors.

## ./bhlff/core/bvp/level_b_analysis/nodes_analyzer.py
Methods: 2

### check_spherical_nodes(self, envelope)
**Описание:** Check for absence of spherical standing nodes.

Physical Meaning:
    Detects spherical standing wave nodes in the BVP envelope,
    which should be absent in the fundamental field configuration
    a...

### analyze_node_distribution(self, envelope)
**Описание:** Analyze distribution of potential nodes.

Physical Meaning:
    Analyzes the spatial distribution of potential nodes to
    understand their clustering and symmetry properties.

Args:
    envelope (np...

## ./bhlff/core/bvp/level_b_analysis/power_law_analyzer.py
Methods: 3

### __init__(self)
**Описание:** Initialize power law analyzer.

### analyze_power_law_tails(self, envelope)
**Описание:** Analyze power law tails in homogeneous medium.

Physical Meaning:
    Computes the power law decay of BVP envelope amplitude
    in the tail region, which characterizes the field's
    long-range beha...

### compute_radial_profile(self, envelope, n_bins)
**Описание:** Compute radial profile of envelope amplitude.

Physical Meaning:
    Computes the radial average of envelope amplitude for
    analysis of field structure and power law behavior.

Args:
    envelope (...

## ./bhlff/core/bvp/level_b_analysis/topological_charge_analyzer.py
Methods: 2

### compute_topological_charge(self, envelope)
**Описание:** Compute topological charge of defects.

Physical Meaning:
    Calculates the topological charge of defects in the BVP envelope
    using the winding number around closed loops in the field.

Mathemati...

### analyze_phase_structure(self, envelope)
**Описание:** Analyze phase structure of the field.

Physical Meaning:
    Analyzes the phase structure of the BVP envelope to understand
    the topological characteristics and phase coherence.

Args:
    envelope...

## ./bhlff/core/bvp/level_b_analysis/zone_separation_analyzer.py
Methods: 3

### analyze_zone_separation(self, envelope)
**Описание:** Analyze separation of core/transition/tail zones.

Physical Meaning:
    Identifies the three characteristic zones in the BVP envelope:
    core (high amplitude, nonlinear), transition (intermediate),...

### compute_zone_statistics(self, envelope)
**Описание:** Compute detailed statistics for each zone.

Physical Meaning:
    Computes detailed statistical properties of each zone
    to characterize their physical properties.

Args:
    envelope (np.ndarray):...

### _compute_region_statistics(self, amplitude, region_mask)
**Описание:** Compute statistics for a specific region.

Args:
    amplitude (np.ndarray): Field amplitude.
    region_mask (np.ndarray): Boolean mask for the region.

Returns:
    Dict[str, Any]: Statistics for th...

## ./bhlff/core/bvp/level_b_interface_facade.py
Methods: 4

### __init__(self, bvp_core)
**Описание:** Initialize Level B interface.

Physical Meaning:
    Sets up the Level B interface with the BVP core and
    initializes all analysis modules for fundamental
    field properties analysis.

Args:
    ...

### process_bvp_data(self, envelope)
**Описание:** Process BVP data for Level B operations.

Physical Meaning:
    Analyzes fundamental properties of BVP envelope including
    power law tails, absence of spherical nodes, and topological charge.

Math...

### get_detailed_analysis(self, envelope)
**Описание:** Get detailed analysis results for Level B.

Physical Meaning:
    Provides comprehensive analysis of all Level B properties
    including detailed statistics and additional metrics.

Args:
    envelop...

### validate_level_b_properties(self, envelope)
**Описание:** Validate Level B properties against theory.

Physical Meaning:
    Validates that the BVP envelope exhibits the expected
    Level B properties according to the 7D phase field theory.

Args:
    envel...

## ./bhlff/core/bvp/level_c_interface.py
Methods: 6

### __init__(self, bvp_core)
**Описание:** Нет докстринга

### process_bvp_data(self, envelope)
**Описание:** Process BVP data for Level C operations.

Physical Meaning:
    Analyzes boundary effects, resonator structures, quench memory,
    and mode beating in BVP envelope.

### _analyze_boundary_effects(self, envelope)
**Описание:** Analyze boundary effects on BVP envelope.

### _analyze_resonator_structures(self, envelope)
**Описание:** Analyze resonator structures.

### _analyze_quench_memory(self, envelope)
**Описание:** Analyze quench memory effects.

### _analyze_mode_beating(self, envelope)
**Описание:** Analyze mode beating patterns.

## ./bhlff/core/bvp/level_d_interface.py
Methods: 5

### __init__(self, bvp_core)
**Описание:** Initialize Level D interface.

Physical Meaning:
    Sets up the interface for Level D analysis with access to
    BVP core functionality and constants.

Args:
    bvp_core (BVPCore): BVP core instanc...

### process_bvp_data(self, envelope)
**Описание:** Process BVP data for Level D operations.

Physical Meaning:
    Analyzes multimode superposition, field projections,
    and streamlines in BVP envelope to understand the
    complex multimode dynamic...

### _analyze_mode_superposition(self, envelope)
**Описание:** Analyze mode superposition patterns.

Physical Meaning:
    Performs FFT analysis to decompose the BVP envelope into
    its constituent modes and identifies dominant frequency
    components and thei...

### _analyze_field_projections(self, envelope)
**Описание:** Analyze field projections onto different subspaces.

Physical Meaning:
    Projects the BVP envelope onto different subspaces to
    understand the field structure in spatial and phase
    dimensions ...

### _analyze_streamlines(self, envelope)
**Описание:** Analyze streamline patterns in the field.

Physical Meaning:
    Computes field gradients to analyze streamline patterns
    and flow characteristics in the BVP envelope, providing
    insights into t...

## ./bhlff/core/bvp/level_e_interface.py
Methods: 7

### __init__(self, bvp_core)
**Описание:** Initialize Level E interface.

Physical Meaning:
    Sets up the interface for Level E analysis with access to
    BVP core functionality and constants.

Args:
    bvp_core (BVPCore): BVP core instanc...

### process_bvp_data(self, envelope)
**Описание:** Process BVP data for Level E operations.

Physical Meaning:
    Analyzes solitons, defect dynamics, interactions,
    and formation in BVP envelope to understand the
    localized structures and their...

### _analyze_solitons(self, envelope)
**Описание:** Analyze soliton structures.

Physical Meaning:
    Identifies and characterizes soliton-like structures in the
    BVP envelope, which are localized, stable field configurations
    that maintain thei...

### _analyze_defect_dynamics(self, envelope)
**Описание:** Analyze defect dynamics.

Physical Meaning:
    Analyzes the dynamics of topological defects in the BVP envelope,
    including phase singularities and their evolution over time.

Mathematical Foundat...

### _analyze_interactions(self, envelope)
**Описание:** Analyze defect interactions.

Physical Meaning:
    Analyzes the interactions between defects in the BVP envelope,
    including energy exchange and mutual influence.

Mathematical Foundation:
    Int...

### _analyze_formation(self, envelope)
**Описание:** Analyze defect formation mechanisms.

Physical Meaning:
    Analyzes the mechanisms by which defects form in the BVP envelope,
    including nucleation processes and formation probabilities.

Mathemat...

### _compute_soliton_stability(self, envelope, local_maxima) -> float
**Описание:** Compute soliton stability measure.

Physical Meaning:
    Calculates the stability of soliton-like structures based on
    their amplitude profile and phase coherence.

Mathematical Foundation:
    St...

## ./bhlff/core/bvp/level_f_interface.py
Methods: 6

### __init__(self, bvp_core)
**Описание:** Initialize Level F interface.

Physical Meaning:
    Sets up the interface for Level F analysis with access to
    BVP core functionality and constants.

Args:
    bvp_core (BVPCore): BVP core instanc...

### process_bvp_data(self, envelope)
**Описание:** Process BVP data for Level F operations.

Physical Meaning:
    Analyzes multi-particle systems, collective modes,
    phase transitions, and nonlinear effects in BVP envelope
    to understand collec...

### _analyze_multi_particle_systems(self, envelope)
**Описание:** Analyze multi-particle systems.

Physical Meaning:
    Analyzes the collective behavior of multiple particles
    in the BVP envelope, including particle density and
    interaction patterns.

Mathema...

### _analyze_collective_modes(self, envelope)
**Описание:** Analyze collective modes.

Physical Meaning:
    Analyzes collective modes that emerge from the interaction
    of multiple particles in the BVP envelope.

Mathematical Foundation:
    Collective mode...

### _analyze_phase_transitions(self, envelope)
**Описание:** Analyze phase transitions.

Physical Meaning:
    Analyzes phase transitions in the BVP envelope,
    including critical behavior and transition probabilities.

Mathematical Foundation:
    Phase tran...

### _analyze_nonlinear_effects(self, envelope)
**Описание:** Analyze nonlinear effects.

Physical Meaning:
    Analyzes nonlinear effects in the BVP envelope,
    including nonlinear response and saturation behavior.

Mathematical Foundation:
    Nonlinear effe...

## ./bhlff/core/bvp/memory_decorator.py
Methods: 6

### memory_protected(memory_threshold, shape_param, dtype_param)
**Описание:** Decorator for automatic memory protection.

Physical Meaning:
    Automatically checks memory usage before executing
    BVP calculations and prevents out-of-memory errors.

Mathematical Foundation:
 ...

### memory_protected_method(memory_threshold, shape_param, dtype_param)
**Описание:** Decorator for automatic memory protection on class methods.

Physical Meaning:
    Automatically checks memory usage before executing
    BVP calculation methods and prevents out-of-memory errors.

Ar...

### memory_protected_class_method(memory_threshold, shape_param, dtype_param)
**Описание:** Decorator for automatic memory protection on class methods with self access.

Physical Meaning:
    Automatically checks memory usage before executing
    BVP calculation methods, using class-level me...

### memory_protected_function(memory_threshold, shape_param, dtype_param)
**Описание:** Decorator for automatic memory protection on standalone functions.

Physical Meaning:
    Automatically checks memory usage before executing
    BVP calculation functions and prevents out-of-memory er...

### decorator(func) -> Callable
**Описание:** Нет докстринга

### wrapper()
**Описание:** Нет докстринга
**Декораторы:** <ast.Call object at 0x753e6f654110>

## ./bhlff/core/bvp/memory_protection.py
Methods: 5

### __init__(self, memory_threshold)
**Описание:** Initialize memory protector.

Physical Meaning:
    Sets up memory protection with configurable threshold
    to prevent out-of-memory errors during calculations.

Args:
    memory_threshold (float): ...

### check_memory_usage(self, domain_shape, data_type) -> bool
**Описание:** Check if memory usage would exceed threshold.

Physical Meaning:
    Estimates memory requirements for the given domain
    and data type, checking against the memory threshold.

Mathematical Foundati...

### get_memory_info(self) -> dict
**Описание:** Get current memory information.

Physical Meaning:
    Returns current memory usage statistics for monitoring
    and debugging purposes.

Returns:
    dict: Memory information including:
        - to...

### estimate_memory_requirement(self, domain_shape, data_type) -> dict
**Описание:** Estimate memory requirement for given domain and data type.

Physical Meaning:
    Calculates estimated memory requirements for BVP calculations
    with the given domain size and data type.

Mathemat...

### check_and_warn(self, domain_shape, data_type) -> bool
**Описание:** Check memory usage and issue warning if approaching threshold.

Physical Meaning:
    Checks memory usage and issues a warning if it approaches
    the threshold, allowing for graceful handling.

Args...

## ./bhlff/core/bvp/phase_vector/electroweak_coupling.py
Methods: 8

### __init__(self, config)
**Описание:** Initialize electroweak coupling.

Physical Meaning:
    Sets up the coefficients for electroweak currents
    that are generated as functionals of the envelope.

Args:
    config (Dict[str, Any]): Ele...

### _setup_electroweak_coefficients(self)
**Описание:** Setup electroweak coupling coefficients.

Physical Meaning:
    Initializes the coefficients for electroweak currents
    that are generated as functionals of the envelope.

### compute_electroweak_currents(self, envelope, phase_components, domain)
**Описание:** Compute electroweak currents as functionals of the envelope.

Physical Meaning:
    Computes electromagnetic and weak currents that are
    generated as functionals of the BVP envelope through
    the...

### get_electroweak_coefficients(self)
**Описание:** Get electroweak coupling coefficients.

Physical Meaning:
    Returns the current electroweak coupling coefficients
    used for current calculations.

Returns:
    Dict[str, float]: Electroweak coupl...

### set_electroweak_coefficients(self, coefficients)
**Описание:** Set electroweak coupling coefficients.

Physical Meaning:
    Updates the electroweak coupling coefficients
    used for current calculations.

Args:
    coefficients (Dict[str, float]): New coupling ...

### get_weinberg_angle(self) -> float
**Описание:** Get the Weinberg mixing angle.

Physical Meaning:
    Returns the Weinberg mixing angle used for
    electroweak mixing calculations.

Returns:
    float: Weinberg mixing angle.

### set_weinberg_angle(self, angle)
**Описание:** Set the Weinberg mixing angle.

Physical Meaning:
    Updates the Weinberg mixing angle used for
    electroweak mixing calculations.

Args:
    angle (float): New Weinberg mixing angle.

### __repr__(self) -> str
**Описание:** String representation of electroweak coupling.

## ./bhlff/core/bvp/phase_vector/phase_components.py
Methods: 7

### __init__(self, domain, config)
**Описание:** Initialize phase components manager.

Physical Meaning:
    Sets up the three U(1) phase components Θ_a (a=1..3)
    with proper spatial distribution and frequencies.

Args:
    domain (Domain): Compu...

### _setup_phase_components(self)
**Описание:** Setup the three U(1) phase components Θ_a (a=1..3).

Physical Meaning:
    Initializes the three independent U(1) phase components
    that form the U(1)³ structure of the BVP field.

### get_components(self)
**Описание:** Get the three U(1) phase components Θ_a (a=1..3).

Physical Meaning:
    Returns the three independent U(1) phase components
    that form the U(1)³ structure.

Returns:
    List[np.ndarray]: List of ...

### get_total_phase(self, coupling_matrix)
**Описание:** Get the total phase from U(1)³ structure.

Physical Meaning:
    Computes the total phase by combining the three
    U(1) components with proper coupling.

Mathematical Foundation:
    Θ_total = Σ_a Θ...

### update_components(self, envelope)
**Описание:** Update phase components from solved envelope.

Physical Meaning:
    Updates the three U(1) phase components Θ_a (a=1..3)
    from the solved BVP envelope field.

Mathematical Foundation:
    Extracts...

### compute_phase_coherence(self)
**Описание:** Compute phase coherence measure.

Physical Meaning:
    Computes a measure of phase coherence across the
    U(1)³ structure, indicating the degree of
    synchronization between the three phase compo...

### __repr__(self) -> str
**Описание:** String representation of phase components.

## ./bhlff/core/bvp/phase_vector/phase_vector.py
Methods: 12

### __init__(self, domain, config, constants)
**Описание:** Initialize U(1)³ phase vector structure.

Physical Meaning:
    Sets up the three-component phase vector Θ_a (a=1..3)
    with proper U(1)³ structure and weak SU(2) coupling.

Args:
    domain (Domain...

### _setup_su2_coupling(self)
**Описание:** Setup weak hierarchical coupling to SU(2)/core.

Physical Meaning:
    Establishes the weak hierarchical coupling between
    the U(1)³ phase structure and SU(2)/core through
    invariant mixed terms...

### get_phase_components(self)
**Описание:** Get the three U(1) phase components Θ_a (a=1..3).

Physical Meaning:
    Returns the three independent U(1) phase components
    that form the U(1)³ structure.

Returns:
    List[np.ndarray]: List of ...

### get_total_phase(self)
**Описание:** Get the total phase from U(1)³ structure.

Physical Meaning:
    Computes the total phase by combining the three
    U(1) components with proper SU(2) coupling.

Mathematical Foundation:
    Θ_total =...

### compute_electroweak_currents(self, envelope)
**Описание:** Compute electroweak currents as functionals of the envelope.

Physical Meaning:
    Computes electromagnetic and weak currents that are
    generated as functionals of the BVP envelope through
    the...

### compute_phase_coherence(self)
**Описание:** Compute phase coherence measure.

Physical Meaning:
    Computes a measure of phase coherence across the
    U(1)³ structure, indicating the degree of
    synchronization between the three phase compo...

### get_su2_coupling_strength(self) -> float
**Описание:** Get the SU(2) coupling strength.

Physical Meaning:
    Returns the strength of the weak hierarchical
    coupling to SU(2)/core.

Returns:
    float: SU(2) coupling strength.

### set_su2_coupling_strength(self, strength)
**Описание:** Set the SU(2) coupling strength.

Physical Meaning:
    Updates the strength of the weak hierarchical
    coupling to SU(2)/core.

Args:
    strength (float): New SU(2) coupling strength.

### update_phase_components(self, envelope)
**Описание:** Update phase components from solved envelope.

Physical Meaning:
    Updates the three U(1) phase components Θ_a (a=1..3)
    from the solved BVP envelope field.

Mathematical Foundation:
    Extracts...

### get_electroweak_coefficients(self)
**Описание:** Get electroweak coupling coefficients.

Physical Meaning:
    Returns the current electroweak coupling coefficients
    used for current calculations.

Returns:
    Dict[str, float]: Electroweak coupl...

### set_electroweak_coefficients(self, coefficients)
**Описание:** Set electroweak coupling coefficients.

Physical Meaning:
    Updates the electroweak coupling coefficients
    used for current calculations.

Args:
    coefficients (Dict[str, float]): New coupling ...

### __repr__(self) -> str
**Описание:** String representation of phase vector.

## ./bhlff/core/bvp/physical_validation_decorator.py
Methods: 13

### physical_validation_required(domain_shape, parameters)
**Описание:** Decorator for automatic physical validation of BVP methods.

Physical Meaning:
    Automatically validates that BVP method results satisfy
    physical constraints and theoretical bounds according to
...

### validate_physical_constraints(domain_shape, parameters)
**Описание:** Decorator for validating physical constraints only.

Physical Meaning:
    Validates that results satisfy physical constraints
    without checking theoretical bounds.

Args:
    domain_shape (tuple):...

### validate_theoretical_bounds(domain_shape, parameters)
**Описание:** Decorator for validating theoretical bounds only.

Physical Meaning:
    Validates that results are within theoretical bounds
    without checking physical constraints.

Args:
    domain_shape (tuple)...

### validate_energy_conservation(domain_shape, parameters)
**Описание:** Decorator for validating energy conservation only.

Physical Meaning:
    Validates that energy is conserved according to
    the 7D phase field theory framework.

Args:
    domain_shape (tuple): Shap...

### validate_causality(domain_shape, parameters)
**Описание:** Decorator for validating causality constraints only.

Physical Meaning:
    Validates that results satisfy causality constraints
    according to the 7D phase field theory framework.

Args:
    domain...

### validate_7d_structure(domain_shape, parameters)
**Описание:** Decorator for validating 7D structure preservation only.

Physical Meaning:
    Validates that results preserve the 7D phase field
    structure according to the theoretical framework.

Args:
    doma...

### decorator(func) -> Callable
**Описание:** Нет докстринга

### __init__(self)
**Описание:** Initialize with physical validation capabilities.

### _setup_physical_validation(self)
**Описание:** Setup physical validation for the class.

### validate_result_physical(self, result)
**Описание:** Validate result using physical constraints.

Physical Meaning:
    Validates that the result satisfies physical constraints
    according to the 7D phase field theory.

Args:
    result (Dict[str, Any...

### validate_result_theoretical(self, result)
**Описание:** Validate result using theoretical bounds.

Physical Meaning:
    Validates that the result is within theoretical bounds
    according to the 7D phase field theory.

Args:
    result (Dict[str, Any]): ...

### validate_result_comprehensive(self, result)
**Описание:** Validate result using both physical constraints and theoretical bounds.

Physical Meaning:
    Performs comprehensive validation of the result according to
    both physical constraints and theoretica...

### wrapper()
**Описание:** Нет докстринга
**Декораторы:** <ast.Call object at 0x753e6f636e10>

## ./bhlff/core/bvp/physical_validator.py
Methods: 19

### __init__(self, domain_shape, parameters)
**Описание:** Initialize BVP physical validator.

Physical Meaning:
    Sets up the validator with comprehensive physical constraints
    and theoretical bounds for BVP validation.

Args:
    domain_shape (Tuple[in...

### _setup_physical_constraints(self)
**Описание:** Setup physical constraints for validation.

### _setup_theoretical_bounds(self)
**Описание:** Setup theoretical bounds for validation.

### validate_physical_constraints(self, result)
**Описание:** Validate physical constraints for BVP results.

Physical Meaning:
    Validates that the BVP result satisfies all physical constraints
    including energy conservation, causality, phase coherence, an...

### validate_theoretical_bounds(self, result)
**Описание:** Validate theoretical bounds for BVP results.

Physical Meaning:
    Validates that the BVP result is within theoretical bounds
    and limits according to the 7D phase field theory framework.

Mathema...

### _setup_bvp_constraints(self)
**Описание:** Setup BVP-specific constraints.

### _validate_energy_conservation(self, field, energy, metadata)
**Описание:** Validate energy conservation.

### _validate_causality(self, field, metadata)
**Описание:** Validate causality constraints.

### _validate_phase_coherence(self, field, phase, metadata)
**Описание:** Validate phase coherence.

### _validate_7d_structure(self, field, metadata)
**Описание:** Validate 7D structure preservation.

### _validate_amplitude_bounds(self, field, metadata)
**Описание:** Validate amplitude bounds.

### _validate_gradient_bounds(self, field, metadata)
**Описание:** Validate gradient bounds.

### _validate_field_energy_bounds(self, field, metadata)
**Описание:** Validate field energy bounds.

### _validate_phase_gradient_bounds(self, field, metadata)
**Описание:** Validate phase gradient bounds.

### _validate_coherence_length_bounds(self, field, metadata)
**Описание:** Validate coherence length bounds.

### _validate_temporal_causality_bounds(self, field, metadata)
**Описание:** Validate temporal causality bounds.

### _validate_spatial_resolution_bounds(self, field, metadata)
**Описание:** Validate spatial resolution bounds.

### _estimate_coherence_length(self, field_amplitude) -> float
**Описание:** Estimate coherence length from field amplitude.

### get_validation_summary(self, physical_result, theoretical_result)
**Описание:** Get comprehensive validation summary.

Physical Meaning:
    Provides a comprehensive summary of both physical and theoretical
    validation results for easy interpretation and reporting.

Args:
    ...

## ./bhlff/core/bvp/postulates/bvp_postulates_7d.py
Methods: 4

### __init__(self, domain_7d, config)
**Описание:** Initialize all 9 BVP postulates.

Physical Meaning:
    Sets up all 9 BVP postulates with the computational domain and
    configuration parameters. Each postulate is initialized with
    its specific...

### validate_all_postulates(self, envelope_7d)
**Описание:** Validate all 9 BVP postulates.

Physical Meaning:
    Applies all 9 BVP postulates to validate the BVP field
    and ensure physical consistency. This comprehensive validation
    ensures that the BVP...

### get_postulate(self, name) -> BVPPostulate
**Описание:** Get specific postulate by name.

Physical Meaning:
    Retrieves a specific postulate by name for individual
    validation or detailed analysis of particular BVP properties.

Args:
    name (str): Po...

### __repr__(self) -> str
**Описание:** String representation of BVP postulates.

Returns:
    str: String representation showing domain and postulate count.

## ./bhlff/core/bvp/postulates/bvp_rigidity_postulate.py
Methods: 2

### __init__(self, domain_7d, config)
**Описание:** Initialize BVP Rigidity postulate.

Physical Meaning:
    Sets up the postulate with the computational domain and
    configuration parameters, including the minimum required
    rigidity ratio.

Args...

### apply(self, envelope)
**Описание:** Apply BVP Rigidity postulate.

Physical Meaning:
    Validates BVP rigidity by computing the ratio of stiffness energy
    to total energy and checking that it dominates. This ensures that
    the BVP...

## ./bhlff/core/bvp/postulates/carrier_primacy_postulate.py
Methods: 2

### __init__(self, domain_7d, config)
**Описание:** Initialize Carrier Primacy postulate.

Physical Meaning:
    Sets up the postulate with the computational domain and
    configuration parameters, including the expected carrier frequency.

Args:
    ...

### apply(self, envelope)
**Описание:** Apply Carrier Primacy postulate.

Physical Meaning:
    Validates that the field exhibits carrier primacy by checking
    that the envelope modulation is much slower than the carrier frequency.
    Th...

## ./bhlff/core/bvp/postulates/core_renormalization_postulate.py
Methods: 4

### __init__(self, domain_7d, config)
**Описание:** Initialize Core Renormalization postulate.

Physical Meaning:
    Sets up the postulate with the computational domain and
    configuration parameters, including the renormalization
    threshold for ...

### apply(self, envelope)
**Описание:** Apply Core Renormalization postulate.

Physical Meaning:
    Validates core renormalization by computing effective coefficients
    and boundary conditions from the BVP envelope. This ensures that
   ...

### _compute_effective_coefficients(self, envelope)
**Описание:** Compute effective renormalized coefficients.

Physical Meaning:
    Computes the renormalized coefficients c_i^eff(|A|,|∇A|) from
    the envelope amplitude and gradients. These coefficients represent...

### _compute_boundary_conditions(self, envelope)
**Описание:** Compute boundary pressure/stiffness.

Physical Meaning:
    Computes the boundary pressure and stiffness conditions set by
    the BVP field. These conditions represent the effective boundary
    cons...

## ./bhlff/core/bvp/postulates/power_balance/boundary_analyzer.py
Methods: 5

### __init__(self, domain_7d)
**Описание:** Initialize boundary analyzer.

Args:
    domain_7d (Domain7D): 7D computational domain.

### compute_radiation_losses(self, envelope) -> float
**Описание:** Compute EM/weak radiation and losses in 7D space-time.

Physical Meaning:
    Computes the EM/weak radiation and losses in 7D space-time M₇,
    representing the energy radiated away from the system t...

### compute_reflection(self, envelope) -> float
**Описание:** Compute reflection component in 7D space-time.

Physical Meaning:
    Computes the reflection component in 7D space-time M₇,
    representing the energy reflected back from the boundaries
    due to i...

### _split_boundary_flux_spatial(self, envelope)
**Описание:** Split spatial boundary flux into outward and inward components.

Physical Meaning:
    Separates the flux through spatial boundaries (x, y, z) into
    outward (positive) and inward (negative) compone...

### _split_boundary_flux_phase(self, envelope)
**Описание:** Split phase boundary flux into outward and inward components.

Physical Meaning:
    Separates the flux through phase boundaries (φ₁, φ₂, φ₃) into
    outward (positive) and inward (negative) componen...

## ./bhlff/core/bvp/postulates/power_balance/energy_analyzer.py
Methods: 4

### __init__(self, domain, constants)
**Описание:** Initialize energy analyzer.

Physical Meaning:
    Sets up the energy analyzer with domain and constants
    for energy growth calculations.

Args:
    domain (Domain): Computational domain for analys...

### compute_core_energy_growth(self, envelope) -> float
**Описание:** Compute growth of static core energy.

Physical Meaning:
    Calculates rate of energy growth in the core region
    from envelope dynamics.

Mathematical Foundation:
    Core energy growth is estimat...

### compute_energy_density(self, envelope)
**Описание:** Compute energy density distribution.

Physical Meaning:
    Computes the spatial distribution of energy density
    from the envelope field.

Mathematical Foundation:
    Energy density is proportiona...

### compute_total_energy(self, envelope) -> float
**Описание:** Compute total energy of the system.

Physical Meaning:
    Computes the total energy of the BVP system
    from the envelope field.

Mathematical Foundation:
    Total energy is the integral of energy...

## ./bhlff/core/bvp/postulates/power_balance/energy_computer.py
Methods: 3

### __init__(self, domain_7d, config)
**Описание:** Initialize energy computer.

Args:
    domain_7d (Domain7D): 7D computational domain.
    config (Dict[str, Any]): Configuration parameters.

### compute_core_energy_growth(self, envelope) -> float
**Описание:** Compute growth of static core energy in 7D space-time.

Physical Meaning:
    Computes the growth of static core energy in 7D space-time M₇,
    representing the energy stored in the core region of th...

### _energy_density_7d(self, a, dx, dy, dz, dphi1, dphi2, dphi3, f_phi, k0, beta4, gamma6)
**Описание:** Compute 7D energy density according to theory.

Physical Meaning:
    Implements the 7D energy functional:
    E[Θ] = f_φ²|∇_xΘ|² + f_φ²|∇_φΘ|² + β₄(ΔΘ)² + γ₆|∇Θ|⁶

Args:
    a (np.ndarray): 7D field ...

## ./bhlff/core/bvp/postulates/power_balance/flux_computer.py
Methods: 3

### __init__(self, domain, constants)
**Описание:** Initialize flux computer.

Physical Meaning:
    Sets up the flux computer with domain and constants
    for boundary flux calculations.

Args:
    domain (Domain): Computational domain for analysis.
...

### compute_bvp_flux(self, envelope) -> float
**Описание:** Compute BVP flux at external boundary.

Physical Meaning:
    Calculates energy flux across external boundaries
    from amplitude gradients.

Mathematical Foundation:
    Flux is proportional to grad...

### compute_boundary_gradients(self, envelope)
**Описание:** Compute gradients at all boundaries.

Physical Meaning:
    Computes gradient components at all boundary faces
    for detailed flux analysis.

Args:
    envelope (np.ndarray): BVP envelope.

Returns:...

## ./bhlff/core/bvp/postulates/power_balance/power_balance_postulate.py
Methods: 4

### __init__(self, domain, constants)
**Описание:** Initialize power balance postulate.

Physical Meaning:
    Sets up the postulate for analyzing power balance
    at external boundaries.

Args:
    domain (Domain): Computational domain for analysis.
...

### apply(self, envelope)
**Описание:** Apply power balance postulate.

Physical Meaning:
    Verifies that power balance is maintained at the external
    boundary with proper accounting of energy flows.

Mathematical Foundation:
    Check...

### _analyze_power_balance(self, bvp_flux, core_energy_growth, radiation_losses, reflection)
**Описание:** Analyze power balance components.

Physical Meaning:
    Computes power balance ratio and error to verify
    energy conservation.

Mathematical Foundation:
    Balance ratio = BVP_flux / (core_growth...

### _validate_power_balance(self, power_balance) -> bool
**Описание:** Validate that power balance is maintained.

Physical Meaning:
    Checks if power balance error is within acceptable
    tolerance for energy conservation.

Args:
    power_balance (Dict[str, Any]): P...

## ./bhlff/core/bvp/postulates/power_balance/radiation_calculator.py
Methods: 4

### __init__(self, domain, constants)
**Описание:** Initialize radiation calculator.

Physical Meaning:
    Sets up the radiation calculator with domain and constants
    for radiation and reflection calculations.

Args:
    domain (Domain): Computatio...

### compute_radiation_losses(self, envelope) -> float
**Описание:** Compute EM/weak radiation and losses.

Physical Meaning:
    Calculates energy losses due to electromagnetic and
    weak radiation from the envelope using full field theory.

Mathematical Foundation:...

### compute_reflection(self, envelope) -> float
**Описание:** Compute reflection at boundaries.

Physical Meaning:
    Calculates energy reflection at boundaries due to
    impedance mismatch using full electromagnetic theory.

Mathematical Foundation:
    Refle...

### _compute_envelope_admittance(self, envelope) -> float
**Описание:** Compute envelope admittance from field properties.

Physical Meaning:
    Calculates admittance from envelope gradient and amplitude
    using transmission line theory.

Mathematical Foundation:
    Y...

## ./bhlff/core/bvp/postulates/quenches_postulate.py
Methods: 2

### __init__(self, domain_7d, config)
**Описание:** Initialize Quenches postulate.

Physical Meaning:
    Sets up the postulate with the computational domain and
    configuration parameters, including the various threshold
    values for quench detect...

### apply(self, envelope)
**Описание:** Apply Quenches postulate.

Physical Meaning:
    Detects quench events by identifying local threshold crossings
    in amplitude, detuning, and gradient. These quenches represent
    dissipative energ...

## ./bhlff/core/bvp/postulates/scale_separation_postulate.py
Methods: 2

### __init__(self, domain_7d, config)
**Описание:** Initialize Scale Separation postulate.

Physical Meaning:
    Sets up the postulate with the computational domain and
    configuration parameters, including the maximum allowed
    scale separation p...

### apply(self, envelope)
**Описание:** Apply Scale Separation postulate.

Physical Meaning:
    Validates that the scale separation parameter ε = Ω/ω₀ << 1
    is satisfied by analyzing the frequency content of the envelope.
    This ensur...

## ./bhlff/core/bvp/postulates/tail_resonatorness_postulate.py
Methods: 3

### __init__(self, domain_7d, config)
**Описание:** Initialize Tail Resonatorness postulate.

Physical Meaning:
    Sets up the postulate with the computational domain and
    configuration parameters, including the minimum required
    resonance count...

### apply(self, envelope)
**Описание:** Apply Tail Resonatorness postulate.

Physical Meaning:
    Validates tail resonatorness by analyzing the frequency spectrum
    and identifying resonant modes with their quality factors. This
    ensu...

### _find_resonance_peaks(self, power_spectrum)
**Описание:** Find resonance peaks in power spectrum.

Physical Meaning:
    Identifies resonance peaks in the power spectrum that correspond
    to the cascade of effective resonators in the tail. These peaks
    ...

## ./bhlff/core/bvp/postulates/transition_zone_postulate.py
Methods: 4

### __init__(self, domain_7d, config)
**Описание:** Initialize Transition Zone postulate.

Physical Meaning:
    Sets up the postulate with the computational domain and
    configuration parameters, including the nonlinear threshold
    for transition ...

### apply(self, envelope)
**Описание:** Apply Transition Zone postulate.

Physical Meaning:
    Validates transition zone by computing nonlinear admittance
    and current generation from the envelope. This ensures that
    the transition z...

### _compute_nonlinear_admittance(self, envelope) -> float
**Описание:** Compute nonlinear admittance.

Physical Meaning:
    Computes the nonlinear admittance Y_tr(ω,|A|) from the envelope
    amplitude. This admittance characterizes the nonlinear interface
    properties...

### _compute_current_generation(self, envelope)
**Описание:** Compute current generation.

Physical Meaning:
    Computes the effective EM/weak currents J(ω) generated from
    the envelope. These currents arise from the nonlinear interface
    characteristics o...

## ./bhlff/core/bvp/postulates/u1_phase_structure_postulate.py
Methods: 8

### __init__(self, domain_7d, config)
**Описание:** Initialize U(1)³ Phase Structure postulate.

Physical Meaning:
    Sets up the postulate with the computational domain and
    configuration parameters, including the minimum required
    phase cohere...

### apply(self, envelope)
**Описание:** Apply U(1)³ Phase Structure postulate.

Physical Meaning:
    Validates U(1)³ phase structure by checking phase coherence
    and electroweak current generation. This ensures that the
    BVP field ex...

### _compute_phase_coherence(self, phase_1, phase_2, phase_3) -> float
**Описание:** Compute phase coherence measure.

Physical Meaning:
    Computes the coherence between the three phase components
    by analyzing their cross-correlations. High coherence indicates
    proper U(1)³ p...

### _compute_electroweak_currents(self, phase_1, phase_2, phase_3)
**Описание:** Compute electroweak currents.

Physical Meaning:
    Computes the electroweak currents generated by the three
    phase components. These currents arise as functionals of
    the envelope and represen...

### _extract_phase_components(self, envelope)
**Описание:** Extract genuine U(1)³ phase components from 7D envelope field.

Physical Meaning:
    Extracts the three independent U(1) phase components from the
    7D envelope field, ensuring proper phase structu...

### _compute_phase_component_1(self, envelope)
**Описание:** Compute first U(1) phase component from 7D envelope.

Physical Meaning:
    Computes the first independent U(1) phase component Θ₁
    from the 7D envelope field. This component represents
    the ele...

### _compute_phase_component_2(self, envelope)
**Описание:** Compute second U(1) phase component from 7D envelope.

Physical Meaning:
    Computes the second independent U(1) phase component Θ₂
    from the 7D envelope field. This component represents
    the w...

### _compute_phase_component_3(self, envelope)
**Описание:** Compute third U(1) phase component from 7D envelope.

Physical Meaning:
    Computes the third independent U(1) phase component Θ₃
    from the 7D envelope field. This component represents
    the mix...

## ./bhlff/core/bvp/power_law/power_law_comparison.py
Methods: 5

### __init__(self, bvp_core)
**Описание:** Initialize power law comparison analyzer.

### compare_power_laws(self, envelope1, envelope2)
**Описание:** Compare power law behavior between two envelope fields.

Physical Meaning:
    Compares power law characteristics between two envelope
    fields to analyze differences in their long-range behavior.

...

### _compare_exponents(self, results1, results2)
**Описание:** Compare power law exponents between results.

### _compare_quality(self, results1, results2)
**Описание:** Compare fitting quality between results.

### _calculate_statistical_significance(self, results1, results2)
**Описание:** Calculate statistical significance of differences.

## ./bhlff/core/bvp/power_law/power_law_core.py
Methods: 10

### __init__(self, bvp_core)
**Описание:** Initialize power law analyzer.

Args:
    bvp_core (BVPCore): BVP core instance for field access.

### analyze_envelope_power_laws(self, envelope)
**Описание:** Analyze power law behavior in envelope field.

Physical Meaning:
    Analyzes power law behavior in the envelope field by
    identifying tail regions and fitting power laws to them.

Args:
    envelo...

### analyze_power_law_tails(self, envelope)
**Описание:** Analyze power law tails in BVP envelope field.

Physical Meaning:
    Analyzes the power law decay of BVP envelope amplitude in the
    tail region, which characterizes the field's long-range behavior...

### _identify_tail_regions(self, envelope)
**Описание:** Identify tail regions in the envelope field.

Physical Meaning:
    Identifies regions in the envelope field that exhibit
    power law behavior, typically in the tails of the distribution.

Args:
   ...

### _find_dimension_tail_regions(self, dim_slice, dimension)
**Описание:** Find tail regions in a specific dimension.

Physical Meaning:
    Finds regions in a specific dimension that exhibit
    power law behavior based on amplitude thresholds.

Args:
    dim_slice (np.ndar...

### _find_contiguous_regions(self, mask)
**Описание:** Find contiguous regions in a boolean mask.

Physical Meaning:
    Finds contiguous regions of True values in a boolean mask,
    representing regions that satisfy the tail criteria.

Args:
    mask (n...

### _analyze_region_power_law(self, envelope, region)
**Описание:** Analyze power law behavior in a specific region.

Physical Meaning:
    Analyzes power law behavior in a specific region of the
    envelope field by fitting power law functions to the data.

Args:
  ...

### _extract_region_data(self, envelope, region)
**Описание:** Extract data from a specific region.

Physical Meaning:
    Extracts relevant data from a specific region of the
    envelope field for power law analysis.

Args:
    envelope (np.ndarray): 7D envelop...

### _fit_power_law(self, region_data)
**Описание:** Fit power law to region data.

Physical Meaning:
    Fits a power law function to the region data to determine
    the power law exponent and coefficient.

Args:
    region_data (Dict[str, np.ndarray]...

### _calculate_fitting_quality(self, region_data, power_law_fit) -> float
**Описание:** Calculate quality of power law fit.

Physical Meaning:
    Calculates the quality of the power law fit based on
    statistical measures and physical constraints.

Args:
    region_data (Dict[str, np....

## ./bhlff/core/bvp/power_law/power_law_optimization.py
Methods: 6

### __init__(self, bvp_core)
**Описание:** Initialize power law optimization analyzer.

### optimize_power_law_fits(self, envelope)
**Описание:** Optimize power law fits for better accuracy.

Physical Meaning:
    Optimizes power law fits using advanced fitting techniques
    to improve accuracy and reliability.

Args:
    envelope (np.ndarray)...

### _optimize_region_fit(self, envelope, region)
**Описание:** Optimize power law fit for a specific region.

### _iterative_refinement(self, region_data, initial_fit)
**Описание:** Perform iterative refinement of power law fit.

### _adjust_fit_parameters(self, fit_params)
**Описание:** Adjust fit parameters for optimization.

### _calculate_optimization_quality(self, optimized_results)
**Описание:** Calculate quality of optimization results.

## ./bhlff/core/bvp/power_law/power_law_statistics.py
Methods: 5

### __init__(self, bvp_core)
**Описание:** Initialize power law statistics analyzer.

### analyze_power_law_statistics(self, envelope)
**Описание:** Analyze statistical properties of power law behavior.

Physical Meaning:
    Analyzes statistical properties of power law behavior
    in the envelope field.

Args:
    envelope (np.ndarray): 7D envel...

### _calculate_statistical_metrics(self, envelope)
**Описание:** Calculate statistical metrics for power law analysis.

Physical Meaning:
    Computes real statistical metrics for power law behavior
    analysis from the envelope data.

### _perform_hypothesis_testing(self, envelope)
**Описание:** Perform hypothesis testing for power law behavior.

Physical Meaning:
    Tests the null hypothesis that the data follows a power law
    distribution using statistical tests.

### _calculate_confidence_intervals(self, envelope)
**Описание:** Calculate confidence intervals for power law parameters.

Physical Meaning:
    Computes confidence intervals for power law parameters
    using bootstrap resampling.

## ./bhlff/core/bvp/power_law_analysis.py
Methods: 10

### __init__(self, domain, config, constants)
**Описание:** Initialize power law analyzer.

Physical Meaning:
    Sets up the power law analyzer with the computational domain
    and configuration parameters for analyzing scaling behavior
    in the BVP field....

### _setup_analysis_parameters(self)
**Описание:** Setup analysis parameters.

Physical Meaning:
    Initializes parameters for power law analysis based on
    the domain properties and configuration.

### analyze_power_law(self, field)
**Описание:** Analyze power law behavior in the field.

Physical Meaning:
    Computes power law exponents and scaling behavior
    in the BVP field according to the theoretical framework.

Mathematical Foundation:...
**Декораторы:** <ast.Call object at 0x753e6f755c90>

### _compute_correlation_function(self, field)
**Описание:** Compute correlation function for power law analysis.

Physical Meaning:
    Computes the spatial correlation function C(r) which
    characterizes the scaling behavior of the BVP field.

Mathematical ...

### _analyze_scaling_behavior(self, correlation_func)
**Описание:** Analyze scaling behavior in correlation function.

Physical Meaning:
    Identifies regions of power law behavior in the
    correlation function and determines scaling properties.

Mathematical Found...

### _find_scaling_region(self, correlation_func)
**Описание:** Find region of power law behavior.

Physical Meaning:
    Identifies the range of distances where the correlation
    function exhibits power law behavior.

Args:
    correlation_func (np.ndarray): Co...

### _compute_scaling_properties(self, correlation_func, scaling_region)
**Описание:** Compute scaling properties in the scaling region.

Physical Meaning:
    Computes the scaling properties including the power law
    exponent and scaling quality in the identified region.

Args:
    c...

### _compute_critical_exponent(self, correlation_func, scaling_analysis) -> float
**Описание:** Compute critical exponent from scaling analysis.

Physical Meaning:
    Computes the critical exponent α from the power law
    behavior C(r) ~ r^(-α) in the scaling region.

Args:
    correlation_fun...

### _compute_quality_metrics(self, correlation_func, scaling_analysis)
**Описание:** Compute quality metrics for the power law analysis.

Physical Meaning:
    Computes metrics to assess the quality of the power law
    fit and scaling behavior identification.

Args:
    correlation_f...

### get_analysis_parameters(self)
**Описание:** Get current analysis parameters.

Physical Meaning:
    Returns the current parameters used for power law analysis.

Returns:
    Dict[str, Any]: Analysis parameters.

## ./bhlff/core/bvp/power_law_core_modules/power_law_core_main.py
Methods: 3

### __init__(self, bvp_core)
**Описание:** Initialize unified power law analyzer.

Physical Meaning:
    Sets up the analyzer with the BVP core for accessing
    field data and computational resources.

Args:
    bvp_core (BVPCore): BVP core i...

### analyze_power_laws(self, envelope)
**Описание:** Analyze power law behavior of BVP envelope field.

Physical Meaning:
    Analyzes the power law decay characteristics of the BVP envelope
    field, which describes the long-range behavior of the fiel...

### _calculate_overall_characteristics(self, power_law_results)
**Описание:** Calculate overall characteristics from power law results.

Physical Meaning:
    Computes overall characteristics of the power law behavior
    across all analyzed regions.

Args:
    power_law_result...

## ./bhlff/core/bvp/power_law_core_modules/power_law_fitting.py
Methods: 4

### __init__(self, bvp_core)
**Описание:** Initialize power law fitting.

### fit_power_law(self, region_data)
**Описание:** Fit power law to region data.

Physical Meaning:
    Fits a power law function to the region data
    to determine the decay characteristics.

Args:
    region_data (Dict[str, np.ndarray]): Region dat...

### calculate_fitting_quality(self, region_data, power_law_fit) -> float
**Описание:** Calculate fitting quality metric.

Physical Meaning:
    Calculates a quality metric for the power law fit
    to assess the reliability of the analysis.

Args:
    region_data (Dict[str, np.ndarray])...

### calculate_decay_rate(self, power_law_fit) -> float
**Описание:** Calculate decay rate from power law fit.

Physical Meaning:
    Calculates the decay rate from the power law exponent
    to characterize the field behavior.

Args:
    power_law_fit (Dict[str, float]...

## ./bhlff/core/bvp/power_law_core_modules/power_law_region_analysis.py
Methods: 4

### __init__(self, bvp_core)
**Описание:** Initialize power law region analysis.

### analyze_regions(self, envelope)
**Описание:** Analyze power law behavior in different regions.

Physical Meaning:
    Analyzes the power law behavior in different regions
    of the BVP envelope field.

Args:
    envelope (np.ndarray): BVP envelo...

### _find_dimension_tail_regions(self, envelope, dimension)
**Описание:** Find tail regions in a specific dimension.

### _analyze_region_power_law(self, envelope, region)
**Описание:** Analyze power law behavior in a specific region.

## ./bhlff/core/bvp/power_law_core_modules/power_law_tail_analysis.py
Methods: 4

### __init__(self, bvp_core)
**Описание:** Initialize power law tail analysis.

### analyze_power_law_tails(self, envelope)
**Описание:** Analyze power law behavior in tail regions.

Physical Meaning:
    Analyzes the power law decay in tail regions of the
    BVP envelope field.

Args:
    envelope (np.ndarray): BVP envelope field data...

### _identify_tail_regions(self, envelope)
**Описание:** Identify tail regions in the envelope field.

### _analyze_tail_region(self, envelope, region)
**Описание:** Analyze a specific tail region.

## ./bhlff/core/bvp/quench_characteristics.py
Methods: 7

### __init__(self, domain_7d)
**Описание:** Initialize quench characteristics computer.

Physical Meaning:
    Sets up the characteristics computer with the computational
    domain to compute quench event properties.

Args:
    domain_7d (Doma...

### compute_center_of_mass(self, component_mask)
**Описание:** Compute center of mass for a quench component.

Physical Meaning:
    Calculates the center of mass of a quench component,
    representing the effective location of the quench event
    in 7D space-t...

### compute_quench_strength(self, component_mask, amplitude) -> float
**Описание:** Compute quench strength for a component.

Physical Meaning:
    Calculates the strength of a quench event based on
    the maximum amplitude within the component region.

Mathematical Foundation:
    ...

### compute_local_frequency(self, envelope)
**Описание:** Compute local frequency from phase evolution.

Physical Meaning:
    Calculates the local frequency at each point in 7D space-time
    by analyzing the phase evolution of the envelope field.
    This ...

### compute_detuning_strength(self, component_mask, detuning) -> float
**Описание:** Compute detuning strength for a component.

Physical Meaning:
    Calculates the strength of a detuning quench event based on
    the maximum detuning within the component region.

Mathematical Founda...

### compute_7d_gradient_magnitude(self, envelope)
**Описание:** Compute 7D gradient magnitude of envelope field.

Physical Meaning:
    Calculates the magnitude of the gradient in all 7 dimensions
    (3 spatial + 3 phase + 1 temporal), representing the rate
    o...

### compute_gradient_strength(self, component_mask, gradient_magnitude) -> float
**Описание:** Compute gradient strength for a component.

Physical Meaning:
    Calculates the strength of a gradient quench event based on
    the maximum gradient magnitude within the component region.

Mathemati...

## ./bhlff/core/bvp/quench_detector.py
Methods: 6

### __init__(self, domain_7d, config)
**Описание:** Initialize quench detector.

Physical Meaning:
    Sets up the quench detector with threshold parameters
    for detecting amplitude, detuning, and gradient quenches
    in the 7D BVP field.

Args:
  ...

### detect_quenches(self, envelope)
**Описание:** Detect quench events based on three thresholds.

Physical Meaning:
    Applies three threshold criteria to detect quench events:
    - amplitude: |A| > |A_q| - detects high-amplitude quenches
    - de...

### _detect_amplitude_quenches(self, envelope)
**Описание:** Detect amplitude quenches: |A| > |A_q| with advanced processing.

Physical Meaning:
    Detects locations where the envelope amplitude exceeds
    the amplitude threshold, indicating potential quench ...

### _detect_detuning_quenches(self, envelope)
**Описание:** Detect detuning quenches: |ω - ω_0| > Δω_q with advanced processing.

Physical Meaning:
    Detects locations where the local frequency deviates
    significantly from the carrier frequency, indicatin...

### _detect_gradient_quenches(self, envelope)
**Описание:** Detect gradient quenches: |∇A| > |∇A_q| with advanced processing.

Physical Meaning:
    Detects locations where the envelope gradient exceeds
    the gradient threshold, indicating potential quench e...

### _validate_thresholds(self)
**Описание:** Validate threshold parameters.

Physical Meaning:
    Ensures that threshold parameters are physically reasonable
    and consistent with the BVP theory.

Raises:
    ValueError: If thresholds are inv...

## ./bhlff/core/bvp/quench_morphology.py
Methods: 8

### __init__(self)
**Описание:** Initialize morphological operations processor.

### apply_morphological_operations(self, mask)
**Описание:** Apply morphological operations to filter noise in quench mask.

Physical Meaning:
    Applies binary morphological operations to remove noise
    and fill gaps in quench regions, improving detection q...

### find_connected_components(self, mask)
**Описание:** Find connected components in quench mask.

Physical Meaning:
    Groups nearby quench events into connected components,
    representing coherent quench regions in 7D space-time.

Mathematical Foundat...

### _apply_scipy_operations(self, mask)
**Описание:** Apply morphological operations using scipy.

Physical Meaning:
    Uses scipy's optimized morphological operations for
    efficient noise filtering in 7D space-time.

Args:
    mask (np.ndarray): Bin...

### _apply_simple_operations(self, mask)
**Описание:** Simple morphological filtering without scipy dependency.

Physical Meaning:
    Basic noise filtering using local neighborhood operations
    to remove isolated pixels and fill small gaps.

Args:
    ...

### _find_scipy_components(self, mask)
**Описание:** Find connected components using scipy.

Physical Meaning:
    Uses scipy's optimized connected component labeling for
    efficient component identification in 7D space-time.

Args:
    mask (np.ndarr...

### _find_simple_components(self, mask)
**Описание:** Simple connected component analysis without scipy.

Physical Meaning:
    Basic grouping of nearby quench events using
    flood-fill algorithm for 7D space.

Args:
    mask (np.ndarray): Binary mask ...

### _flood_fill_7d(self, mask, visited, component_mask, start_point)
**Описание:** Flood-fill algorithm for 7D connected components.

Physical Meaning:
    Recursively fills connected quench regions starting from
    a seed point, identifying coherent quench structures.

Args:
    m...

## ./bhlff/core/bvp/quench_thresholds.py
Methods: 6

### __init__(self, domain_7d)
**Описание:** Initialize quench threshold computer.

Physical Meaning:
    Sets up the threshold computer with the computational domain
    to compute physical thresholds based on domain properties
    and theoreti...

### compute_all_thresholds(self)
**Описание:** Compute all quench thresholds from physical principles.

Physical Meaning:
    Computes all quench thresholds (amplitude, detuning, gradient,
    carrier frequency) based on the physical properties of...

### compute_amplitude_threshold(self) -> float
**Описание:** Compute amplitude threshold from field energy density.

Physical Meaning:
    Computes the amplitude threshold based on the energy density
    of the BVP field. The threshold represents the critical
 ...

### compute_detuning_threshold(self) -> float
**Описание:** Compute detuning threshold from frequency analysis.

Physical Meaning:
    Computes the detuning threshold based on the frequency
    characteristics of the BVP field. The threshold represents
    the...

### compute_gradient_threshold(self) -> float
**Описание:** Compute gradient threshold from field gradients.

Physical Meaning:
    Computes the gradient threshold based on the gradient
    characteristics of the BVP field. The threshold represents
    the cri...

### compute_carrier_frequency(self) -> float
**Описание:** Compute carrier frequency from domain properties.

Physical Meaning:
    Computes the carrier frequency based on the temporal
    characteristics of the BVP field. The carrier frequency
    represents...

## ./bhlff/core/bvp/quenches_analyzer.py
Methods: 6

### __init__(self, domain, constants)
**Описание:** Initialize quenches analyzer.

Args:
    domain (Domain): Computational domain for analysis.
    constants (BVPConstants): BVP physical constants.

### analyze_quench_properties(self, envelope, quench_detection)
**Описание:** Analyze properties of detected quenches.

Physical Meaning:
    Computes detailed properties of quench events
    including size, shape, and amplitude characteristics.

Args:
    envelope (np.ndarray)...

### analyze_energy_dumps(self, envelope, quench_detection)
**Описание:** Analyze energy dumps at quench locations.

Physical Meaning:
    Computes energy dissipation at quench locations
    to quantify energy dump events.

Args:
    envelope (np.ndarray): BVP envelope.
   ...

### _analyze_individual_quench(self, amplitude, quench_mask, location, quench_id)
**Описание:** Analyze properties of individual quench.

Physical Meaning:
    Computes detailed properties of a single quench
    event including size, amplitude, and depth.

Args:
    amplitude (np.ndarray): Field...

### _extract_quench_region(self, amplitude, quench_mask, location)
**Описание:** Extract region around quench location.

Physical Meaning:
    Creates mask for quench region around specified
    location for detailed analysis.

Args:
    amplitude (np.ndarray): Field amplitude.
  ...

### _compute_surrounding_amplitude(self, amplitude, location) -> float
**Описание:** Compute amplitude in surrounding region.

Physical Meaning:
    Calculates average amplitude in region surrounding
    quench to determine quench depth.

Args:
    amplitude (np.ndarray): Field amplit...

## ./bhlff/core/bvp/quenches_detector.py
Methods: 4

### __init__(self, domain, constants)
**Описание:** Initialize quenches detector.

Args:
    domain (Domain): Computational domain for analysis.
    constants (BVPConstants): BVP physical constants.

### detect_quenches(self, envelope)
**Описание:** Detect quench events in the field.

Physical Meaning:
    Identifies localized regions where field amplitude
    drops below critical thresholds.

Args:
    envelope (np.ndarray): BVP envelope.

Retur...

### _filter_small_quenches(self, quench_mask)
**Описание:** Filter out small quench regions.

Physical Meaning:
    Removes quench regions that are too small to be
    physically significant.

Args:
    quench_mask (np.ndarray): Binary quench mask.

Returns:
 ...

### _find_quench_locations(self, quench_mask)
**Описание:** Find center locations of quench regions.

Physical Meaning:
    Identifies center coordinates of each quench region
    for further analysis.

Args:
    quench_mask (np.ndarray): Binary quench mask.

...

## ./bhlff/core/bvp/quenches_postulate.py
Methods: 3

### __init__(self, domain, constants)
**Описание:** Initialize quenches postulate.

Physical Meaning:
    Sets up the postulate with domain and constants for
    detecting and analyzing quench events.

Args:
    domain (Domain): Computational domain fo...

### apply(self, envelope)
**Описание:** Apply quenches postulate.

Physical Meaning:
    Detects and analyzes quench events in the BVP field,
    including energy dumps and phase discontinuities.

Mathematical Foundation:
    Identifies reg...

### _validate_quenches(self, quench_analysis, energy_analysis) -> bool
**Описание:** Validate that quenches satisfy the postulate.

Physical Meaning:
    Checks that detected quenches exhibit the expected
    properties of energy dissipation and phase discontinuities.

Args:
    quenc...

## ./bhlff/core/bvp/quenches_postulate_core.py
Methods: 3

### __init__(self, domain, constants)
**Описание:** Initialize quenches postulate.

Physical Meaning:
    Sets up the postulate with domain and constants for
    detecting and analyzing quench events.

Args:
    domain (Domain): Computational domain fo...

### apply(self, envelope)
**Описание:** Apply quenches postulate.

Physical Meaning:
    Detects and analyzes quench events in the BVP field,
    including energy dumps and phase discontinuities.

Mathematical Foundation:
    Identifies reg...

### _validate_quenches(self, quench_analysis, energy_analysis) -> bool
**Описание:** Validate that quenches satisfy the postulate.

Physical Meaning:
    Checks that detected quenches exhibit the expected
    properties of energy dissipation and phase discontinuities.

Args:
    quenc...

## ./bhlff/core/bvp/residual_computer_base.py
Methods: 7

### __init__(self, domain, config_or_constants)
**Описание:** Initialize residual computer base.

Physical Meaning:
    Sets up the base residual computer with the computational domain
    and configuration parameters or constants for computing residuals
    of ...

### _setup_parameters(self)
**Описание:** Setup envelope equation parameters.

Physical Meaning:
    Initializes the parameters needed for computing residuals
    of the envelope equation, including stiffness and susceptibility
    coefficien...
**Декораторы:** abstractmethod

### compute_residual(self, envelope, source)
**Описание:** Compute residual of the envelope equation.

Physical Meaning:
    Computes the residual R = ∇·(κ(|a|)∇a) + k₀²χ(|a|)a - s(x,φ,t)
    for the current envelope solution, representing how well
    the so...
**Декораторы:** abstractmethod

### _compute_div_kappa_grad(self, envelope, kappa)
**Описание:** Compute divergence of kappa times gradient.

Physical Meaning:
    Computes the divergence of κ times the gradient of the envelope
    using appropriate finite difference methods for the domain type.
...
**Декораторы:** abstractmethod

### compute_residual_norm(self, residual) -> float
**Описание:** Compute norm of residual for convergence checking.

Physical Meaning:
    Computes the L2 norm of the residual vector for monitoring
    convergence of the Newton-Raphson iterations.

Args:
    residu...

### analyze_residual_components(self, envelope, source)
**Описание:** Analyze components of the residual.

Physical Meaning:
    Analyzes the individual components of the residual to understand
    the relative contributions of different terms in the equation.

Args:
  ...

### __repr__(self) -> str
**Описание:** String representation of residual computer.

## ./bhlff/core/bvp/resonance_detector.py
Methods: 5

### __init__(self, constants)
**Описание:** Initialize resonance detector.

Args:
    constants (BVPConstants, optional): BVP constants instance.

### find_resonance_peaks(self, frequencies, admittance)
**Описание:** Find resonance peaks in admittance.

Physical Meaning:
    Identifies resonance frequencies and quality factors
    from the admittance spectrum.

Args:
    frequencies (np.ndarray): Frequency array.
...

### set_quality_factor_threshold(self, threshold)
**Описание:** Set quality factor threshold for peak filtering.

Args:
    threshold (float): Quality factor threshold.

### get_quality_factor_threshold(self) -> float
**Описание:** Get current quality factor threshold.

Returns:
    float: Current quality factor threshold.

### __repr__(self) -> str
**Описание:** String representation of resonance detector.

## ./bhlff/core/bvp/resonance_peak_detector.py
Methods: 7

### __init__(self, constants)
**Описание:** Initialize peak detector.

Args:
    constants (BVPConstants): BVP constants instance.

### detect_peaks(self, frequencies, magnitude, phase)
**Описание:** Advanced peak detection using multiple criteria and signal processing.

Physical Meaning:
    Identifies resonance peaks using advanced signal processing
    techniques including magnitude, phase, and...

### _smooth_signal(self, signal, window_size)
**Описание:** Smooth signal using moving average filter.

Physical Meaning:
    Reduces noise while preserving peak characteristics
    using polynomial smoothing.

Args:
    signal (np.ndarray): Input signal.
    ...

### _find_prominent_peaks(self, magnitude)
**Описание:** Find prominent peaks using height and prominence criteria.

Physical Meaning:
    Identifies peaks that are significantly higher than
    surrounding values and have sufficient prominence.

Args:
    ...

### _find_phase_peaks(self, phase)
**Описание:** Find peaks based on phase behavior analysis.

Physical Meaning:
    Identifies peaks where phase changes rapidly,
    indicating resonance behavior.

Args:
    phase (np.ndarray): Signal phase.

Retur...

### _find_sharp_peaks(self, magnitude)
**Описание:** Find sharp peaks using second derivative analysis.

Physical Meaning:
    Identifies peaks with high sharpness using
    second derivative analysis.

Args:
    magnitude (np.ndarray): Signal magnitude...

### _combine_peak_criteria(self, magnitude_peaks, phase_peaks, sharpness_peaks)
**Описание:** Combine peak detection criteria.

Physical Meaning:
    Combines results from different peak detection methods
    to identify the most reliable resonance peaks.

Args:
    magnitude_peaks (List[int])...

## ./bhlff/core/bvp/resonance_quality_core.py
Methods: 9

### __init__(self, constants)
**Описание:** Initialize quality analyzer.

Args:
    constants (BVPConstants): BVP constants instance.

### calculate_quality_factors(self, frequencies, magnitude, peak_indices)
**Описание:** Calculate quality factors for multiple resonance peaks.

Physical Meaning:
    Calculates quality factors for multiple resonance peaks
    using Lorentzian fitting and FWHM analysis.

Mathematical Fou...

### calculate_quality_factor(self, frequencies, magnitude, peak_idx) -> float
**Описание:** Calculate quality factor for a single resonance peak.

Physical Meaning:
    Calculates quality factor for a single resonance peak
    using Lorentzian fitting and FWHM analysis.

Mathematical Foundat...

### analyze_resonance_quality(self, frequencies, magnitude, peak_indices)
**Описание:** Analyze resonance quality for multiple peaks.

Physical Meaning:
    Performs comprehensive analysis of resonance quality
    for multiple peaks, including quality factors, FWHM,
    and resonance cha...

### _extract_peak_region(self, frequencies, magnitude, peak_idx)
**Описание:** Extract region around a resonance peak.

Args:
    frequencies (np.ndarray): Frequency array.
    magnitude (np.ndarray): Magnitude response array.
    peak_idx (int): Index of the resonance peak.

Re...

### _fit_lorentzian(self, peak_region)
**Описание:** Fit Lorentzian function to peak region.

Mathematical Foundation:
    Lorentzian function: L(f) = A / (1 + ((f - f₀) / (Δf/2))²)
    where A is amplitude, f₀ is center frequency, and Δf is FWHM.

Args...

### _calculate_fwhm(self, lorentzian_params) -> float
**Описание:** Calculate full width at half maximum from Lorentzian parameters.

Args:
    lorentzian_params (Dict[str, float]): Lorentzian parameters.

Returns:
    float: FWHM value.

### validate_quality_factor(self, quality_factor) -> bool
**Описание:** Validate quality factor value.

Args:
    quality_factor (float): Quality factor to validate.

Returns:
    bool: True if quality factor is valid, False otherwise.

### calculate_quality_factor_statistics(self, quality_factors)
**Описание:** Calculate statistics for quality factors.

Args:
    quality_factors (List[float]): List of quality factors.

Returns:
    Dict[str, float]: Quality factor statistics.

## ./bhlff/core/bvp/scale_separation_postulate.py
Methods: 6

### __init__(self, domain, constants)
**Описание:** Initialize scale separation postulate.

Physical Meaning:
    Sets up the postulate with domain and constants for
    analyzing length scale separation.

Args:
    domain (Domain): Computational domai...

### apply(self, envelope)
**Описание:** Apply scale separation postulate.

Physical Meaning:
    Verifies that envelope length scale is much larger than
    carrier wavelength, ensuring proper scale separation.

Mathematical Foundation:
   ...

### _analyze_envelope_length_scale(self, envelope)
**Описание:** Analyze envelope length scale from spatial gradients.

Physical Meaning:
    Computes characteristic length scale of envelope variations
    from spatial gradient analysis.

Mathematical Foundation:
 ...

### _compute_carrier_wavelength(self) -> float
**Описание:** Compute carrier wavelength.

Physical Meaning:
    Calculates carrier wavelength from carrier frequency
    and propagation speed.

Mathematical Foundation:
    λ₀ = 2πc/ω₀ where c is propagation spee...

### _check_scale_separation(self, length_scale_analysis, carrier_wavelength)
**Описание:** Check scale separation between envelope and carrier.

Physical Meaning:
    Verifies that envelope length scale is much larger than
    carrier wavelength.

Args:
    length_scale_analysis (Dict[str, ...

### _validate_scale_separation(self, scale_separation) -> bool
**Описание:** Validate scale separation postulate.

Physical Meaning:
    Checks that scale separation is sufficient for envelope
    approximation validity.

Args:
    scale_separation (Dict[str, Any]): Scale sepa...

## ./bhlff/core/bvp/tail_resonatorness_postulate.py
Methods: 9

### __init__(self, domain, constants)
**Описание:** Initialize tail resonatorness postulate.

Physical Meaning:
    Sets up the postulate with domain and constants for
    analyzing resonator-like behavior in particle tails.

Args:
    domain (Domain):...

### apply(self, envelope)
**Описание:** Apply tail resonatorness postulate.

Physical Meaning:
    Verifies that the tail exhibits resonator-like behavior
    with frequency-dependent impedance and resonance peaks.

Mathematical Foundation:...

### _compute_frequency_spectrum(self, envelope)
**Описание:** Compute frequency spectrum of the envelope.

Physical Meaning:
    Performs FFT in temporal dimension to obtain frequency
    spectrum for resonance analysis.

Args:
    envelope (np.ndarray): BVP env...

### _find_resonance_peaks(self, spectrum)
**Описание:** Find resonance peaks in the frequency spectrum.

Physical Meaning:
    Identifies local maxima in frequency spectrum that
    correspond to resonance frequencies.

Args:
    spectrum (np.ndarray): Fre...

### _compute_quality_factors(self, resonance_peaks, spectrum)
**Описание:** Compute quality factors for resonance peaks.

Physical Meaning:
    Calculates Q factors from peak width at half maximum,
    indicating resonator quality and energy storage.

Args:
    resonance_peak...

### _analyze_impedance_characteristics(self, envelope)
**Описание:** Analyze impedance characteristics of the tail.

Physical Meaning:
    Computes admittance and analyzes frequency dependence
    to characterize impedance behavior.

Args:
    envelope (np.ndarray): BV...

### _compute_admittance(self, envelope)
**Описание:** Compute admittance from envelope.

Physical Meaning:
    Calculates admittance using full transmission line theory
    with proper impedance matching and frequency dependence.

Mathematical Foundation...

### _analyze_frequency_dependence(self, admittance) -> float
**Описание:** Analyze frequency dependence of admittance.

Physical Meaning:
    Computes variance across frequency domain to quantify
    frequency dependence of impedance.

Args:
    admittance (np.ndarray): Admi...

### _validate_resonatorness(self, resonance_peaks, quality_factors) -> bool
**Описание:** Validate that the tail exhibits resonator-like behavior.

Physical Meaning:
    Checks for sufficient number of resonances and adequate
    quality factors to confirm resonator behavior.

Args:
    re...

## ./bhlff/core/bvp/topological_charge_analyzer.py
Methods: 19

### __init__(self, domain, config, constants)
**Описание:** Initialize topological charge analyzer.

Physical Meaning:
    Sets up the topological charge analyzer with the computational domain
    and configuration parameters for analyzing topological defects
...

### _setup_analysis_parameters(self)
**Описание:** Setup analysis parameters.

Physical Meaning:
    Initializes parameters for topological charge analysis based on
    the domain properties and configuration.

### compute_topological_charge(self, field)
**Описание:** Compute topological charge using block processing and vectorization.

Physical Meaning:
    Computes the topological charge using block processing to handle
    large domains efficiently with CUDA acc...

### _compute_defect_charge(self, phase, defect_location) -> float
**Описание:** Compute topological charge around a defect with CUDA optimization.

Physical Meaning:
    Computes the winding number around a topological defect
    using the circulation of phase gradients with CUDA...

### _compute_defect_charge_cuda(self, phase, defect_location) -> float
**Описание:** Compute topological charge using CUDA acceleration.

Physical Meaning:
    CUDA-accelerated computation of topological charge
    using vectorized operations on GPU.

### _compute_defect_charge_cpu(self, phase, defect_location) -> float
**Описание:** Compute topological charge using CPU with vectorized operations.

Physical Meaning:
    CPU-optimized computation of topological charge
    using vectorized NumPy operations.

### _determine_optimal_block_size(self, field_shape)
**Описание:** Determine optimal block size for memory-efficient processing.

Physical Meaning:
    Calculates block size that fits within memory constraints
    while maintaining sufficient resolution for topologic...

### _generate_overlapping_blocks(self, field_shape, block_size)
**Описание:** Generate overlapping blocks for processing large fields.

Physical Meaning:
    Creates overlapping blocks to ensure no defects are missed
    at block boundaries, with vectorized operations.

Args:
 ...

### _find_defects_vectorized(self, phase_block)
**Описание:** Find topological defects using vectorized operations.

Physical Meaning:
    Identifies topological defects using vectorized gradient
    computation and threshold analysis for maximum performance.

A...

### _find_defects_cuda_vectorized(self, phase_block)
**Описание:** Find defects using CUDA-accelerated vectorized operations.

Physical Meaning:
    CUDA-accelerated identification of topological defects
    using vectorized gradient computation on GPU.

### _find_defects_cpu_vectorized(self, phase_block)
**Описание:** Find defects using CPU vectorized operations.

Physical Meaning:
    CPU-optimized identification of topological defects
    using vectorized NumPy operations.

### _extract_defects_vectorized(self, high_grad_mask)
**Описание:** Extract defect locations using vectorized operations.

Physical Meaning:
    Identifies connected components of high gradient regions
    using vectorized morphological operations.

Args:
    high_gra...

### _compute_defect_charge_vectorized(self, phase, defect_location) -> float
**Описание:** Compute topological charge using vectorized operations.

Physical Meaning:
    Computes the winding number around a topological defect
    using vectorized operations for maximum performance.

Args:
 ...

### _compute_charge_cuda_vectorized(self, neighborhood) -> float
**Описание:** Compute charge using CUDA vectorized operations.

Physical Meaning:
    CUDA-accelerated computation of topological charge
    using vectorized operations on GPU.

### _compute_charge_cpu_vectorized(self, neighborhood) -> float
**Описание:** Compute charge using CPU vectorized operations.

Physical Meaning:
    CPU-optimized computation of topological charge
    using vectorized NumPy operations.

### _compute_charge_stability(self, charges, locations) -> float
**Описание:** Compute stability of topological charges.

Physical Meaning:
    Computes a measure of how stable the topological charges are
    based on their magnitudes and spatial distribution.

Args:
    charges...

### _analyze_defects(self, phase, charge_locations, charges)
**Описание:** Analyze topological defects in detail.

Physical Meaning:
    Performs detailed analysis of topological defects including
    their types, strengths, and interactions.

Args:
    phase (np.ndarray): P...

### analyze_phase_structure(self, field)
**Описание:** Analyze phase structure of the field.

Physical Meaning:
    Analyzes the phase structure of the BVP field to understand
    the topological characteristics and phase coherence.

Args:
    field (np.n...

### get_analysis_parameters(self)
**Описание:** Get current analysis parameters.

Physical Meaning:
    Returns the current parameters used for topological charge analysis.

Returns:
    Dict[str, Any]: Analysis parameters.

## ./bhlff/core/bvp/topological_defect_analyzer.py
Methods: 9

### __init__(self, domain, config, constants)
**Описание:** Initialize topological defect analyzer.

Physical Meaning:
    Sets up the defect analyzer with the computational domain
    and configuration parameters for analyzing topological defects
    in the B...

### _setup_analysis_parameters(self)
**Описание:** Setup analysis parameters.

Physical Meaning:
    Initializes parameters for topological defect analysis based on
    the domain properties and configuration.

### find_topological_defects(self, phase)
**Описание:** Find topological defects in the phase field with CUDA optimization.

Physical Meaning:
    Identifies points where the phase field has singularities
    or rapid changes, indicating topological defect...
**Декораторы:** <ast.Call object at 0x753e6f8c7850>

### _find_topological_defects_cuda(self, phase)
**Описание:** Find topological defects using CUDA acceleration.

Physical Meaning:
    CUDA-accelerated identification of topological defects
    using vectorized gradient computation on GPU.

### _find_topological_defects_cpu(self, phase)
**Описание:** Find topological defects using CPU with vectorized operations.

Physical Meaning:
    CPU-optimized identification of topological defects
    using vectorized NumPy operations.

### analyze_defect_types(self, phase, defect_locations)
**Описание:** Analyze types of topological defects.

Physical Meaning:
    Classifies topological defects based on their local
    phase structure and gradient patterns.

Mathematical Foundation:
    Defect types a...

### analyze_defect_interactions(self, defect_locations, defect_charges)
**Описание:** Analyze interactions between topological defects.

Physical Meaning:
    Computes interaction strengths between topological defects
    based on their charges and spatial separation.

Mathematical Fou...

### _extract_neighborhood(self, field, center, radius)
**Описание:** Extract neighborhood around a point.

Physical Meaning:
    Extracts a small neighborhood around a point for
    local analysis of field properties.

Args:
    field (np.ndarray): Field to extract fro...

### get_analysis_parameters(self)
**Описание:** Get current analysis parameters.

Physical Meaning:
    Returns the current parameters used for topological defect analysis.

Returns:
    Dict[str, Any]: Analysis parameters.

## ./bhlff/core/bvp/transition_zone_postulate.py
Methods: 10

### __init__(self, domain, constants)
**Описание:** Initialize transition zone postulate.

Physical Meaning:
    Sets up the postulate for analyzing nonlinear interface
    behavior in the transition zone.

Args:
    domain (Domain): Computational doma...

### apply(self, envelope)
**Описание:** Apply transition zone postulate.

Physical Meaning:
    Verifies that the transition zone exhibits nonlinear
    admittance characteristics and generates effective
    EM/weak currents from the envelo...

### _compute_nonlinear_admittance(self, envelope)
**Описание:** Compute nonlinear admittance Y_tr(ω,|A|).

Physical Meaning:
    Calculates admittance that depends on both frequency
    and amplitude, representing nonlinear interface behavior.

Mathematical Founda...

### _compute_base_admittance(self, envelope)
**Описание:** Compute base linear admittance.

Physical Meaning:
    Calculates linear component of admittance from
    envelope gradient and amplitude using full transmission line theory.

Mathematical Foundation:...

### _generate_effective_currents(self, envelope)
**Описание:** Generate effective EM/weak currents J(ω) from envelope.

Physical Meaning:
    Computes electromagnetic and weak currents as functionals
    of the envelope amplitude and phase.

Mathematical Foundati...

### _analyze_transition_zone_properties(self, envelope)
**Описание:** Analyze properties of the transition zone.

Physical Meaning:
    Computes transition zone boundaries, nonlinearity strength,
    and current generation efficiency.

Args:
    envelope (np.ndarray): B...

### _compute_transition_boundaries(self, amplitude)
**Описание:** Compute boundaries of the transition zone.

Physical Meaning:
    Identifies inner and outer boundaries of transition zone
    based on amplitude gradient thresholds.

Args:
    amplitude (np.ndarray)...

### _compute_nonlinearity_strength(self, amplitude) -> float
**Описание:** Compute strength of nonlinearity in transition zone.

Physical Meaning:
    Quantifies nonlinearity strength based on amplitude
    variation relative to mean amplitude.

Args:
    amplitude (np.ndarr...

### _compute_current_efficiency(self, amplitude) -> float
**Описание:** Compute efficiency of current generation.

Physical Meaning:
    Calculates efficiency of current generation based on
    amplitude gradients relative to mean amplitude.

Args:
    amplitude (np.ndarr...

### _validate_nonlinear_interface(self, nonlinear_admittance, effective_currents) -> bool
**Описание:** Validate that the transition zone is a nonlinear interface.

Physical Meaning:
    Checks nonlinearity strength and current generation
    to confirm nonlinear interface behavior.

Args:
    nonlinear...

## ./bhlff/core/bvp/u1_phase_structure/coherence_analysis.py
Methods: 5

### __init__(self, domain)
**Описание:** Initialize coherence analysis.

Physical Meaning:
    Sets up the analyzer with domain information
    for phase coherence analysis.

Args:
    domain (Domain): Computational domain for analysis.

### analyze_phase_coherence(self, envelope)
**Описание:** Analyze phase coherence across the field.

Physical Meaning:
    Computes phase coherence measures to verify that
    phase relationships are maintained across spatial scales.

Args:
    envelope (np....

### _compute_local_phase_coherence(self, phase)
**Описание:** Compute local phase coherence.

Physical Meaning:
    Calculates phase coherence in local neighborhoods
    to measure phase consistency.

Args:
    phase (np.ndarray): Phase field.

Returns:
    np.n...

### _compute_global_phase_coherence(self, phase) -> float
**Описание:** Compute global phase coherence.

Physical Meaning:
    Calculates overall phase coherence across the
    entire field domain.

Args:
    phase (np.ndarray): Phase field.

Returns:
    float: Global co...

### __repr__(self) -> str
**Описание:** String representation of coherence analysis.

## ./bhlff/core/bvp/u1_phase_structure/phase_analysis.py
Methods: 6

### __init__(self, domain)
**Описание:** Initialize phase analysis.

Physical Meaning:
    Sets up the analyzer with domain information
    for phase structure analysis.

Args:
    domain (Domain): Computational domain for analysis.

### analyze_phase_structure(self, envelope)
**Описание:** Analyze U(1)³ phase structure of the field.

Physical Meaning:
    Extracts and analyzes the three phase components Θ_a (a=1..3)
    from the complex envelope field.

Mathematical Foundation:
    Enve...

### _decompose_phase_components(self, total_phase)
**Описание:** Decompose total phase into three U(1) components.

Physical Meaning:
    Separates total phase into three independent U(1)
    phase components using spatial frequency analysis.

Args:
    total_phase...

### _create_frequency_mask(self, shape, component_idx)
**Описание:** Create frequency mask for phase component extraction.

Physical Meaning:
    Creates frequency domain mask to separate different
    phase components based on spatial frequencies.

Args:
    shape (tu...

### _compute_phase_statistics(self, phase_components)
**Описание:** Compute statistics for phase components.

Physical Meaning:
    Calculates statistical properties of each phase
    component to characterize U(1)³ structure.

Args:
    phase_components (List[np.ndar...

### __repr__(self) -> str
**Описание:** String representation of phase analysis.

## ./bhlff/core/bvp/u1_phase_structure/u1_phase_structure_postulate.py
Methods: 5

### __init__(self, domain, constants)
**Описание:** Initialize U(1)³ phase structure postulate.

Physical Meaning:
    Sets up the postulate with domain and constants for
    analyzing U(1)³ phase structure properties.

Args:
    domain (Domain): Compu...

### apply(self, envelope)
**Описание:** Apply U(1)³ phase structure postulate.

Physical Meaning:
    Verifies that BVP field exhibits U(1)³ phase structure
    with proper phase coherence and phase vector properties.

Mathematical Foundati...

### _check_u1_properties(self, phase_structure, phase_coherence)
**Описание:** Check U(1)³ properties of the field.

Physical Meaning:
    Verifies that field exhibits proper U(1)³ phase
    structure with adequate coherence.

Args:
    phase_structure (Dict[str, Any]): Phase st...

### _validate_u1_phase_structure(self, u1_properties) -> bool
**Описание:** Validate U(1)³ phase structure postulate.

Physical Meaning:
    Checks that field exhibits proper U(1)³ phase
    structure for BVP framework validity.

Args:
    u1_properties (Dict[str, Any]): U(1)...

### __repr__(self) -> str
**Описание:** String representation of U(1)³ phase structure postulate.

## ./bhlff/core/domain/domain.py
Methods: 9

### __post_init__(self)
**Описание:** Initialize derived attributes after object creation.

Physical Meaning:
    Computes grid spacing and coordinate arrays for 7D space-time
    M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ based on domain size and resolution p...

### get_differentials(self)
**Описание:** Get differential elements for 7D space-time.

Physical Meaning:
    Returns the differential elements dx, dphi, dt for
    the 7D space-time structure M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Returns:
    Dict[str, flo...

### _setup_coordinates(self)
**Описание:** Setup coordinate arrays for 7D space-time domain.

Physical Meaning:
    Creates coordinate arrays for grid points in the 7D space-time
    M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ, used for source placement and field vi...

### get_wave_numbers(self) -> dict
**Описание:** Get wave number arrays for 7D FFT operations.

Physical Meaning:
    Computes the wave number arrays for 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ
    for FFT operations, where wave numbers are defined for e...

### get_center_index(self) -> dict
**Описание:** Get the index of the domain center for 7D space-time.

Physical Meaning:
    Returns the grid indices corresponding to the center of each
    dimension type in the 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ,
...

### get_volume(self) -> dict
**Описание:** Get the domain volume for 7D space-time.

Physical Meaning:
    Computes the total volume of each dimension type in the 7D space-time
    M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Returns:
    dict: Volume for each dime...

### get_grid_spacing(self) -> dict
**Описание:** Get the grid spacing for 7D space-time.

Physical Meaning:
    Returns the uniform grid spacing for each dimension type
    in the 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Returns:
    dict: Grid spacing ...

### get_coordinates(self, dim)
**Описание:** Get coordinates for specific dimension.

Physical Meaning:
    Returns coordinate array for the specified dimension in the 7D space-time.

Args:
    dim (int): Dimension index:
        - 0, 1, 2: spat...

### __repr__(self) -> str
**Описание:** String representation of the 7D domain.

## ./bhlff/core/domain/domain_7d.py
Methods: 18

### __init__(self, spatial_config, phase_config, temporal_config)
**Описание:** Initialize 7D space-time domain.

Physical Meaning:
    Sets up the complete 7D space-time structure with spatial,
    phase, and temporal coordinates for BVP calculations.

Args:
    spatial_config (...

### _setup_spatial_coordinates(self)
**Описание:** Setup spatial coordinates ℝ³ₓ.

### _setup_phase_coordinates(self)
**Описание:** Setup phase coordinates 𝕋³_φ.

### _setup_temporal_coordinates(self)
**Описание:** Setup temporal coordinate ℝₜ.

### _setup_metric_tensor(self)
**Описание:** Setup metric tensor for 7D space-time.

### _create_full_coordinate_grids(self)
**Описание:** Create full 7D coordinate grids.

### get_spatial_coordinates(self)
**Описание:** Get spatial coordinates ℝ³ₓ.

Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: (X, Y, Z) coordinate grids.

### get_phase_coordinates(self)
**Описание:** Get phase coordinates 𝕋³_φ.

Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: (PHI_1, PHI_2, PHI_3) coordinate grids.

### get_temporal_coordinates(self)
**Описание:** Get temporal coordinates ℝₜ.

Returns:
    np.ndarray: Temporal coordinate array.

### get_full_7d_coordinates(self)
**Описание:** Get full 7D coordinate array.

Returns:
    np.ndarray: Full 7D coordinate array with shape (7, N_x, N_y, N_z, N_phi_1, N_phi_2, N_phi_3).

### get_metric_tensor(self)
**Описание:** Get 7D metric tensor.

Returns:
    np.ndarray: 7D metric tensor with shape (7, 7).

### get_spatial_shape(self)
**Описание:** Get spatial grid shape.

Returns:
    Tuple[int, int, int]: (N_x, N_y, N_z) spatial grid dimensions.

### get_phase_shape(self)
**Описание:** Get phase grid shape.

Returns:
    Tuple[int, int, int]: (N_phi_1, N_phi_2, N_phi_3) phase grid dimensions.

### get_full_7d_shape(self)
**Описание:** Get full 7D grid shape.

Returns:
    Tuple[int, int, int, int, int, int]: Full 7D grid dimensions.

### get_differentials(self)
**Описание:** Get coordinate differentials.

Returns:
    Dict[str, float]: Dictionary of coordinate differentials.

### compute_7d_volume_element(self)
**Описание:** Compute 7D volume element.

Physical Meaning:
    Computes the volume element dV₇ for integration over the 7D space-time.

Mathematical Foundation:
    dV₇ = dx dy dz dφ₁ dφ₂ dφ₃ dt

Returns:
    np.n...

### compute_7d_distance(self, point1, point2) -> float
**Описание:** Compute 7D distance between two points.

Physical Meaning:
    Computes the proper distance between two points in 7D space-time
    using the metric tensor.

Mathematical Foundation:
    ds² = g_μν dx...

### __repr__(self) -> str
**Описание:** String representation of 7D domain.

## ./bhlff/core/domain/domain_7d_bvp.py
Methods: 21

### __post_init__(self)
**Описание:** Initialize computed properties.

### _validate_parameters(self)
**Описание:** Validate domain parameters.

### _compute_derived_properties(self)
**Описание:** Compute derived properties.

### spatial_shape(self)
**Описание:** Spatial domain shape (N_x, N_y, N_z).
**Декораторы:** property

### phase_shape(self)
**Описание:** Phase domain shape (N_φ₁, N_φ₂, N_φ₃).
**Декораторы:** property

### temporal_shape(self)
**Описание:** Temporal domain shape (N_t,).
**Декораторы:** property

### shape(self)
**Описание:** Full 7D domain shape (N_x, N_y, N_z, N_φ₁, N_φ₂, N_φ₃, N_t).
**Декораторы:** property

### size(self) -> int
**Описание:** Get total number of grid points.
**Декораторы:** property

### get_grid_spacing(self)
**Описание:** Get grid spacing for all dimensions.

Physical Meaning:
    Returns the grid spacing for spatial, phase, and temporal
    dimensions, representing the resolution of the computational grid.

Returns:
 ...

### get_total_volume(self) -> float
**Описание:** Get total volume of the 7D domain.

Physical Meaning:
    Returns the total volume of the 7D space-time domain,
    representing the total computational space available.

Returns:
    float: Total vol...

### spatial_coordinates(self)
**Описание:** Spatial coordinate arrays (x, y, z).
**Декораторы:** property

### phase_coordinates(self)
**Описание:** Phase coordinate arrays (φ₁, φ₂, φ₃).
**Декораторы:** property

### temporal_coordinates(self)
**Описание:** Temporal coordinate array (t).
**Декораторы:** property

### spatial_wave_vectors(self)
**Описание:** Spatial wave vector arrays (k_x, k_y, k_z).
**Декораторы:** property

### phase_wave_vectors(self)
**Описание:** Phase wave vector arrays (k_φ₁, k_φ₂, k_φ₃).
**Декораторы:** property

### temporal_wave_vectors(self)
**Описание:** Temporal wave vector array (k_t).
**Декораторы:** property

### get_coordinate_meshgrids(self)
**Описание:** Get coordinate meshgrids for all 7 dimensions.

Physical Meaning:
    Returns meshgrids for all 7 coordinates, enabling
    evaluation of fields at specific points in M₇.

Returns:
    Tuple[np.ndarra...

### get_wave_vector_meshgrids(self)
**Описание:** Get wave vector meshgrids for all 7 dimensions.

Physical Meaning:
    Returns meshgrids for all 7 wave vectors, enabling
    spectral operations in frequency space.

Returns:
    Tuple[np.ndarray, .....

### compute_wave_vector_magnitude(self)
**Описание:** Compute 7D wave vector magnitude |k|.

Physical Meaning:
    Computes the magnitude of the 7D wave vector, representing
    the spatial frequency of field components in M₇.

Mathematical Foundation:
 ...

### get_volume_element(self, coordinate_type) -> float
**Описание:** Get volume element for specified coordinate type.

Physical Meaning:
    Returns the volume element for integration over the specified
    coordinate subspace in M₇.

Args:
    coordinate_type (str): ...

### __repr__(self) -> str
**Описание:** String representation of domain.

## ./bhlff/core/domain/field.py
Methods: 14

### __post_init__(self)
**Описание:** Initialize field after object creation.

Physical Meaning:
    Validates field data and sets up metadata for
    phase field operations.

### _validate_field(self)
**Описание:** Validate field data.

Physical Meaning:
    Ensures field data has correct shape and properties
    for the computational domain.

Raises:
    ValueError: If field data is invalid.

### get_amplitude(self)
**Описание:** Get field amplitude |a(x)|.

Physical Meaning:
    Computes the amplitude of the phase field, representing
    the magnitude of the field at each spatial point.

Mathematical Foundation:
    Amplitude...

### get_phase(self)
**Описание:** Get field phase arg(a(x)).

Physical Meaning:
    Computes the phase of the phase field, representing
    the phase angle at each spatial point.

Mathematical Foundation:
    Phase is arg(a(x)) = arct...

### get_gradient(self)
**Описание:** Get field gradient ∇a(x).

Physical Meaning:
    Computes the spatial gradient of the phase field,
    representing the rate of change of the field in space.

Mathematical Foundation:
    Gradient is ...

### get_laplacian(self)
**Описание:** Get field Laplacian Δa(x).

Physical Meaning:
    Computes the Laplacian of the phase field, representing
    the second-order spatial derivatives.

Mathematical Foundation:
    Laplacian is Δa = ∂²a/...

### get_energy_density(self)
**Описание:** Get field energy density.

Physical Meaning:
    Computes the local energy density of the phase field,
    representing the energy content per unit volume.

Mathematical Foundation:
    Energy density...

### get_total_energy(self) -> float
**Описание:** Get total field energy.

Physical Meaning:
    Computes the total energy of the phase field configuration,
    representing the integrated energy content over the domain.

Mathematical Foundation:
   ...

### fft(self)
**Описание:** Compute FFT of the field.

Physical Meaning:
    Transforms the field to frequency space for spectral
    analysis and operations.

Mathematical Foundation:
    FFT transforms a(x) → â(k) in frequency...

### ifft(self, spectral_data)
**Описание:** Compute inverse FFT of spectral data.

Physical Meaning:
    Transforms spectral data back to real space.

Mathematical Foundation:
    IFFT transforms â(k) → a(x) in real space.

Args:
    spectral_d...

### copy(self)
**Описание:** Create a copy of the field.

Physical Meaning:
    Creates an independent copy of the field for
    manipulation without affecting the original.

Returns:
    Field: Copy of the field.

### set_metadata(self, key, value)
**Описание:** Set field metadata.

Physical Meaning:
    Stores additional information about the field
    for analysis and visualization.

Args:
    key (str): Metadata key.
    value (Any): Metadata value.

### get_metadata(self, key, default) -> Any
**Описание:** Get field metadata.

Physical Meaning:
    Retrieves additional information about the field.

Args:
    key (str): Metadata key.
    default (Any): Default value if key not found.

Returns:
    Any: M...

### __repr__(self) -> str
**Описание:** String representation of the field.

## ./bhlff/core/domain/parameters.py
Methods: 9

### __post_init__(self)
**Описание:** Validate parameters after object creation.

Physical Meaning:
    Ensures all parameters are within physically meaningful ranges
    and compatible with the mathematical framework.

Raises:
    ValueE...

### _validate_parameters(self)
**Описание:** Validate all parameters for physical consistency.

Physical Meaning:
    Checks that parameters satisfy the constraints required for
    well-posedness of the fractional Riesz equation and numerical
 ...

### get_spectral_coefficients(self, k_magnitude)
**Описание:** Compute spectral coefficients for the fractional operator.

Physical Meaning:
    Computes the spectral representation D(k) = μ|k|^(2β) + λ
    of the fractional Riesz operator for FFT-based solution....

### get_time_coefficients(self, k_magnitude)
**Описание:** Compute time evolution coefficients.

Physical Meaning:
    Computes the coefficients α_k = ν|k|^(2β) + λ for time
    evolution in the spectral domain.

Mathematical Foundation:
    Time evolution co...

### to_dict(self)
**Описание:** Convert parameters to dictionary.

Physical Meaning:
    Provides a dictionary representation of parameters for
    serialization and configuration management.

Returns:
    Dict[str, Any]: Dictionary...

### from_dict(cls, params)
**Описание:** Create parameters from dictionary.

Physical Meaning:
    Constructs a Parameters object from a dictionary representation,
    useful for loading configurations from files.

Args:
    params (Dict[str...
**Декораторы:** classmethod

### default_cosmic(cls)
**Описание:** Create default parameters for cosmic (homogeneous) regime.

Physical Meaning:
    Provides default parameters for simulations in homogeneous
    "cosmic" media where power law tails A(r) ∝ r^(2β-3) ar...
**Декораторы:** classmethod

### default_matter(cls)
**Описание:** Create default parameters for matter (inhomogeneous) regime.

Physical Meaning:
    Provides default parameters for simulations in inhomogeneous
    matter where resonator structures and boundaries ar...
**Декораторы:** classmethod

### __repr__(self) -> str
**Описание:** String representation of parameters.

## ./bhlff/core/domain/parameters_7d_bvp.py
Methods: 10

### __post_init__(self)
**Описание:** Initialize and validate parameters.

### _validate_parameters(self)
**Описание:** Validate parameter values.

### compute_stiffness(self, amplitude)
**Описание:** Compute nonlinear stiffness κ(|a|) = κ₀ + κ₂|a|².

Physical Meaning:
    Computes the nonlinear stiffness coefficient that depends
    on the field amplitude, representing the local "rigidity"
    of ...

### compute_susceptibility(self, amplitude)
**Описание:** Compute nonlinear susceptibility χ(|a|) = χ' + iχ''(|a|).

Physical Meaning:
    Computes the complex susceptibility that depends on field amplitude,
    representing the local response of the phase f...

### compute_stiffness_derivative(self, amplitude)
**Описание:** Compute derivative of stiffness with respect to amplitude.

Physical Meaning:
    Computes dκ/d|a| = 2κ₂|a| for Newton-Raphson iterations.

Mathematical Foundation:
    dκ/d|a| = 2κ₂|a|

Args:
    amp...

### compute_susceptibility_derivative(self, amplitude)
**Описание:** Compute derivative of susceptibility with respect to amplitude.

Physical Meaning:
    Computes dχ/d|a| = i2χ''₂|a| for Newton-Raphson iterations.

Mathematical Foundation:
    dχ/d|a| = i2χ''₂|a|

Ar...

### get_fractional_laplacian_coefficients(self)
**Описание:** Get coefficients for fractional Laplacian L_β = μ(-Δ)^β + λ.

Physical Meaning:
    Returns the coefficients for the fractional Laplacian operator
    used in the linearized version of the BVP equatio...

### get_numerical_parameters(self)
**Описание:** Get numerical parameters for solvers.

Returns:
    Dict[str, Any]: Numerical parameters.

### get_physical_parameters(self)
**Описание:** Get physical parameters for the BVP equation.

Returns:
    Dict[str, float]: Physical parameters.

### __repr__(self) -> str
**Описание:** String representation of parameters.

## ./bhlff/core/fft/advanced/fft_adaptive.py
Methods: 4

### __init__(self, domain, parameters)
**Описание:** Initialize FFT adaptive methods.

### solve_adaptive(self, source)
**Описание:** Solve using adaptive methods.

Physical Meaning:
    Solves the fractional Laplacian equation using adaptive
    methods for improved convergence and accuracy.

Args:
    source (np.ndarray): Source t...

### setup_adaptive_methods(self)
**Описание:** Setup adaptive methods.

### _get_spectral_coefficients(self)
**Описание:** Get spectral coefficients for adaptive solving.

## ./bhlff/core/fft/advanced/fft_advanced_core.py
Methods: 11

### __init__(self, domain, parameters)
**Описание:** Initialize advanced 7D FFT solver.

Physical Meaning:
    Sets up the advanced FFT solver with all necessary components
    for optimized, adaptive, and comprehensive solving capabilities.

Args:
    ...

### solve_optimized(self, source)
**Описание:** Solve using optimization techniques.

Physical Meaning:
    Solves the fractional Laplacian equation using optimization
    techniques for improved efficiency and accuracy.

Mathematical Foundation:
 ...

### solve_adaptive(self, source)
**Описание:** Solve using adaptive methods.

Physical Meaning:
    Solves the fractional Laplacian equation using adaptive
    methods for improved convergence and accuracy.

Mathematical Foundation:
    Uses adapt...

### solve_with_analysis(self, source)
**Описание:** Solve with comprehensive analysis.

Physical Meaning:
    Solves the fractional Laplacian equation and provides
    comprehensive analysis of the solution and solving process.

Mathematical Foundation...

### solve_time_evolution(self, initial_condition, time_steps)
**Описание:** Solve time evolution of the system.

Physical Meaning:
    Solves the time evolution of the fractional Laplacian equation
    starting from an initial condition.

Mathematical Foundation:
    Uses tim...

### validate_solution_comprehensive(self, solution, source)
**Описание:** Perform comprehensive solution validation.

Physical Meaning:
    Performs comprehensive validation of the solution including
    accuracy, stability, and physical reasonableness checks.

Mathematical...

### _setup_advanced_components(self)
**Описание:** Setup advanced solver components.

Physical Meaning:
    Initializes all advanced components including spectral coefficients,
    FFT plans, optimization settings, and adaptive methods.

### _setup_spectral_coefficients(self)
**Описание:** Setup spectral coefficients for advanced solving.

### _setup_fft_plan(self)
**Описание:** Setup FFT plan for advanced solving.

### _setup_optimization(self)
**Описание:** Setup optimization components.

### _setup_adaptive_methods(self)
**Описание:** Setup adaptive methods.

## ./bhlff/core/fft/advanced/fft_analysis.py
Methods: 5

### __init__(self, domain, parameters)
**Описание:** Initialize FFT analysis.

### solve_with_analysis(self, source)
**Описание:** Solve with comprehensive analysis.

Physical Meaning:
    Solves the fractional Laplacian equation and provides
    comprehensive analysis of the solution and solving process.

Args:
    source (np.nd...

### _analyze_solution(self, solution, source)
**Описание:** Analyze solution quality.

### _apply_operator(self, field)
**Описание:** Apply the fractional Laplacian operator.

### _get_spectral_coefficients(self)
**Описание:** Get spectral coefficients for analysis.

## ./bhlff/core/fft/advanced/fft_optimization.py
Methods: 4

### __init__(self, domain, parameters)
**Описание:** Initialize FFT optimization.

### solve_optimized(self, source)
**Описание:** Solve using optimization techniques.

Physical Meaning:
    Solves the fractional Laplacian equation using optimization
    techniques for improved efficiency and accuracy.

Args:
    source (np.ndarr...

### setup_optimization(self)
**Описание:** Setup optimization components.

### _get_spectral_coefficients(self)
**Описание:** Get spectral coefficients for optimization.

## ./bhlff/core/fft/bvp_advanced/bvp_adaptive.py
Methods: 6

### __init__(self, domain, parameters, derivatives)
**Описание:** Initialize BVP adaptive methods.

### solve_adaptive(self, solution, source)
**Описание:** Solve using adaptive methods.

Physical Meaning:
    Solves the BVP envelope equation using adaptive methods
    for improved convergence and accuracy.

Args:
    solution (np.ndarray): Initial soluti...

### _compute_jacobian(self, solution)
**Описание:** Compute Jacobian matrix.

### _solve_linear_system_adaptive(self, jacobian, residual)
**Описание:** Solve linear system with adaptive methods.

### _compute_adaptive_step_size(self, solution, update, residual) -> float
**Описание:** Compute adaptive step size.

### _apply_operator(self, field)
**Описание:** Apply the BVP operator.

## ./bhlff/core/fft/bvp_advanced/bvp_advanced_core.py
Methods: 7

### __init__(self, domain, parameters, derivatives)
**Описание:** Initialize advanced BVP solver core.

Physical Meaning:
    Sets up the advanced BVP solver with all necessary components
    for optimized, preconditioned, and adaptive solving capabilities.

Args:
 ...

### solve_with_preconditioning(self, solution, source)
**Описание:** Solve with preconditioning.

Physical Meaning:
    Solves the BVP envelope equation using preconditioning techniques
    for improved convergence and numerical stability.

Mathematical Foundation:
   ...

### solve_with_optimization(self, solution, source)
**Описание:** Solve with optimization.

Physical Meaning:
    Solves the BVP envelope equation using optimization techniques
    for improved efficiency and accuracy.

Mathematical Foundation:
    Uses optimized it...

### solve_adaptive(self, solution, source)
**Описание:** Solve using adaptive methods.

Physical Meaning:
    Solves the BVP envelope equation using adaptive methods
    for improved convergence and accuracy.

Mathematical Foundation:
    Uses adaptive iter...

### _compute_residual_basic(self, solution, source)
**Описание:** Compute basic residual.

Physical Meaning:
    Computes the residual of the BVP envelope equation
    for basic solving methods.

Args:
    solution (np.ndarray): Current solution field.
    source (n...

### _compute_jacobian_basic(self, solution)
**Описание:** Compute basic Jacobian.

Physical Meaning:
    Computes the Jacobian matrix of the BVP envelope equation
    for basic solving methods.

Args:
    solution (np.ndarray): Current solution field.

Retur...

### _apply_operator(self, field)
**Описание:** Apply the BVP operator.

Physical Meaning:
    Applies the BVP envelope equation operator to a field.

Args:
    field (np.ndarray): Field to apply operator to.

Returns:
    np.ndarray: Result of ope...

## ./bhlff/core/fft/bvp_advanced/bvp_optimization.py
Methods: 6

### __init__(self, domain, parameters, derivatives)
**Описание:** Initialize BVP optimization.

### solve_with_optimization(self, solution, source)
**Описание:** Solve with optimization.

Physical Meaning:
    Solves the BVP envelope equation using optimization techniques
    for improved efficiency and accuracy.

Args:
    solution (np.ndarray): Initial solut...

### _compute_jacobian(self, solution)
**Описание:** Compute Jacobian matrix.

### _solve_linear_system_optimized(self, jacobian, residual)
**Описание:** Solve linear system with optimization.

### _compute_optimal_step_size(self, solution, update, residual) -> float
**Описание:** Compute optimal step size.

### _apply_operator(self, field)
**Описание:** Apply the BVP operator.

## ./bhlff/core/fft/bvp_advanced/bvp_preconditioning.py
Methods: 4

### __init__(self, domain, parameters, derivatives)
**Описание:** Initialize BVP preconditioning.

### solve_with_preconditioning(self, solution, source)
**Описание:** Solve with preconditioning.

Physical Meaning:
    Solves the BVP envelope equation using preconditioning techniques
    for improved convergence and numerical stability.

Args:
    solution (np.ndarr...

### _compute_preconditioner(self, solution)
**Описание:** Compute preconditioner matrix.

### _apply_operator(self, field)
**Описание:** Apply the BVP operator.

## ./bhlff/core/fft/bvp_basic/bvp_basic_core.py
Methods: 10

### __init__(self, domain, parameters, derivatives)
**Описание:** Initialize comprehensive BVP solver core.

Physical Meaning:
    Sets up the comprehensive BVP solver with all necessary components
    for full theoretical solving capabilities according to the 7D
  ...

### compute_residual(self, solution, source)
**Описание:** Compute residual of the BVP equation.

Physical Meaning:
    Computes the residual of the BVP envelope equation
    to measure how well the current solution satisfies the equation.

Mathematical Found...

### compute_jacobian(self, solution)
**Описание:** Compute Jacobian matrix of the BVP equation.

Physical Meaning:
    Computes the Jacobian matrix for the BVP envelope equation
    to enable Newton-Raphson iteration.

Mathematical Foundation:
    Jac...

### solve_linear_system(self, jacobian, residual)
**Описание:** Solve linear system for Newton-Raphson update.

Physical Meaning:
    Solves the linear system J·δa = -r for the Newton-Raphson update,
    where J is the Jacobian and r is the residual.

Mathematical...

### validate_solution(self, solution, source)
**Описание:** Validate solution quality.

Physical Meaning:
    Validates the quality of the solution by checking
    residual norms and other quality metrics.

Args:
    solution (np.ndarray): Solution field to va...

### solve_envelope_comprehensive(self, source)
**Описание:** Comprehensive envelope equation solution.

Physical Meaning:
    Solves the 7D envelope equation using full theoretical methods
    without simplifications or approximations, ensuring complete
    adh...

### _validate_theoretical_consistency(self, solution, source)
**Описание:** Validate theoretical consistency of solution.

Physical Meaning:
    Validates that the solution satisfies theoretical constraints
    of the 7D phase field theory, including energy conservation
    a...

### _compute_energy_balance(self, solution, source) -> float
**Описание:** Compute energy balance for theoretical validation.

Physical Meaning:
    Computes the energy balance to verify energy conservation
    according to the 7D phase field theory.

Args:
    solution (np....

### _check_causality(self, solution) -> bool
**Описание:** Check causality constraints.

Physical Meaning:
    Verifies that the solution satisfies causality constraints
    of the 7D phase field theory.

Args:
    solution (np.ndarray): Solution field.

Retu...

### _check_7d_structure(self, solution) -> bool
**Описание:** Check 7D structure preservation.

Physical Meaning:
    Verifies that the solution preserves the 7D phase field
    structure according to the theoretical framework.

Args:
    solution (np.ndarray): ...

## ./bhlff/core/fft/bvp_basic/bvp_jacobian.py
Methods: 5

### __init__(self, domain, parameters, derivatives)
**Описание:** Initialize BVP Jacobian computation.

### compute_jacobian(self, solution)
**Описание:** Compute Jacobian matrix of the BVP equation.

Physical Meaning:
    Computes the Jacobian matrix for the BVP envelope equation
    to enable Newton-Raphson iteration.

Args:
    solution (np.ndarray):...

### _compute_jacobian_row(self, solution, idx)
**Описание:** Compute Jacobian row for a given index.

### _compute_diagonal_jacobian_entry(self, solution, idx) -> float
**Описание:** Compute diagonal Jacobian entry.

### _compute_neighbor_jacobian_entries(self, solution, idx)
**Описание:** Compute neighbor Jacobian entries.

## ./bhlff/core/fft/bvp_basic/bvp_linear_solver.py
Methods: 2

### __init__(self, domain, parameters, derivatives)
**Описание:** Initialize BVP linear solver.

### solve_linear_system(self, jacobian, residual)
**Описание:** Solve linear system for Newton-Raphson update.

Physical Meaning:
    Solves the linear system J·δa = -r for the Newton-Raphson update,
    where J is the Jacobian and r is the residual.

Args:
    ja...

## ./bhlff/core/fft/bvp_basic/bvp_residual.py
Methods: 6

### __init__(self, domain, parameters, derivatives)
**Описание:** Initialize BVP residual computation.

### compute_residual(self, solution, source)
**Описание:** Compute residual of the BVP equation.

Physical Meaning:
    Computes the residual of the BVP envelope equation
    to measure how well the current solution satisfies the equation.

Args:
    solution...

### _compute_nonlinear_stiffness(self, solution)
**Описание:** Compute nonlinear stiffness.

### _compute_effective_susceptibility(self, solution)
**Описание:** Compute effective susceptibility.

### _compute_divergence_term(self, solution, stiffness)
**Описание:** Compute divergence term.

### _compute_susceptibility_term(self, solution, susceptibility)
**Описание:** Compute susceptibility term.

## ./bhlff/core/fft/bvp_solver_newton.py
Methods: 3

### __init__(self, core, parameters)
**Описание:** Initialize Newton-Raphson solver.

Physical Meaning:
    Sets up the Newton-Raphson solver with core functionality
    and parameters for iterative solution of the BVP equation.

Args:
    core (BVPSo...

### solve(self, source_field, initial_guess)
**Описание:** Solve full nonlinear equation using Newton-Raphson method.

Physical Meaning:
    Solves the complete nonlinear BVP equation using Newton-Raphson
    iteration to handle the nonlinear terms κ(|a|) and...

### _solve_linearized(self, source_field)
**Описание:** Solve linearized version using fractional Laplacian.

Physical Meaning:
    Solves the linearized version of the BVP equation:
    L_β a = μ(-Δ)^β a + λa = s(x,φ,t)
    which provides a good initial g...

## ./bhlff/core/fft/bvp_solver_validation.py
Methods: 4

### __init__(self, core, parameters)
**Описание:** Initialize BVP solver validation.

Physical Meaning:
    Sets up the validation methods with core functionality
    and parameters for solution validation.

Args:
    core (BVPSolverCore): Core BVP so...

### validate_solution(self, solution, source, tolerance, method)
**Описание:** Validate BVP solution.

Physical Meaning:
    Validates the solution by computing the residual and checking
    that it satisfies the BVP equation within the specified tolerance.

Args:
    solution (...

### check_energy_conservation(self, field, expected_energy, tolerance)
**Описание:** Check energy conservation for the field.

Physical Meaning:
    Verifies that the field satisfies energy conservation principles
    within the specified tolerance.

Mathematical Foundation:
    Compu...

### check_boundary_conditions(self, field, boundary_type)
**Описание:** Check boundary conditions for the field.

Physical Meaning:
    Verifies that the field satisfies the specified boundary conditions
    at the domain boundaries.

Mathematical Foundation:
    For peri...

## ./bhlff/core/fft/fft_backend_core.py
Methods: 14

### __init__(self, domain, plan_type, precision)
**Описание:** Initialize FFT backend.

Physical Meaning:
    Sets up the FFT backend with specified planning strategy
    and precision for efficient spectral operations.

Args:
    domain (Domain): Computational d...

### _setup_memory_pools(self)
**Описание:** Setup memory pools for efficient allocation.

Physical Meaning:
    Creates memory pools for efficient allocation and deallocation
    of temporary arrays during FFT operations.

### fft(self, real_data)
**Описание:** Compute forward FFT using unified spectral operations.

Physical Meaning:
    Transforms real space data to frequency space using Fast
    Fourier Transform with proper normalization.

Mathematical Fo...

### ifft(self, spectral_data)
**Описание:** Compute inverse FFT using unified spectral operations.

Physical Meaning:
    Transforms frequency space data back to real space using
    inverse Fast Fourier Transform with proper normalization.

Ma...

### fft_shift(self, spectral_data)
**Описание:** Shift FFT data to center zero frequency.

Physical Meaning:
    Shifts the FFT data so that zero frequency is at the center
    of the array, which is useful for visualization and analysis.

Mathemati...

### ifft_shift(self, spectral_data)
**Описание:** Inverse shift FFT data.

Physical Meaning:
    Applies inverse fftshift to restore the original frequency
    ordering of the FFT data.

Mathematical Foundation:
    Applies ifftshift to restore origi...

### get_frequency_arrays(self)
**Описание:** Get frequency arrays for the domain.

Physical Meaning:
    Returns the frequency arrays corresponding to the computational
    domain for spectral analysis.

Mathematical Foundation:
    Computes fre...

### get_wave_vector_magnitude(self)
**Описание:** Get wave vector magnitude for 7D BVP theory.

Physical Meaning:
    Computes the magnitude of the 7D wave vector |k| = √(kx² + ky² + kz² + kφ₁² + kφ₂² + kφ₃² + kt²)
    for spectral operations in 7D s...

### get_plan_type(self) -> str
**Описание:** Get the FFT plan type.

Physical Meaning:
    Returns the FFT planning strategy being used.

Returns:
    str: FFT plan type.

### get_precision(self) -> str
**Описание:** Get the numerical precision.

Physical Meaning:
    Returns the numerical precision being used for FFT operations.

Returns:
    str: Numerical precision.

### forward_transform(self, real_data)
**Описание:** Alias for fft() method.

Args:
    real_data (np.ndarray): Real space data.

Returns:
    np.ndarray: Frequency space data.

### inverse_transform(self, spectral_data)
**Описание:** Alias for ifft() method.

Args:
    spectral_data (np.ndarray): Frequency space data.

Returns:
    np.ndarray: Real space data.

### get_wave_vectors(self, dim)
**Описание:** Get wave vector for specific dimension.

Args:
    dim (int): Dimension index (0-6 for 7D).

Returns:
    np.ndarray: Wave vector for the specified dimension.

### __repr__(self) -> str
**Описание:** String representation of the FFT backend.

## ./bhlff/core/fft/fft_butterfly_computer.py
Methods: 12

### __init__(self, domain)
**Описание:** Initialize FFT butterfly operations computer.

Physical Meaning:
    Sets up the butterfly operations computer for efficient
    computation of FFT operation patterns.

Args:
    domain (Domain): Comp...

### compute_butterfly_tables_1d(self)
**Описание:** Compute butterfly operation tables for 1D FFT.

Physical Meaning:
    Pre-computes butterfly operation patterns for efficient
    FFT computation using divide-and-conquer algorithms.

Returns:
    Dic...

### compute_butterfly_tables_2d(self)
**Описание:** Compute butterfly operation tables for 2D FFT.

Physical Meaning:
    Pre-computes butterfly operation patterns for 2D FFT
    using row-column decomposition.

Returns:
    Dict[str, Any]: 2D butterfl...

### compute_butterfly_tables_3d(self)
**Описание:** Compute butterfly operation tables for 3D FFT.

Physical Meaning:
    Pre-computes butterfly operation patterns for 3D FFT
    using multi-dimensional decomposition.

Returns:
    Dict[str, Any]: 3D b...

### compute_butterfly(self, data)
**Описание:** Compute butterfly operation on data.

Args:
    data (np.ndarray): Input data.

Returns:
    np.ndarray: Result of butterfly operation.

### compute_inverse_butterfly(self, data)
**Описание:** Compute inverse butterfly operation on data.

Args:
    data (np.ndarray): Input data.

Returns:
    np.ndarray: Result of inverse butterfly operation.

### _butterfly_1d(self, data)
**Описание:** 1D butterfly operation.

### _butterfly_2d(self, data)
**Описание:** 2D butterfly operation.

### _butterfly_nd(self, data)
**Описание:** N-D butterfly operation.

### _inverse_butterfly_1d(self, data)
**Описание:** 1D inverse butterfly operation.

### _inverse_butterfly_2d(self, data)
**Описание:** 2D inverse butterfly operation.

### _inverse_butterfly_nd(self, data)
**Описание:** N-D inverse butterfly operation.

## ./bhlff/core/fft/fft_plan_7d.py
Methods: 12

### __init__(self, domain_shape, precision)
**Описание:** Initialize FFT plans.

Physical Meaning:
    Sets up optimized FFT plans for 7D computations with
    the specified precision and domain dimensions.

Args:
    domain_shape: Dimensions of 7D domain.
 ...

### execute_fft(self, field, direction)
**Описание:** Execute optimized FFT operation.

Physical Meaning:
    Performs FFT operation using pre-computed plans for
    maximum efficiency in 7D spectral computations.

Args:
    field: 7D field for transform...

### execute_block_fft(self, field, block_size, direction)
**Описание:** Execute FFT on blocks for large fields.

Physical Meaning:
    Performs FFT operations on blocks of the field to manage
    memory usage for large 7D computations.

Args:
    field: 7D field for trans...

### get_performance_stats(self)
**Описание:** Get performance statistics.

Physical Meaning:
    Returns detailed performance statistics for FFT operations,
    including timing and cache efficiency metrics.

Returns:
    Dict[str, Any]: Performa...

### optimize_plans(self)
**Описание:** Optimize FFT plans for better performance.

Physical Meaning:
    Performs optimization of FFT plans based on usage patterns
    and performance statistics.

### _setup_fft_plans(self, optimize)
**Описание:** Setup FFT plans for 7D operations.

Physical Meaning:
    Creates optimized plans for all necessary FFT operations
    in 7D space, including forward, inverse, and block operations.

Args:
    optimiz...

### setup_optimized_plans(self, precision, plan_type)
**Описание:** Setup or reconfigure optimized FFT plans.

Physical Meaning:
    Configures FFT planning parameters. For numpy backend this is a no-op,
    but we keep the API to satisfy advanced core expectations.

...

### _create_fft_plan(self, direction, optimize)
**Описание:** Create FFT plan for specified direction.

Physical Meaning:
    Creates an optimized FFT plan for the specified direction
    (forward or inverse) with the current domain configuration.

Args:
    dir...

### _create_block_fft_plan(self, direction, optimize)
**Описание:** Create block FFT plan for specified direction.

Physical Meaning:
    Creates an optimized FFT plan for block processing,
    enabling efficient FFT operations on large fields.

Args:
    direction: F...

### _execute_forward_fft(self, field)
**Описание:** Execute forward FFT using optimized plan.

Physical Meaning:
    Performs forward FFT transformation using the pre-computed
    plan for maximum efficiency.

Args:
    field: Input field.

Returns:
  ...

### _execute_inverse_fft(self, field)
**Описание:** Execute inverse FFT using optimized plan.

Physical Meaning:
    Performs inverse FFT transformation using the pre-computed
    plan for maximum efficiency.

Args:
    field: Input field.

Returns:
  ...

### cleanup(self)
**Описание:** Cleanup FFT plans and free resources.

Physical Meaning:
    Releases all FFT plan resources and clears caches
    to free memory.

## ./bhlff/core/fft/fft_plan_manager.py
Methods: 15

### __init__(self, domain, plan_type, precision)
**Описание:** Initialize FFT plan manager.

Physical Meaning:
    Sets up the FFT plan manager with specified planning strategy
    and precision for efficient spectral operations.

Args:
    domain (Domain): Compu...

### setup_fft_plans(self)
**Описание:** Setup FFT plans for optimization.

Physical Meaning:
    Pre-computes FFT plans for different array shapes to optimize
    subsequent FFT operations.

Mathematical Foundation:
    Creates optimized FF...

### _setup_optimized_fft_plans(self)
**Описание:** Setup optimized FFT plans using advanced algorithms.

Physical Meaning:
    Creates optimized FFT plans using advanced algorithms including
    cache optimization, memory alignment, and SIMD instructi...

### _create_1d_fft_plan(self)
**Описание:** Create optimized 1D FFT plan.

Physical Meaning:
    Creates an optimized plan for 1D FFT operations with
    cache-friendly memory access and SIMD optimization.

Returns:
    Dict[str, Any]: 1D FFT p...

### _create_1d_ifft_plan(self)
**Описание:** Create optimized 1D IFFT plan.

Physical Meaning:
    Creates an optimized plan for 1D inverse FFT operations.

Returns:
    Dict[str, Any]: 1D IFFT plan configuration.

### _create_2d_fft_plan(self)
**Описание:** Create optimized 2D FFT plan.

Physical Meaning:
    Creates an optimized plan for 2D FFT operations using
    row-column decomposition with cache optimization.

Returns:
    Dict[str, Any]: 2D FFT pl...

### _create_2d_ifft_plan(self)
**Описание:** Create optimized 2D IFFT plan.

Physical Meaning:
    Creates an optimized plan for 2D inverse FFT operations.

Returns:
    Dict[str, Any]: 2D IFFT plan configuration.

### _create_3d_fft_plan(self)
**Описание:** Create optimized 3D FFT plan.

Physical Meaning:
    Creates an optimized plan for 3D FFT operations using
    multi-dimensional decomposition with advanced optimization.

Returns:
    Dict[str, Any]:...

### _create_3d_ifft_plan(self)
**Описание:** Create optimized 3D IFFT plan.

Physical Meaning:
    Creates an optimized plan for 3D inverse FFT operations.

Returns:
    Dict[str, Any]: 3D IFFT plan configuration.

### get_fft_plans(self)
**Описание:** Get the FFT plans.

Physical Meaning:
    Returns the pre-computed FFT plans for forward and inverse
    transformations.

Returns:
    Dict[str, Any]: FFT plans dictionary.

### get_plan_type(self) -> str
**Описание:** Get the FFT plan type.

Physical Meaning:
    Returns the FFT planning strategy being used.

Returns:
    str: FFT plan type.

### get_precision(self) -> str
**Описание:** Get the numerical precision.

Physical Meaning:
    Returns the numerical precision being used for FFT operations.

Returns:
    str: Numerical precision.

### create_plan(self, field) -> str
**Описание:** Create FFT plan for field.

Args:
    field (np.ndarray): Field to create plan for.

Returns:
    str: Plan identifier.

### get_plan(self, field) -> str
**Описание:** Get existing FFT plan for field.

Args:
    field (np.ndarray): Field to get plan for.

Returns:
    str: Plan identifier.

### clear_plans(self)
**Описание:** Clear all FFT plans.

## ./bhlff/core/fft/fft_solver_7d_basic.py
Methods: 8

### __init__(self, domain, parameters)
**Описание:** Initialize 7D FFT solver.

Physical Meaning:
    Sets up the solver with the computational domain and
    physical parameters, pre-computing spectral coefficients
    for efficient solution of the fra...

### solve_stationary(self, source)
**Описание:** Solve the stationary fractional Laplacian equation.

Physical Meaning:
    Solves the stationary fractional Laplacian equation
    L_β a = s in spectral space, representing the steady-state
    soluti...

### solve_envelope(self, source)
**Описание:** Solve the envelope equation.

Physical Meaning:
    Solves the envelope equation for the 7D phase field,
    representing the evolution of the field envelope
    in 7D space-time.

Args:
    source (n...

### validate_solution(self, solution, source)
**Описание:** Validate the computed solution.

Physical Meaning:
    Validates the accuracy of the computed solution by
    checking how well it satisfies the original equation
    and computing quality metrics.

A...

### _setup_spectral_coefficients(self)
**Описание:** Setup spectral coefficients for fractional Laplacian.

Physical Meaning:
    Pre-computes the spectral representation of the fractional
    Laplacian operator, which is essential for efficient
    sol...

### _setup_fft_plan(self)
**Описание:** Setup FFT plan for efficient computations.

Physical Meaning:
    Pre-computes FFT plans to optimize the spectral
    transformations required for solving the fractional
    Laplacian equation efficie...

### _compute_residual(self, solution, source)
**Описание:** Compute residual of the fractional Laplacian equation.

Physical Meaning:
    Computes the residual R = L_β a - s, which measures
    how well the solution satisfies the original equation.

Args:
    ...

### get_solver_info(self)
**Описание:** Get solver information.

Returns:
    Dict[str, Any]: Solver information.

## ./bhlff/core/fft/fft_solver_time.py
Methods: 7

### __init__(self, domain, parameters)
**Описание:** Initialize time-dependent methods.

Physical Meaning:
    Sets up the time-dependent methods with the computational domain
    and solver parameters, initializing temporal integrators and
    supporti...

### solve_time_dependent(self, initial_field, source_field, time_steps, method)
**Описание:** Solve time-dependent problem using temporal integrators.

Physical Meaning:
    Solves the dynamic phase field equation:
    ∂a/∂t + ν(-Δ)^β a + λa = s(x,φ,t)
    using high-precision temporal integra...

### _setup_integrator_components(self, integrator)
**Описание:** Setup memory kernel and quench detector for integrator.

Physical Meaning:
    Configures the integrator with memory kernel for non-local
    temporal effects and quench detector for monitoring energy...

### set_memory_kernel(self, num_memory_vars, relaxation_times, coupling_strengths)
**Описание:** Configure memory kernel for non-local temporal effects.

Physical Meaning:
    Sets up the memory kernel with specified number of memory
    variables, relaxation times, and coupling strengths for
   ...

### set_quench_detector(self, energy_threshold, rate_threshold, magnitude_threshold)
**Описание:** Configure quench detection system.

Physical Meaning:
    Sets up the quench detection system with specified thresholds
    for monitoring energy dumping events during integration.

Args:
    energy_t...

### get_quench_history(self)
**Описание:** Get quench detection history.

Returns:
    List[Dict]: History of detected quench events.

### get_memory_contribution(self)
**Описание:** Get current memory kernel contribution.

Returns:
    np.ndarray: Current memory contribution to field.

## ./bhlff/core/fft/fft_solver_validation.py
Methods: 5

### __init__(self, domain, parameters, fractional_laplacian)
**Описание:** Initialize validation methods.

Physical Meaning:
    Sets up the validation methods with the computational domain,
    solver parameters, and fractional Laplacian operator.

Args:
    domain (Domain)...

### validate_solution(self, solution, source, tolerance)
**Описание:** Validate solution to fractional Laplacian equation.

Physical Meaning:
    Validates the solution by computing the residual of the equation
    L_β a = s and checking that it satisfies the equation wi...

### _compute_residual(self, solution, source)
**Описание:** Compute residual r = L_β a - s.

Physical Meaning:
    Computes the residual of the fractional Laplacian equation
    to validate the solution quality.

Mathematical Foundation:
    Residual: r = L_β ...

### check_energy_conservation(self, field, expected_energy, tolerance)
**Описание:** Check energy conservation for field.

Physical Meaning:
    Checks energy conservation by computing the total energy
    of the field and comparing with expected value.

Mathematical Foundation:
    E...

### check_boundary_conditions(self, field, boundary_type)
**Описание:** Check boundary conditions for field.

Physical Meaning:
    Checks that the field satisfies the specified boundary conditions,
    which are crucial for the correctness of the solution.

Args:
    fie...

## ./bhlff/core/fft/fft_twiddle_computer.py
Methods: 9

### __init__(self, domain, precision)
**Описание:** Initialize FFT twiddle factors computer.

Physical Meaning:
    Sets up the twiddle factors computer with specified precision
    for efficient computation of complex exponential factors.

Args:
    d...

### precompute_twiddle_factors(self)
**Описание:** Pre-compute twiddle factors for all FFT plans.

Physical Meaning:
    Pre-computes complex exponential factors used in FFT
    operations to avoid repeated computation during runtime.

Mathematical Fo...

### compute_twiddle_factors(self, dimensions, conjugate)
**Описание:** Compute twiddle factors for given dimensions.

Physical Meaning:
    Computes the complex exponential factors used in FFT
    operations for the specified number of dimensions.

Mathematical Foundatio...

### _compute_1d_twiddle_factors(self, conjugate)
**Описание:** Compute 1D twiddle factors.

Physical Meaning:
    Computes complex exponential factors for 1D FFT operations.

Args:
    conjugate (bool): Whether to compute conjugate factors.

Returns:
    np.ndarr...

### _compute_2d_twiddle_factors(self, conjugate)
**Описание:** Compute 2D twiddle factors.

Physical Meaning:
    Computes complex exponential factors for 2D FFT operations
    using row-column decomposition.

Args:
    conjugate (bool): Whether to compute conjug...

### _compute_3d_twiddle_factors(self, conjugate)
**Описание:** Compute 3D twiddle factors.

Physical Meaning:
    Computes complex exponential factors for 3D FFT operations
    using multi-dimensional decomposition.

Args:
    conjugate (bool): Whether to compute...

### get_twiddle_cache(self) -> Dict
**Описание:** Get the twiddle factors cache.

Physical Meaning:
    Returns the pre-computed twiddle factors cache.

Returns:
    Dict: Twiddle factors cache.

### get_twiddle_factor(self, dim1, dim2)
**Описание:** Get twiddle factor for specific dimensions.

Args:
    dim1 (int): First dimension.
    dim2 (int): Second dimension.

Returns:
    np.ndarray: Twiddle factor.

### compute_inverse_twiddle_factors(self)
**Описание:** Compute inverse twiddle factors.

Returns:
    Dict[str, np.ndarray]: Inverse twiddle factors.

## ./bhlff/core/fft/memory_manager_7d.py
Methods: 13

### __init__(self, domain_shape, max_memory_gb)
**Описание:** Initialize memory manager.

Physical Meaning:
    Sets up memory management for 7D computations with
    optimal block size calculation and memory monitoring.

Args:
    domain_shape: Dimensions of 7D...

### get_block(self, block_id)
**Описание:** Get block data, loading from storage if necessary.

Physical Meaning:
    Retrieves a block of the 7D field, loading it from
    compressed storage if not already in memory.

Args:
    block_id: Block...

### store_block(self, block_id, block_data)
**Описание:** Store block data in memory or compressed storage.

Physical Meaning:
    Stores block data, using compression if memory is limited
    and the block is not actively being used.

Args:
    block_id: Bl...

### release_block(self, block_id)
**Описание:** Release block from memory.

Physical Meaning:
    Removes block from active memory, optionally compressing
    it for later retrieval.

Args:
    block_id: Block identifier.

### get_memory_status(self)
**Описание:** Get current memory status.

Physical Meaning:
    Returns detailed information about memory usage and
    available resources for 7D computations.

Returns:
    Dict[str, Any]: Memory status informati...

### optimize_memory(self)
**Описание:** Optimize memory usage by compressing inactive blocks.

Physical Meaning:
    Performs memory optimization by compressing blocks that
    are not actively being used, freeing up memory for
    new comp...

### _calculate_optimal_block_size(self)
**Описание:** Calculate optimal block size for memory management.

Physical Meaning:
    Determines the optimal block size that fits in available
    memory while maintaining computational efficiency.

Returns:
   ...

### _create_empty_block(self, block_id)
**Описание:** Create empty block with appropriate size.

Physical Meaning:
    Creates a new empty block with dimensions determined
    by the block size and position.

Args:
    block_id: Block identifier.

Return...

### _should_compress_block(self, block_id) -> bool
**Описание:** Determine if block should be compressed.

Physical Meaning:
    Decides whether a block should be compressed based on
    memory usage and block access patterns.

Args:
    block_id: Block identifier....

### _compress_block(self, block_id, block_data)
**Описание:** Compress block data for storage.

Physical Meaning:
    Compresses block data to reduce memory usage while
    maintaining data integrity for later retrieval.

Args:
    block_id: Block identifier.
  ...

### _decompress_block(self, block_id)
**Описание:** Decompress block data from storage.

Physical Meaning:
    Decompresses block data from storage, restoring it
    to its original form for computation.

Args:
    block_id: Block identifier.

Returns:...

### _setup_memory_monitoring(self)
**Описание:** Setup memory monitoring and logging.

Physical Meaning:
    Initializes memory monitoring to track usage and
    provide warnings when memory limits are approached.

### cleanup(self)
**Описание:** Cleanup memory manager and free all resources.

Physical Meaning:
    Releases all memory resources and cleans up
    compressed storage.

## ./bhlff/core/fft/spectral_coefficient_cache.py
Methods: 11

### __init__(self, max_cache_size)
**Описание:** Initialize spectral coefficient cache.

Physical Meaning:
    Sets up the cache for spectral coefficients with
    configurable size limits and memory management.

Args:
    max_cache_size: Maximum nu...

### get_coefficients(self, mu, beta, lambda_param, domain_shape)
**Описание:** Get spectral coefficients from cache.

Physical Meaning:
    Returns spectral coefficients for the fractional operator,
    using cache for optimization or computing new ones if needed.

Args:
    mu:...

### clear_cache(self)
**Описание:** Clear all cached coefficients.

Physical Meaning:
    Removes all cached coefficients to free memory
    and reset the cache state.

### get_cache_stats(self)
**Описание:** Get cache statistics.

Physical Meaning:
    Returns detailed statistics about cache performance
    and memory usage.

Returns:
    Dict[str, Any]: Cache statistics.

### optimize_cache(self)
**Описание:** Optimize cache by removing least used entries.

Physical Meaning:
    Performs cache optimization by removing entries that
    are least frequently accessed to make room for new ones.

### _create_cache_key(self, mu, beta, lambda_param, domain_shape) -> str
**Описание:** Create cache key for parameters.

Physical Meaning:
    Creates a unique key for the given parameters to
    identify cached coefficients.

Args:
    mu: Diffusion coefficient.
    beta: Fractional or...

### _compute_coefficients(self, mu, beta, lambda_param, domain_shape)
**Описание:** Compute spectral coefficients.

Physical Meaning:
    Computes spectral coefficients μ|k|^(2β) + λ for the
    fractional operator in the given domain.

Args:
    mu: Diffusion coefficient.
    beta: ...

### _add_to_cache(self, cache_key, coefficients)
**Описание:** Add coefficients to cache.

Physical Meaning:
    Adds computed coefficients to the cache with
    memory management and access tracking.

Args:
    cache_key: Cache key.
    coefficients: Coefficient...

### _remove_from_cache(self, cache_key)
**Описание:** Remove coefficients from cache.

Physical Meaning:
    Removes coefficients from cache and updates
    memory usage tracking.

Args:
    cache_key: Cache key to remove.

### _estimate_memory_usage(self, domain_shape) -> int
**Описание:** Estimate memory usage for coefficients.

Physical Meaning:
    Estimates the memory required to store coefficients
    for the given domain shape.

Args:
    domain_shape: Domain dimensions.

Returns:...

### get_memory_info(self)
**Описание:** Get memory information for cache.

Physical Meaning:
    Returns detailed memory usage information
    for the coefficient cache.

Returns:
    Dict[str, Any]: Memory information.

## ./bhlff/core/fft/spectral_derivatives_base.py
Methods: 7

### __init__(self, domain, precision)
**Описание:** Initialize spectral derivatives base.

Physical Meaning:
    Sets up the base interface for spectral derivative operations
    with the computational domain and numerical precision.

Args:
    domain ...

### compute_gradient(self, field)
**Описание:** Compute gradient of field in spectral space.

Physical Meaning:
    Computes the gradient ∇a of the phase field in 7D space-time,
    representing the spatial and phase variations of the field.

Mathe...
**Декораторы:** abstractmethod

### compute_divergence(self, field)
**Описание:** Compute divergence of vector field in spectral space.

Physical Meaning:
    Computes the divergence ∇·a of the vector field in 7D space-time,
    representing the net flux of the field.

Mathematical...
**Декораторы:** abstractmethod

### compute_curl(self, field)
**Описание:** Compute curl of vector field in spectral space.

Physical Meaning:
    Computes the curl ∇×a of the vector field in 7D space-time,
    representing the rotational component of the field.

Mathematical...
**Декораторы:** abstractmethod

### compute_laplacian(self, field)
**Описание:** Compute Laplacian of field in spectral space.

Physical Meaning:
    Computes the Laplacian Δa of the phase field in 7D space-time,
    representing the second-order spatial variations of the field.

...
**Декораторы:** abstractmethod

### validate_field(self, field) -> bool
**Описание:** Validate field for derivative computation.

Physical Meaning:
    Ensures that the field is suitable for derivative computation,
    checking for proper shape, finite values, and compatibility
    wit...

### __repr__(self) -> str
**Описание:** String representation of spectral derivatives base.

## ./bhlff/core/fft/spectral_derivatives_impl.py
Methods: 11

### __init__(self, domain, precision)
**Описание:** Initialize spectral derivatives.

Physical Meaning:
    Sets up the spectral derivative operations with the computational
    domain and numerical precision, pre-computing wave vectors
    for efficie...

### compute_derivative(self, field, axis, order)
**Описание:** Compute nth derivative along a given axis using spectral method.

### compute_mixed_derivative(self, field, axes, orders)
**Описание:** Нет докстринга

### compute_gradient(self, field)
**Описание:** Compute gradient of field in spectral space.

Physical Meaning:
    Computes the gradient ∇a of the phase field in 7D space-time,
    representing the spatial and phase variations of the field.

Mathe...

### compute_divergence(self, field)
**Описание:** Compute divergence of vector field in spectral space.

Physical Meaning:
    Computes the divergence ∇·a of the vector field in 7D space-time,
    representing the net flux of the field.

Mathematical...

### compute_curl(self, field)
**Описание:** Compute curl of vector field in spectral space.

Physical Meaning:
    Computes the curl ∇×a of the vector field in 7D space-time,
    representing the rotational component of the field.

Mathematical...

### compute_laplacian(self, field)
**Описание:** Compute Laplacian of field in spectral space.

Physical Meaning:
    Computes the Laplacian Δa of the phase field in 7D space-time,
    representing the second-order spatial variations of the field.

...

### compute_bi_laplacian(self, field)
**Описание:** Compute bi-Laplacian (fourth-order derivative) of field.

Physical Meaning:
    Computes the bi-Laplacian Δ²a of the phase field, representing
    fourth-order spatial variations of the field.

Mathem...

### _compute_wave_vectors(self)
**Описание:** Compute wave vectors for the domain.

Physical Meaning:
    Computes the wave vectors k for each dimension of the domain,
    representing the frequency components in spectral space.

Returns:
    Tup...

### _compute_k_magnitude_squared(self)
**Описание:** Compute squared magnitude of wave vectors.

Physical Meaning:
    Computes |k|² for each point in the domain, representing
    the squared magnitude of the wave vector in spectral space.

Returns:
   ...

### __repr__(self) -> str
**Описание:** String representation of spectral derivatives.

## ./bhlff/core/fft/spectral_filtering.py
Methods: 8

### __init__(self, domain, precision)
**Описание:** Initialize spectral filtering.

Physical Meaning:
    Sets up the spectral filtering calculator with the computational
    domain and numerical precision, pre-computing wave vector magnitudes
    for ...

### apply_low_pass_filter(self, field, cutoff_frequency, order)
**Описание:** Apply low-pass filter to field in spectral space.

Physical Meaning:
    Applies a low-pass filter to remove high-frequency components
    from the phase field, representing smoothing of the field
   ...

### apply_high_pass_filter(self, field, cutoff_frequency, order)
**Описание:** Apply high-pass filter to field in spectral space.

Physical Meaning:
    Applies a high-pass filter to remove low-frequency components
    from the phase field, representing enhancement of small-scal...

### apply_band_pass_filter(self, field, center_frequency, bandwidth, order)
**Описание:** Apply band-pass filter to field in spectral space.

Physical Meaning:
    Applies a band-pass filter to preserve components within a
    specific frequency range, representing selective enhancement
  ...

### apply_gaussian_filter(self, field, sigma)
**Описание:** Apply Gaussian filter to field in spectral space.

Physical Meaning:
    Applies a Gaussian filter for smooth noise reduction,
    representing convolution with a Gaussian kernel in real space.

Mathe...

### apply_spectral_filter(self, field, filter_type)
**Описание:** Apply spectral filter of specified type.

Physical Meaning:
    Applies a spectral filter of the specified type to the field,
    providing a unified interface for different filtering operations.

Arg...

### apply_noise_reduction(self, field, noise_level, method)
**Описание:** Apply noise reduction to field.

Physical Meaning:
    Applies noise reduction to the phase field, representing
    removal of high-frequency noise while preserving signal content.

Args:
    field (n...

### _compute_k_magnitude(self)
**Описание:** Compute magnitude of wave vectors.

Physical Meaning:
    Computes the magnitude |k| of the wave vectors, representing
    the spatial frequency of the field components.

Mathematical Foundation:
    ...

## ./bhlff/core/fft/spectral_operations.py
Methods: 1

### __init__(self, domain, precision)
**Описание:** Initialize spectral operations.

Physical Meaning:
    Sets up the spectral operations calculator with the computational
    domain and numerical precision, initializing FFT backend and
    specialize...

## ./bhlff/core/fft/unified_spectral_operations.py
Methods: 15

### __init__(self, domain, precision)
**Описание:** Initialize unified spectral operations.

Physical Meaning:
    Sets up the spectral operations calculator with the computational
    domain and numerical precision, initializing FFT backend and
    sp...

### forward_fft(self, field, normalization)
**Описание:** Compute forward FFT of field.

Physical Meaning:
    Transforms the phase field from real space to frequency space,
    representing the field in terms of its frequency components.

Mathematical Found...

### inverse_fft(self, spectral_field, normalization)
**Описание:** Compute inverse FFT of spectral field.

Physical Meaning:
    Transforms the spectral field from frequency space back to real space,
    reconstructing the original phase field.

Mathematical Foundati...

### compute_spectral_derivatives(self, field, order)
**Описание:** Compute spectral derivatives of field.

Physical Meaning:
    Computes derivatives of the field using spectral methods,
    which are more accurate than finite difference methods
    for smooth fields...

### apply_spectral_filter(self, field, filter_type)
**Описание:** Apply spectral filter to field.

Physical Meaning:
    Applies various types of spectral filters to the field
    for noise reduction, smoothing, or feature extraction.

Args:
    field (np.ndarray): ...

### compute_spectral_energy(self, field)
**Описание:** Compute spectral energy density.

Physical Meaning:
    Computes the energy density in frequency space,
    representing the power spectrum of the field.

Mathematical Foundation:
    E(k) = |â(k)|² w...

### _setup_fft_plans(self)
**Описание:** Setup FFT plans for efficient computations.

Physical Meaning:
    Pre-computes FFT plans to optimize the spectral
    transformations required for solving the phase field
    equation efficiently.

### _compute_volume_element(self) -> float
**Описание:** Compute 7D volume element for physics normalization.

Physical Meaning:
    Computes the volume element Δ^7 = (dx^3) * (dphi^3) * dt
    for proper physics normalization of FFT operations.

Returns:
 ...

### _get_wave_vectors(self) -> list
**Описание:** Get wave vectors for each dimension.

Returns:
    list: List of wave vectors for each dimension.

### _create_wave_vector_grid(self, k_vec, axis, shape)
**Описание:** Create wave vector grid for a specific axis.

Args:
    k_vec (np.ndarray): Wave vector for the axis.
    axis (int): Axis index.
    shape (Tuple[int, ...]): Field shape.

Returns:
    np.ndarray: Wa...

### _create_lowpass_filter(self, cutoff)
**Описание:** Create lowpass filter mask.

### _create_highpass_filter(self, cutoff)
**Описание:** Create highpass filter mask.

### _create_bandpass_filter(self, low_cutoff, high_cutoff)
**Описание:** Create bandpass filter mask.

### get_spectral_info(self)
**Описание:** Get information about spectral operations.

Physical Meaning:
    Returns information about the spectral operations setup
    for monitoring and analysis purposes.

Returns:
    Dict[str, Any]: Spectr...

### __repr__(self) -> str
**Описание:** String representation of spectral operations.

## ./bhlff/core/operators/fractional_laplacian.py
Methods: 7

### __init__(self, domain, beta, lambda_param)
**Описание:** Initialize fractional Laplacian operator.

Physical Meaning:
    Sets up the fractional Laplacian with the specified fractional
    order β for non-local phase field interactions.

Args:
    domain (D...

### _setup_spectral_coefficients(self)
**Описание:** Setup spectral coefficients for fractional Laplacian.

Physical Meaning:
    Pre-computes the spectral representation |k|^(2β) of the
    fractional Laplacian for efficient application.

Mathematical ...

### apply(self, field)
**Описание:** Apply fractional Laplacian to field.

Physical Meaning:
    Applies the fractional Laplacian (-Δ)^β to the field,
    computing the non-local fractional derivative.

Mathematical Foundation:
    Compu...

### get_spectral_coefficients(self)
**Описание:** Get spectral coefficients of the fractional Laplacian.

Physical Meaning:
    Returns the pre-computed spectral coefficients |k|^(2β) for
    the fractional Laplacian.

Returns:
    np.ndarray: Spectr...

### get_fractional_order(self) -> float
**Описание:** Get the fractional order of the Laplacian.

Physical Meaning:
    Returns the fractional order β that determines the degree
    of non-locality in the operator.

Returns:
    float: Fractional order β...

### _compute_k_magnitude(self)
**Описание:** Compute 7D wave vector magnitude |k| for the domain grids.

### __repr__(self) -> str
**Описание:** String representation of the fractional Laplacian.

## ./bhlff/core/operators/memory_kernel.py
Methods: 10

### __init__(self, domain, kernel_type, parameters)
**Описание:** Initialize memory kernel.

Physical Meaning:
    Sets up the memory kernel with specified type and parameters
    for non-local phase field interactions.

Args:
    domain (Domain): Computational doma...

### _setup_kernel(self)
**Описание:** Setup memory kernel data.

Physical Meaning:
    Pre-computes the memory kernel data based on the specified
    kernel type and parameters.

Mathematical Foundation:
    Computes the kernel function K...

### _setup_power_law_kernel(self)
**Описание:** Setup power law memory kernel.

Physical Meaning:
    Implements a power law kernel K(r) ∝ r^(-α) that represents
    long-range interactions with power law decay.

Mathematical Foundation:
    Power ...

### _setup_exponential_kernel(self)
**Описание:** Setup exponential memory kernel.

Physical Meaning:
    Implements an exponential kernel K(r) ∝ exp(-r/λ) that represents
    short-range interactions with exponential decay.

Mathematical Foundation:...

### _setup_gaussian_kernel(self)
**Описание:** Setup Gaussian memory kernel.

Physical Meaning:
    Implements a Gaussian kernel K(r) ∝ exp(-r²/(2σ²)) that represents
    localized interactions with Gaussian decay.

Mathematical Foundation:
    Ga...

### apply(self, field)
**Описание:** Apply memory kernel to field.

Physical Meaning:
    Applies the memory kernel to the field, computing the non-local
    convolution operation.

Mathematical Foundation:
    Computes (K * a)(x) = ∫ K(...

### get_kernel_data(self)
**Описание:** Get the memory kernel data.

Physical Meaning:
    Returns the pre-computed memory kernel data K(x,y).

Returns:
    np.ndarray: Memory kernel data.

### get_kernel_type(self) -> str
**Описание:** Get the kernel type.

Physical Meaning:
    Returns the type of memory kernel being used.

Returns:
    str: Kernel type.

### get_parameters(self)
**Описание:** Get kernel parameters.

Physical Meaning:
    Returns the parameters used to define the memory kernel.

Returns:
    Dict[str, Any]: Kernel parameters.

### __repr__(self) -> str
**Описание:** String representation of the memory kernel.

## ./bhlff/core/operators/operator_riesz.py
Methods: 5

### __init__(self, domain, parameters)
**Описание:** Initialize fractional Riesz operator.

Physical Meaning:
    Sets up the operator with computational domain and physics
    parameters, pre-computing spectral coefficients for efficient
    applicatio...

### _setup_spectral_coefficients(self)
**Описание:** Setup spectral coefficients for fractional operator.

Physical Meaning:
    Pre-computes the spectral representation of the fractional
    Riesz operator for efficient application in frequency space.
...

### apply(self, field)
**Описание:** Apply fractional Riesz operator to field.

Physical Meaning:
    Applies the fractional Riesz operator L_β to the field,
    computing the result of the operator action on the phase
    field configur...

### get_spectral_coefficients(self)
**Описание:** Get spectral coefficients of the operator.

Physical Meaning:
    Returns the pre-computed spectral coefficients D(k) for
    the fractional Riesz operator.

Returns:
    np.ndarray: Spectral coeffici...

### __repr__(self) -> str
**Описание:** String representation of the operator.

## ./bhlff/core/phase/u1_phase_field.py
Methods: 10

### __init__(self, domain, initial_amplitudes, initial_phases)
**Описание:** Initialize U(1)³ phase field.

Physical Meaning:
    Creates a U(1)³ phase field with specified initial conditions
    for amplitudes and phases of each component.

Args:
    domain (Domain7DBVP): 7D ...

### get_field_at_point(self, x, phi, t)
**Описание:** Get field value at specific point.

Physical Meaning:
    Returns the U(1)³ phase field value at a specific point
    in 7D space-time.

Args:
    x (Tuple[int, int, int]): Spatial coordinates (x, y, ...

### set_field_at_point(self, x, phi, t, field_vector)
**Описание:** Set field value at specific point.

Physical Meaning:
    Sets the U(1)³ phase field value at a specific point
    in 7D space-time.

Args:
    x (Tuple[int, int, int]): Spatial coordinates (x, y, z)....

### compute_phase_coherence(self)
**Описание:** Compute phase coherence for each component.

Physical Meaning:
    Computes the phase coherence for each U(1) component,
    measuring the uniformity of phase distribution.

Mathematical Foundation:
 ...

### compute_amplitude_distribution(self)
**Описание:** Compute amplitude distribution statistics.

Physical Meaning:
    Computes statistical properties of the amplitude distribution
    for each component of the U(1)³ field.

Returns:
    Dict[str, np.nd...

### apply_gauge_transformation(self, gauge_function)
**Описание:** Apply U(1) gauge transformation to each component.

Physical Meaning:
    Applies a U(1) gauge transformation to each component of the field:
    aᵢ(x,φ,t) → aᵢ(x,φ,t) * e^(iαᵢ(x,φ,t))
    where αᵢ(x,...

### compute_field_norm(self)
**Описание:** Compute field norm at each point.

Physical Meaning:
    Computes the norm of the U(1)³ field at each point in space-time:
    |a(x,φ,t)| = √(|a₁|² + |a₂|² + |a₃|²)

Returns:
    np.ndarray: Field nor...

### get_field_component(self, index)
**Описание:** Get specific field component.

Args:
    index (int): Component index (0, 1, or 2).

Returns:
    np.ndarray: Field component.

### set_field_component(self, index, component)
**Описание:** Set specific field component.

Args:
    index (int): Component index (0, 1, or 2).
    component (np.ndarray): New field component.

### __repr__(self) -> str
**Описание:** String representation of U(1)³ phase field.

## ./bhlff/core/sources/bvp_source_core.py
Methods: 13

### __init__(self, domain, config)
**Описание:** Initialize BVP-modulated source.

Physical Meaning:
    Sets up the BVP-modulated source with carrier frequency and
    envelope parameters for generating modulated source terms.

Args:
    domain (Do...

### _setup_bvp_parameters(self)
**Описание:** Setup BVP source parameters.

Physical Meaning:
    Initializes the BVP source parameters from configuration
    including carrier frequency and envelope properties.

### generate(self)
**Описание:** Generate BVP-modulated source field.

Physical Meaning:
    Generates the BVP-modulated source field s(x) that represents
    external excitations modulated by the high-frequency carrier.

Mathematica...

### generate_base_source(self)
**Описание:** Generate base source without modulation.

Physical Meaning:
    Generates the base source s₀(x) without BVP modulation,
    representing the unmodulated external excitation.

Returns:
    np.ndarray: ...

### generate_envelope(self)
**Описание:** Generate envelope modulation.

Physical Meaning:
    Generates the envelope modulation A(x) that modulates the
    base source in the BVP framework.

Returns:
    np.ndarray: Envelope modulation field...

### generate_carrier(self, time)
**Описание:** Generate high-frequency carrier.

Physical Meaning:
    Generates the high-frequency carrier component exp(iω₀t) that
    modulates the source in the BVP framework.

Args:
    time (float): Time for c...

### get_source_type(self) -> str
**Описание:** Get the source type.

Physical Meaning:
    Returns the type of BVP-modulated source being generated.

Returns:
    str: Source type.

### get_carrier_frequency(self) -> float
**Описание:** Get the carrier frequency.

Physical Meaning:
    Returns the high-frequency carrier frequency used in
    BVP modulation.

Returns:
    float: Carrier frequency.

### get_envelope_amplitude(self) -> float
**Описание:** Get the envelope amplitude.

Physical Meaning:
    Returns the envelope amplitude used in BVP modulation.

Returns:
    float: Envelope amplitude.

### get_base_source_type(self) -> str
**Описание:** Get the base source type.

Physical Meaning:
    Returns the type of base source being used.

Returns:
    str: Base source type.

### get_supported_source_types(self) -> list
**Описание:** Get supported source types.

Physical Meaning:
    Returns the list of supported base source types for
    BVP modulation.

Returns:
    list: Supported source types.

### get_source_info(self)
**Описание:** Get source information.

Physical Meaning:
    Returns comprehensive information about the BVP-modulated
    source including parameters and configuration.

Returns:
    Dict[str, Any]: Source informa...

### __repr__(self) -> str
**Описание:** String representation of the BVP-modulated source.

## ./bhlff/core/sources/bvp_source_envelope.py
Methods: 6

### __init__(self, domain, config)
**Описание:** Initialize BVP source envelope generator.

Physical Meaning:
    Sets up the BVP source envelope generator with domain and
    configuration for generating envelope and carrier components.

Args:
    ...

### generate_envelope(self)
**Описание:** Generate envelope modulation.

Physical Meaning:
    Creates the envelope modulation A(x) that modulates the
    base source in the BVP framework.

Mathematical Foundation:
    Envelope: A(x) = A₀ * f...

### generate_carrier(self, time)
**Описание:** Generate high-frequency carrier.

Physical Meaning:
    Creates the high-frequency carrier component exp(iω₀t) that
    modulates the source in the BVP framework.

Mathematical Foundation:
    Carrier...

### generate_modulated_source(self, base_source, time)
**Описание:** Generate BVP-modulated source.

Physical Meaning:
    Creates the complete BVP-modulated source by combining the
    base source, envelope, and carrier components.

Mathematical Foundation:
    BVP-mo...

### get_envelope_info(self)
**Описание:** Get envelope information.

Physical Meaning:
    Returns information about the envelope generation including
    parameters and configuration.

Returns:
    Dict[str, Any]: Envelope information.

### get_carrier_info(self)
**Описание:** Get carrier information.

Physical Meaning:
    Returns information about the carrier generation including
    frequency and mathematical description.

Returns:
    Dict[str, Any]: Carrier information...

## ./bhlff/core/sources/bvp_source_generators.py
Methods: 7

### __init__(self, domain, config)
**Описание:** Initialize BVP source generators.

Physical Meaning:
    Sets up the BVP source generators with domain and configuration
    for generating various types of base sources.

Args:
    domain (Domain): C...

### generate_gaussian_source(self)
**Описание:** Generate Gaussian source.

Physical Meaning:
    Creates a Gaussian source distribution centered at a specified
    location with given width and amplitude.

Mathematical Foundation:
    Gaussian sour...

### generate_point_source(self)
**Описание:** Generate point source.

Physical Meaning:
    Creates a point source at a specified location with given
    amplitude, representing a localized excitation.

Mathematical Foundation:
    Point source: ...

### generate_distributed_source(self)
**Описание:** Generate distributed source.

Physical Meaning:
    Creates a distributed source with specified spatial distribution
    and amplitude profile.

Mathematical Foundation:
    Distributed source: s(x) =...

### generate_base_source(self, source_type)
**Описание:** Generate base source of specified type.

Physical Meaning:
    Generates a base source of the specified type for BVP modulation.

Args:
    source_type (str): Type of source to generate.

Returns:
   ...

### get_supported_source_types(self) -> list
**Описание:** Get supported source types.

Physical Meaning:
    Returns the list of supported source types for BVP modulation.

Returns:
    list: Supported source types.

### get_source_info(self, source_type)
**Описание:** Get information about source type.

Physical Meaning:
    Returns information about the specified source type including
    parameters and mathematical description.

Args:
    source_type (str): Sourc...

## ./bhlff/core/sources/source.py
Methods: 6

### __init__(self, domain, config)
**Описание:** Initialize source.

Physical Meaning:
    Sets up the source with computational domain and configuration
    parameters for generating source terms.

Args:
    domain (Domain): Computational domain fo...

### generate(self)
**Описание:** Generate source field.

Physical Meaning:
    Generates the source field s(x) that represents external
    excitations or initial conditions for phase field evolution.

Mathematical Foundation:
    Cr...
**Декораторы:** abstractmethod

### get_source_type(self) -> str
**Описание:** Get the source type.

Physical Meaning:
    Returns the type of source being used.

Returns:
    str: Source type.

Raises:
    NotImplementedError: Must be implemented by subclasses.
**Декораторы:** abstractmethod

### get_domain(self) -> Domain
**Описание:** Get the computational domain.

Physical Meaning:
    Returns the computational domain for the source.

Returns:
    Domain: Computational domain.

### get_config(self)
**Описание:** Get the source configuration.

Physical Meaning:
    Returns the configuration parameters for the source.

Returns:
    Dict[str, Any]: Source configuration.

### __repr__(self) -> str
**Описание:** String representation of the source.

## ./bhlff/core/time/adaptive/adaptive_integrator.py
Methods: 12

### __init__(self, domain, parameters, tolerance, safety_factor, min_dt, max_dt)
**Описание:** Initialize adaptive integrator.

Physical Meaning:
    Sets up the adaptive integrator with the computational domain
    and physics parameters, configuring error control and time step
    management ...

### _setup_spectral_coefficients(self)
**Описание:** Setup spectral coefficients for adaptive integrator.

Physical Meaning:
    Pre-computes the spectral representation of the operator
    for efficient adaptive integration with error estimation.

### integrate(self, initial_field, source_field, time_steps)
**Описание:** Integrate the dynamic equation over time using adaptive method.

Physical Meaning:
    Solves the dynamic phase field equation over the specified
    time steps using adaptive time stepping with autom...

### step(self, current_field, source_field, dt)
**Описание:** Perform a single adaptive time step.

Physical Meaning:
    Advances the field configuration by one time step using
    adaptive error control and step size adjustment.

Args:
    current_field (np.nd...

### _adaptive_step_to_time(self, current_field, current_time, target_time, source_field, time_index)
**Описание:** Adaptively step from current_time to target_time.

Physical Meaning:
    Performs adaptive integration from current time to target time,
    automatically adjusting step size to maintain accuracy.

### _compute_rhs(self, field, source)
**Описание:** Compute right-hand side of the dynamic equation.

Physical Meaning:
    Computes the right-hand side of the dynamic phase field equation:
    RHS = -ν(-Δ)^β a - λa + s(x,φ,t)

### _adjust_time_step(self, error_estimate, current_dt)
**Описание:** Adjust time step based on full error analysis.

Physical Meaning:
    Automatically adjusts the time step based on comprehensive error
    estimation including Richardson extrapolation, stability anal...

### _apply_stability_constraints(self, proposed_dt, error_estimate) -> float
**Описание:** Apply stability constraints to proposed time step.

Physical Meaning:
    Applies stability constraints including CFL conditions
    and spectral stability requirements to ensure numerical stability.

### get_current_time_step(self) -> float
**Описание:** Get current adaptive time step.

### set_tolerance(self, tolerance)
**Описание:** Set error tolerance for adaptive control.

### set_time_step_bounds(self, min_dt, max_dt)
**Описание:** Set time step bounds.

### get_integrator_info(self)
**Описание:** Get information about the integrator.

## ./bhlff/core/time/adaptive/error_estimation.py
Methods: 8

### __init__(self, tolerance, safety_factor)
**Описание:** Initialize error estimator.

### compute_richardson_error(self, field_4th, field_5th, dt) -> float
**Описание:** Compute error estimate using full Richardson extrapolation.

Physical Meaning:
    Uses Richardson extrapolation to provide a more accurate
    error estimate for adaptive step size control. Implement...

### _analyze_error_components(self, error_diff, field)
**Описание:** Analyze different components of the error.

### _compute_high_frequency_mask(self, shape)
**Описание:** Compute mask for high-frequency components.

### _combine_error_estimates(self, basic_error, error_components) -> float
**Описание:** Combine different error estimates into a single error measure.

### _analyze_stability(self, field_4th, field_5th, dt)
**Описание:** Analyze stability of the integration step.

Physical Meaning:
    Analyzes the stability of the integration step by examining
    the growth of errors and the spectral properties of the solution.

### _compute_local_truncation_error(self, field, dt) -> float
**Описание:** Compute local truncation error estimate.

Physical Meaning:
    Estimates the local truncation error by analyzing the
    high-order derivatives of the field.

### _combine_error_estimates_full(self, basic_error, error_components, stability_analysis, truncation_error) -> float
**Описание:** Combine all error estimates into a single error measure.

Physical Meaning:
    Combines Richardson extrapolation error, component analysis,
    stability analysis, and truncation error into a compreh...

## ./bhlff/core/time/adaptive/runge_kutta.py
Methods: 3

### __init__(self)
**Описание:** Initialize Runge-Kutta methods.

### embedded_rk_step(self, field, source, dt, compute_rhs)
**Описание:** Perform embedded Runge-Kutta step with full error estimation.

Physical Meaning:
    Uses embedded Runge-Kutta method to compute both fourth-order
    accurate solution and fifth-order error estimate ...

### _compute_richardson_error(self, field_4th, field_5th, dt) -> float
**Описание:** Compute error estimate using Richardson extrapolation.

Physical Meaning:
    Uses Richardson extrapolation to provide a more accurate
    error estimate for adaptive step size control.

## ./bhlff/core/time/base_integrator.py
Methods: 10

### __init__(self, domain, parameters)
**Описание:** Initialize base time integrator.

Physical Meaning:
    Sets up the integrator with the computational domain and
    physics parameters, preparing for temporal integration of
    the dynamic phase fie...

### integrate(self, initial_field, source_field, time_steps)
**Описание:** Integrate the dynamic equation over time.

Physical Meaning:
    Solves the dynamic phase field equation over the specified
    time steps, representing the temporal evolution of the phase
    field c...
**Декораторы:** abstractmethod

### step(self, current_field, source_field, dt)
**Описание:** Perform a single time step.

Physical Meaning:
    Advances the field configuration by one time step,
    representing the local temporal evolution of the phase field.

Args:
    current_field (np.nda...
**Декораторы:** abstractmethod

### set_memory_kernel(self, memory_kernel)
**Описание:** Set memory kernel for non-local effects.

Physical Meaning:
    Configures the memory kernel to account for non-local
    temporal effects in the phase field evolution.

Args:
    memory_kernel (Memor...

### set_quench_detector(self, quench_detector)
**Описание:** Set quench detection system.

Physical Meaning:
    Configures the quench detection system to monitor
    for energy dumping events during integration.

Args:
    quench_detector (QuenchDetector): Que...

### _validate_parameters(self)
**Описание:** Validate integrator parameters.

Physical Meaning:
    Ensures all physical parameters are within valid ranges
    for the dynamic phase field equation.

### _check_quench(self, field, time) -> bool
**Описание:** Check for quench events.

Physical Meaning:
    Monitors the field for energy dumping events that may
    require special handling during integration.

Args:
    field (np.ndarray): Current field conf...

### _apply_memory_kernel(self, field, time)
**Описание:** Apply memory kernel effects.

Physical Meaning:
    Applies non-local temporal effects through the memory kernel,
    accounting for the system's memory of past configurations.

Args:
    field (np.nd...

### is_initialized(self) -> bool
**Описание:** Check if integrator is initialized.
**Декораторы:** property

### __repr__(self) -> str
**Описание:** String representation of integrator.

## ./bhlff/core/time/bvp_envelope_integrator.py
Methods: 7

### __init__(self, domain, parameters)
**Описание:** Initialize BVP envelope integrator.

Physical Meaning:
    Sets up the envelope integrator with the computational domain
    and physics parameters, pre-computing envelope coefficients
    for efficie...

### _setup_envelope_coefficients(self)
**Описание:** Setup envelope coefficients for BVP integrator.

Physical Meaning:
    Pre-computes the envelope coefficients for the BVP envelope equation
    including nonlinear stiffness and susceptibility terms.

### integrate(self, initial_field, source_field, time_steps)
**Описание:** Integrate the envelope equation over time using BVP approach.

Physical Meaning:
    Solves the BVP envelope equation over the specified
    time steps using envelope modulation approach,
    represen...

### step(self, current_field, source_field, dt)
**Описание:** Perform a single time step using BVP envelope approach.

Physical Meaning:
    Advances the field configuration by one time step using the
    BVP envelope approach, representing envelope modulations ...

### integrate_envelope_modulation(self, initial_field, carrier_frequency, modulation_depth, time_steps)
**Описание:** Integrate with envelope modulation using BVP approach.

Physical Meaning:
    Solves the envelope equation with envelope modulation
    representing the BVP approach where all observed "modes"
    are...

### _handle_quench_event(self, field, time)
**Описание:** Handle quench event according to BVP theory.

Physical Meaning:
    Implements quench handling according to BVP theory where
    quenches represent dissipative energy dumps in the envelope.

Args:
   ...

### __repr__(self) -> str
**Описание:** String representation of integrator.

## ./bhlff/core/time/crank_nicolson_integrator.py
Methods: 6

### __init__(self, domain, parameters)
**Описание:** Initialize Crank-Nicolson integrator.

Physical Meaning:
    Sets up the Crank-Nicolson integrator with the computational domain
    and physics parameters, pre-computing spectral coefficients
    for...

### _setup_spectral_coefficients(self)
**Описание:** Setup spectral coefficients for Crank-Nicolson integrator.

Physical Meaning:
    Pre-computes the spectral representation of the operator
    for efficient Crank-Nicolson integration.

### integrate(self, initial_field, source_field, time_steps)
**Описание:** Integrate the dynamic equation over time using Crank-Nicolson method.

Physical Meaning:
    Solves the dynamic phase field equation over the specified
    time steps using the Crank-Nicolson scheme, ...

### step(self, current_field, current_source, next_source, dt)
**Описание:** Perform a single time step using Crank-Nicolson method.

Physical Meaning:
    Advances the field configuration by one time step using the
    Crank-Nicolson scheme, providing second-order accuracy
  ...

### step_implicit(self, current_field, source_field, dt)
**Описание:** Perform a single time step using implicit Crank-Nicolson method.

Physical Meaning:
    Advances the field configuration by one time step using the
    implicit Crank-Nicolson scheme, providing uncond...

### __repr__(self) -> str
**Описание:** String representation of integrator.

## ./bhlff/core/time/memory_kernel.py
Methods: 10

### __init__(self, domain, num_memory_vars)
**Описание:** Initialize memory kernel.

Physical Meaning:
    Sets up the memory kernel with specified number of memory
    variables, each with its own relaxation time and coupling strength.

Args:
    domain (Do...

### _setup_memory_system(self)
**Описание:** Setup memory system with default parameters.

Physical Meaning:
    Initializes memory variables with default relaxation times
    and coupling strengths for typical 7D phase field dynamics.

### apply(self, field, time)
**Описание:** Apply memory kernel effects to field.

Physical Meaning:
    Applies the combined effect of all memory variables to the
    current field configuration, representing non-local temporal
    influences ...

### evolve(self, field, dt)
**Описание:** Evolve memory variables.

Physical Meaning:
    Updates memory variables according to their evolution equation,
    incorporating the current field configuration as a source term.

Mathematical Founda...

### reset(self)
**Описание:** Reset memory variables to zero.

Physical Meaning:
    Clears all memory of past configurations, effectively
    starting with a fresh memory state.

### set_relaxation_times(self, taus)
**Описание:** Set relaxation times for memory variables.

Physical Meaning:
    Configures the relaxation times τⱼ for each memory variable,
    controlling how quickly each variable forgets past information.

Args...

### set_coupling_strengths(self, gammas)
**Описание:** Set coupling strengths for memory variables.

Physical Meaning:
    Configures the coupling strengths γⱼ for each memory variable,
    controlling how strongly each variable influences the field.

Arg...

### _validate_passivity(self)
**Описание:** Validate PASS-1: ReY(ω)≥0 for memory kernels below resonances.

Physical Meaning:
    Ensures that the memory kernel frequency response Y(ω) has
    non-negative real part below resonances, maintainin...

### get_memory_contribution(self)
**Описание:** Get total memory contribution to field.

Physical Meaning:
    Returns the combined contribution of all memory variables
    to the field evolution.

Returns:
    np.ndarray: Total memory contribution...

### __repr__(self) -> str
**Описание:** String representation of memory kernel.

## ./bhlff/core/time/quench_detector.py
Methods: 8

### __init__(self, domain, energy_threshold, rate_threshold, magnitude_threshold)
**Описание:** Initialize quench detector.

Physical Meaning:
    Sets up the quench detection system with specified thresholds
    for monitoring energy dumping events in the phase field.

Args:
    domain (Domain)...

### detect_quench(self, field, time) -> bool
**Описание:** Detect quench events in the field.

Physical Meaning:
    Analyzes the current field configuration for signs of energy
    dumping or sudden changes that may indicate quench events.

Mathematical Foun...

### _calculate_energy(self, field) -> float
**Описание:** Calculate energy of the field configuration.

Physical Meaning:
    Computes the total energy of the phase field configuration,
    which is used for quench detection.

Mathematical Foundation:
    En...

### get_quench_history(self)
**Описание:** Get history of detected quench events.

Returns:
    List[Dict]: List of quench events with details.

### clear_history(self)
**Описание:** Clear quench event history.

### set_thresholds(self, energy_threshold, rate_threshold, magnitude_threshold)
**Описание:** Update detection thresholds.

Physical Meaning:
    Adjusts the sensitivity of quench detection by modifying
    the thresholds for energy change, rate of change, and
    field magnitude.

Args:
    e...

### get_statistics(self)
**Описание:** Get quench detection statistics.

Returns:
    Dict[str, Any]: Statistics about detected quench events.

### __repr__(self) -> str
**Описание:** String representation of quench detector.

## ./bhlff/geometry/layers/layer_stack.py
Methods: 14

### __init__(self, center)
**Описание:** Initialize layer stack.

Physical Meaning:
    Sets up the layer stack with a common center for all
    spherical layers.

Args:
    center (Tuple[float, float, float]): Common center for all layers.

### add_layer(self, inner_radius, outer_radius, properties) -> int
**Описание:** Add a spherical layer to the stack.

Physical Meaning:
    Adds a new spherical layer to the stack with specified
    inner and outer radii and optional properties.

Mathematical Foundation:
    Creat...

### _validate_layer_boundaries(self, inner_radius, outer_radius)
**Описание:** Validate layer boundaries against existing layers.

Physical Meaning:
    Ensures that new layer boundaries do not conflict with
    existing layers in the stack.

Mathematical Foundation:
    Validat...

### get_layer(self, index) -> SphericalLayer
**Описание:** Get layer by index.

Physical Meaning:
    Returns the spherical layer at the specified index.

Args:
    index (int): Index of the layer.

Returns:
    SphericalLayer: The requested layer.

Raises:
 ...

### get_layer_properties(self, index)
**Описание:** Get properties of a layer.

Physical Meaning:
    Returns the properties associated with the specified layer.

Args:
    index (int): Index of the layer.

Returns:
    Dict[str, Any]: Properties of th...

### set_layer_properties(self, index, properties)
**Описание:** Set properties of a layer.

Physical Meaning:
    Sets the properties associated with the specified layer.

Args:
    index (int): Index of the layer.
    properties (Dict[str, Any]): Properties to se...

### get_total_volume(self) -> float
**Описание:** Get total volume of all layers.

Physical Meaning:
    Computes the total volume of all layers in the stack,
    representing the total volume of the geometric structure.

Mathematical Foundation:
   ...

### get_total_surface_area(self) -> float
**Описание:** Get total surface area of all layer boundaries.

Physical Meaning:
    Computes the total surface area of all layer boundaries,
    representing the total surface area of the geometric structure.

Mat...

### get_layer_containing_point(self, x, y, z)
**Описание:** Get index of layer containing the specified point.

Physical Meaning:
    Determines which layer (if any) contains the specified point,
    providing spatial localization within the layer structure.

...

### get_radial_coordinate(self, x, y, z) -> float
**Описание:** Get radial coordinate of a point.

Physical Meaning:
    Computes the radial distance of a point from the center
    of the layer stack.

Mathematical Foundation:
    Radial coordinate: r = √((x-cx)² ...

### get_number_of_layers(self) -> int
**Описание:** Get number of layers in the stack.

Physical Meaning:
    Returns the total number of layers in the stack.

Returns:
    int: Number of layers.

### get_center(self)
**Описание:** Get center coordinates of the stack.

Physical Meaning:
    Returns the common center coordinates of all layers.

Returns:
    Tuple[float, float, float]: Center coordinates (x, y, z).

### clear(self)
**Описание:** Clear all layers from the stack.

Physical Meaning:
    Removes all layers from the stack, resetting it to
    an empty state.

### __repr__(self) -> str
**Описание:** String representation of the layer stack.

## ./bhlff/geometry/layers/spherical_layer.py
Methods: 10

### __init__(self, inner_radius, outer_radius, center, resolution)
**Описание:** Initialize spherical layer.

Physical Meaning:
    Sets up the spherical layer with specified inner and outer
    radii, center position, and angular resolution.

Args:
    inner_radius (float): Inner...

### get_coordinates(self)
**Описание:** Get spherical coordinates for the layer.

Physical Meaning:
    Generates spherical coordinates (r, θ, φ) for the layer,
    providing the coordinate system for phase field calculations.

Mathematical...

### get_cartesian_coordinates(self)
**Описание:** Get Cartesian coordinates for the layer.

Physical Meaning:
    Converts spherical coordinates to Cartesian coordinates,
    providing the standard coordinate system for computations.

Mathematical Fo...

### get_volume(self) -> float
**Описание:** Get volume of the spherical layer.

Physical Meaning:
    Computes the volume of the spherical layer, representing
    the total volume enclosed by the layer boundaries.

Mathematical Foundation:
    ...

### get_surface_area(self)
**Описание:** Get surface areas of the layer boundaries.

Physical Meaning:
    Computes the surface areas of the inner and outer
    boundaries of the spherical layer.

Mathematical Foundation:
    Surface area of...

### contains_point(self, x, y, z) -> bool
**Описание:** Check if point is inside the spherical layer.

Physical Meaning:
    Determines whether a given point lies within the
    spherical layer boundaries.

Mathematical Foundation:
    Point (x, y, z) is i...

### get_layer_thickness(self) -> float
**Описание:** Get thickness of the spherical layer.

Physical Meaning:
    Computes the radial thickness of the spherical layer,
    representing the distance between inner and outer boundaries.

Mathematical Found...

### get_center(self)
**Описание:** Get center coordinates of the layer.

Physical Meaning:
    Returns the center coordinates of the spherical layer.

Returns:
    Tuple[float, float, float]: Center coordinates (x, y, z).

### get_radii(self)
**Описание:** Get inner and outer radii of the layer.

Physical Meaning:
    Returns the inner and outer radii of the spherical layer.

Returns:
    Tuple[float, float]: (inner_radius, outer_radius).

### __repr__(self) -> str
**Описание:** String representation of the spherical layer.

## ./bhlff/models/base/abstract_model.py
Methods: 8

### __init__(self, domain)
**Описание:** Initialize abstract model.

Physical Meaning:
    Sets up the base functionality for all model components,
    providing access to domain information.

Args:
    domain (Domain): Computational domain

### analyze(self, data)
**Описание:** Analyze data for this model.

Physical Meaning:
    Performs model-specific analysis of the input data,
    extracting relevant physical quantities and properties.

Mathematical Foundation:
    Implem...
**Декораторы:** abstractmethod

### validate_domain(self) -> bool
**Описание:** Validate computational domain.

Physical Meaning:
    Checks that the computational domain is properly
    configured for the model to function correctly.

Returns:
    bool: True if domain is valid

### get_domain_info(self)
**Описание:** Get domain information.

Physical Meaning:
    Retrieves information about the computational domain,
    including dimensions, size, and resolution.

Returns:
    Dict: Domain information

### log_analysis_start(self, analysis_type)
**Описание:** Log analysis start.

Physical Meaning:
    Provides consistent logging for analysis operations,
    helping with debugging and monitoring.

Args:
    analysis_type (str): Type of analysis being perfor...

### log_analysis_complete(self, analysis_type, results)
**Описание:** Log analysis completion.

Physical Meaning:
    Provides consistent logging for analysis completion,
    including summary of results.

Args:
    analysis_type (str): Type of analysis performed
    re...

### __str__(self) -> str
**Описание:** String representation of the model.

### __repr__(self) -> str
**Описание:** Detailed string representation of the model.

## ./bhlff/models/base/abstract_models.py
Methods: 9

### __init__(self, domain, parameters)
**Описание:** Initialize abstract level models.

Physical Meaning:
    Sets up the base functionality for all model components,
    providing access to domain information and parameters.

Args:
    domain (Domain):...

### analyze_field(self, field)
**Описание:** Analyze field for this level.

Physical Meaning:
    Performs level-specific analysis of the phase field,
    extracting relevant physical quantities and properties.

Mathematical Foundation:
    Impl...
**Декораторы:** abstractmethod

### validate_parameters(self) -> bool
**Описание:** Validate model parameters.

Physical Meaning:
    Checks that all required parameters are present and
    within valid ranges for the model to function correctly.

Returns:
    bool: True if parameter...

### get_parameter(self, key, default) -> Any
**Описание:** Get parameter value.

Physical Meaning:
    Retrieves parameter values with fallback to default
    values, ensuring robust parameter access.

Args:
    key (str): Parameter name
    default (Any): De...

### set_parameter(self, key, value)
**Описание:** Set parameter value.

Physical Meaning:
    Updates parameter values, ensuring that changes
    are properly tracked and validated.

Args:
    key (str): Parameter name
    value (Any): Parameter valu...

### log_analysis_start(self, analysis_type)
**Описание:** Log analysis start.

Physical Meaning:
    Provides consistent logging for analysis operations,
    helping with debugging and monitoring.

Args:
    analysis_type (str): Type of analysis being perfor...

### log_analysis_complete(self, analysis_type, results)
**Описание:** Log analysis completion.

Physical Meaning:
    Provides consistent logging for analysis completion,
    including summary of results.

Args:
    analysis_type (str): Type of analysis performed
    re...

### __str__(self) -> str
**Описание:** String representation of the model.

### __repr__(self) -> str
**Описание:** Detailed string representation of the model.

## ./bhlff/models/base/model_base.py
Methods: 10

### __init__(self, domain)
**Описание:** Initialize base model.

Physical Meaning:
    Sets up the base functionality for all model components,
    providing access to domain information and logging.

Args:
    domain (Optional[Domain]): Com...

### validate_domain(self) -> bool
**Описание:** Validate computational domain.

Physical Meaning:
    Checks that the computational domain is properly
    configured for the model to function correctly.

Returns:
    bool: True if domain is valid

### get_domain_info(self)
**Описание:** Get domain information.

Physical Meaning:
    Retrieves information about the computational domain,
    including dimensions, size, and resolution.

Returns:
    Dict: Domain information

### log_analysis_start(self, analysis_type)
**Описание:** Log analysis start.

Physical Meaning:
    Provides consistent logging for analysis operations,
    helping with debugging and monitoring.

Args:
    analysis_type (str): Type of analysis being perfor...

### log_analysis_complete(self, analysis_type, results)
**Описание:** Log analysis completion.

Physical Meaning:
    Provides consistent logging for analysis completion,
    including summary of results.

Args:
    analysis_type (str): Type of analysis performed
    re...

### validate_array(self, array, name) -> bool
**Описание:** Validate numpy array.

Physical Meaning:
    Checks that the array is properly formatted and contains
    valid numerical data for the model calculations.

Args:
    array (np.ndarray): Array to valid...

### validate_parameters(self, parameters) -> bool
**Описание:** Validate model parameters.

Physical Meaning:
    Checks that the model parameters are within
    physically reasonable ranges and are consistent.

Args:
    parameters (Dict[str, Any]): Parameters to...

### compute_statistics(self, data)
**Описание:** Compute basic statistics for data.

Physical Meaning:
    Computes fundamental statistical properties of the data,
    providing insight into the distribution and characteristics.

Args:
    data (np....

### __str__(self) -> str
**Описание:** String representation of the model.

### __repr__(self) -> str
**Описание:** Detailed string representation of the model.

## ./bhlff/models/level_b/node_analysis/charge_computation.py
Methods: 8

### __init__(self, bvp_core)
**Описание:** Initialize charge computer.

### compute_topological_charge(self, envelope) -> float
**Описание:** Compute full 7D topological charge.

Physical Meaning:
    Computes the complete topological charge in 7D space-time
    using full topological analysis according to the 7D theory.

### _compute_7d_phase_gradients(self, phase)
**Описание:** Compute full 7D phase gradients.

### _compute_7d_charge_density(self, phase_gradients)
**Описание:** Compute full 7D topological charge density.

Physical Meaning:
    Computes the complete 7D topological charge density using
    the full 7D Levi-Civita tensor and gauge field gradients.

Mathematical...

### _compute_full_7d_charge_density(self, phase_gradients)
**Описание:** Compute full 7D topological charge density using 7D Levi-Civita tensor.

### _compute_3d_charge_density(self, phase_gradients)
**Описание:** Compute 3D topological charge density as fallback.

### _compute_7d_levi_civita(self, mu, nu, rho, sigma, tau) -> int
**Описание:** Compute 7D Levi-Civita symbol.

Physical Meaning:
    Computes the 7D Levi-Civita symbol ε^{μνρστ} which is
    +1 for even permutations, -1 for odd permutations, and 0 otherwise.

### _compute_7d_volume_element(self) -> float
**Описание:** Compute 7D volume element.

## ./bhlff/models/level_b/node_analysis/node_analysis.py
Methods: 11

### __init__(self, bvp_core)
**Описание:** Initialize node analyzer.

Args:
    bvp_core (BVPCore): BVP core instance for analysis.

### identify_nodes(self, envelope)
**Описание:** Identify node locations in the field using full topological analysis.

Physical Meaning:
    Identifies critical points in the BVP field where the
    field gradient vanishes, using complete topologic...

### _compute_adaptive_threshold(self, grad_magnitude, amplitude) -> float
**Описание:** Compute adaptive threshold for node detection.

### _find_critical_points(self, gradients, threshold)
**Описание:** Find critical points using gradient analysis.

### _is_valid_node(self, envelope, point) -> bool
**Описание:** Check if a point is a valid node using topological criteria.

### classify_nodes(self, envelope)
**Описание:** Classify nodes by type.

Physical Meaning:
    Classifies identified nodes into different types based on
    their local field properties and topological characteristics.

Mathematical Foundation:
   ...

### compute_node_density(self, envelope) -> float
**Описание:** Compute spatial density of nodes.

Physical Meaning:
    Computes the spatial density of nodes in the BVP field,
    providing a measure of field complexity.

Args:
    envelope (np.ndarray): BVP enve...

### compute_topological_charge(self, envelope) -> float
**Описание:** Compute topological charge of the field.

Physical Meaning:
    Computes the total topological charge of the BVP field
    by analyzing the phase structure and winding numbers.

Mathematical Foundatio...

### _is_source_node(self, envelope, node) -> bool
**Описание:** Check if node is a source node using full topological analysis.

Physical Meaning:
    Determines if a node is a source node based on complete
    topological analysis including Hessian matrix and sta...

### _is_sink_node(self, envelope, node) -> bool
**Описание:** Check if node is a sink node using full topological analysis.

Physical Meaning:
    Determines if a node is a sink node based on complete
    topological analysis including Hessian matrix and stabili...

### __repr__(self) -> str
**Описание:** String representation of node analyzer.

## ./bhlff/models/level_b/node_analysis/topological_analysis.py
Methods: 13

### __init__(self, bvp_core)
**Описание:** Initialize topological analyzer.

### is_saddle_node(self, envelope, node) -> bool
**Описание:** Full topological analysis of saddle nodes in 7D.

Physical Meaning:
    Performs complete topological analysis of saddle nodes
    in 7D space-time using full Hessian analysis and
    topological inva...

### _compute_7d_hessian(self, envelope, node)
**Описание:** Compute full 7D Hessian matrix at node.

### _compute_3d_hessian(self, envelope, node)
**Описание:** Compute 3D Hessian matrix at node.

### _extract_7d_neighborhood(self, envelope, node)
**Описание:** Extract 7D neighborhood around node.

### _extract_3d_neighborhood(self, envelope, node)
**Описание:** Extract 3D neighborhood around node.

### _compute_mixed_derivative(self, neighborhood, i, j) -> float
**Описание:** Compute mixed derivative ∂²φ/∂xᵢ∂xⱼ from neighborhood.

### _compute_mixed_derivative_3d(self, neighborhood, i, j) -> float
**Описание:** Compute mixed derivative ∂²φ/∂xᵢ∂xⱼ from 3D neighborhood.

### _compute_topological_index(self, hessian) -> int
**Описание:** Compute topological index from Hessian matrix.

### _apply_morse_theory(self, hessian)
**Описание:** Apply Morse theory to analyze critical point.

### _analyze_stability(self, hessian)
**Описание:** Analyze stability of critical point.

### is_source_node(self, envelope, node) -> bool
**Описание:** Check if node is a source node using full topological analysis.

Physical Meaning:
    Determines if a node is a source node based on complete
    topological analysis including Hessian matrix and sta...

### is_sink_node(self, envelope, node) -> bool
**Описание:** Check if node is a sink node using full topological analysis.

Physical Meaning:
    Determines if a node is a sink node based on complete
    topological analysis including Hessian matrix and stabili...

## ./bhlff/models/level_b/node_analyzer.py
Methods: 14

### __init__(self)
**Описание:** Initialize node analyzer.

### check_spherical_nodes(self, field, center, max_sign_changes)
**Описание:** Check for absence of spherical standing nodes.

Physical Meaning:
    In pure fractional regime (λ=0), the operator symbol D(k) = μk^(2β)
    has no poles, preventing formation of spherical standing w...

### compute_topological_charge(self, field, center, contour_points)
**Описание:** Compute topological charge of the defect.

Physical Meaning:
    The topological charge characterizes the degree of "winding"
    of the phase field around the defect and ensures its
    topological s...

### _compute_radial_profile(self, field, center)
**Описание:** Compute radial profile of the field.

### _count_sign_changes(self, derivative) -> int
**Описание:** Count sign changes in derivative.

### _find_amplitude_zeros(self, amplitude, radius)
**Описание:** Find zeros in amplitude.

### _check_periodicity(self, zeros, tolerance) -> bool
**Описание:** Check for periodicity in zeros.

### _check_monotonicity(self, amplitude, radius) -> bool
**Описание:** Check for monotonic decay.

### _estimate_integration_radius(self, field, center) -> float
**Описание:** Estimate optimal radius for integration.

### _create_spherical_contour(self, center, radius, n_points)
**Описание:** Create spherical contour for integration.

### _compute_phase_gradient(self, phase, field_shape)
**Описание:** Compute phase gradient.

### _integrate_phase_around_contour(self, grad_phase, contour_points) -> float
**Описание:** Integrate phase gradient around contour.

### _interpolate_gradient(self, grad_phase, point)
**Описание:** Interpolate gradient at given point.

### visualize_node_analysis(self, analysis_result, output_path)
**Описание:** Visualize node analysis results.

Physical Meaning:
    Creates visualization of the node analysis showing
    radial profile, derivative, and node detection results.

Args:
    analysis_result (Dict[...

## ./bhlff/models/level_b/power_law/correlation_analysis.py
Methods: 10

### __init__(self, bvp_core)
**Описание:** Initialize correlation analyzer.

### compute_correlation_functions(self, envelope)
**Описание:** Compute full 7D spatial correlation functions.

Physical Meaning:
    Computes the complete 7D spatial correlation function
    C(r) = ⟨a(x)a(x+r)⟩ for all 7 dimensions according to
    the 7D phase f...

### _compute_7d_correlation_function(self, amplitude)
**Описание:** Compute full 7D correlation function preserving dimensional structure.

### _compute_dimension_correlation(self, amplitude, dim)
**Описание:** Compute correlation along a specific dimension.

### _compute_7d_correlation_lengths(self, correlation_7d)
**Описание:** Compute correlation lengths in each dimension.

### _analyze_7d_correlation_structure(self, correlation_7d)
**Описание:** Analyze 7D correlation structure.

### _compute_dimensional_coupling(self, correlation_7d)
**Описание:** Compute coupling between different dimensions.

### _compute_correlation_decay(self, correlation_7d)
**Описание:** Compute correlation decay characteristics.

### _compute_radial_correlation(self, correlation_7d, center)
**Описание:** Compute radial correlation from center point.

### _compute_dimensional_correlations(self, amplitude)
**Описание:** Compute correlations for individual dimensions.

## ./bhlff/models/level_b/power_law/critical_exponents.py
Methods: 17

### __init__(self, bvp_core)
**Описание:** Initialize critical exponents analyzer.

### analyze_critical_behavior(self, envelope)
**Описание:** Analyze critical behavior with full 7D critical exponents.

Physical Meaning:
    Analyzes critical behavior of the BVP field using
    complete 7D critical exponent analysis according to
    the 7D p...

### _compute_full_critical_exponents(self, amplitude)
**Описание:** Compute full set of critical exponents.

### _compute_correlation_length_exponent(self, amplitude) -> float
**Описание:** Compute correlation length exponent ν.

### _compute_order_parameter_exponent(self, amplitude) -> float
**Описание:** Compute order parameter exponent β.

### _compute_susceptibility_exponent(self, amplitude) -> float
**Описание:** Compute susceptibility exponent γ.

### _compute_critical_isotherm_exponent(self, amplitude) -> float
**Описание:** Compute critical isotherm exponent δ.

### _compute_anomalous_dimension(self, amplitude) -> float
**Описание:** Compute anomalous dimension η.

### _compute_specific_heat_exponent(self, amplitude) -> float
**Описание:** Compute specific heat exponent α.

### _compute_dynamic_exponent(self, amplitude) -> float
**Описание:** Compute dynamic exponent z.

### _identify_critical_regions(self, amplitude, critical_exponents)
**Описание:** Identify critical regions with scaling analysis.

### _compute_7d_scaling_dimension(self, critical_exponents) -> float
**Описание:** Compute effective 7D scaling dimension.

### _determine_universality_class(self, critical_exponents) -> str
**Описание:** Determine universality class from critical exponents.

### _compute_critical_scaling_functions(self, amplitude, critical_exponents)
**Описание:** Compute critical scaling functions.

### _compute_correlation_scaling_function(self, amplitude, critical_exponents)
**Описание:** Compute correlation scaling function.

### _compute_susceptibility_scaling_function(self, amplitude, critical_exponents)
**Описание:** Compute susceptibility scaling function.

### _compute_order_parameter_scaling_function(self, amplitude, critical_exponents)
**Описание:** Compute order parameter scaling function.

## ./bhlff/models/level_b/power_law/power_law_core.py
Methods: 6

### __init__(self, bvp_core)
**Описание:** Initialize power law core analyzer.

Args:
    bvp_core (BVPCore): BVP core instance for analysis.

### compute_power_law_exponents(self, envelope)
**Описание:** Compute power law exponents from field distribution.

Physical Meaning:
    Computes power law exponents by analyzing the amplitude
    distribution of the BVP field, identifying scaling behavior
    ...

### identify_scaling_regions(self, envelope)
**Описание:** Identify regions with power law scaling behavior.

Physical Meaning:
    Identifies spatial regions where the BVP field exhibits
    power law scaling behavior, indicating critical regions
    in the ...

### compute_correlation_functions(self, envelope)
**Описание:** Compute full 7D spatial correlation functions.

Physical Meaning:
    Computes the complete 7D spatial correlation function
    C(r) = ⟨a(x)a(x+r)⟩ for all 7 dimensions according to
    the 7D phase f...

### analyze_critical_behavior(self, envelope)
**Описание:** Analyze critical behavior in the field.

Physical Meaning:
    Analyzes critical behavior and phase transitions in the
    BVP field, identifying critical points and scaling behavior.

Mathematical Fo...

### __repr__(self) -> str
**Описание:** String representation of power law core.

## ./bhlff/models/level_b/power_law/scaling_regions.py
Methods: 14

### __init__(self, bvp_core)
**Описание:** Initialize scaling regions analyzer.

### identify_scaling_regions(self, envelope)
**Описание:** Identify scaling regions with full 7D analysis.

Physical Meaning:
    Identifies spatial regions where the BVP field exhibits
    power law scaling behavior using complete 7D analysis
    according t...

### _compute_multiscale_decomposition(self, amplitude)
**Описание:** Compute multi-scale decomposition of the field.

### _downsample_field(self, field, scale)
**Описание:** Downsample field by given scale factor.

### _compute_scale_exponent(self, field) -> float
**Описание:** Compute power law exponent at given scale.

### _compute_wavelet_analysis(self, amplitude)
**Описание:** Compute wavelet analysis for scaling detection.

### _estimate_wavelet_scaling_exponent(self, coeffs, scale) -> float
**Описание:** Estimate scaling exponent from wavelet coefficients.

### _compute_rg_flow(self, amplitude)
**Описание:** Compute renormalization group flow.

### _coarse_grain_field(self, field, step)
**Описание:** Coarse grain field by averaging over blocks.

### _compute_effective_parameters(self, field)
**Описание:** Compute effective parameters after coarse graining.

### _estimate_correlation_length(self, field) -> float
**Описание:** Estimate correlation length from field.

### _compute_flow_direction(self, original, coarse)
**Описание:** Compute RG flow direction.

### _identify_scaling_regions_from_analysis(self, scales, wavelet_coeffs, rg_flow, amplitude)
**Описание:** Identify scaling regions from multi-scale analysis.

### _compute_scaling_consistency(self, exponents) -> float
**Описание:** Compute scaling consistency across scales.

## ./bhlff/models/level_b/power_law_analyzer.py
Methods: 6

### __init__(self)
**Описание:** Initialize power law analyzer.

### analyze_power_law_tail(self, field, beta, center, min_decades)
**Описание:** Analyze power law tail A(r) ∝ r^(2β-3).

Physical Meaning:
    Validates that the phase field exhibits power law decay
    A(r) ∝ r^(2β-3) in homogeneous medium, confirming the
    fundamental behavio...

### _compute_radial_profile(self, field, center)
**Описание:** Compute radial profile of the field.

Physical Meaning:
    Computes the radial profile A(r) by averaging the field
    over spherical shells centered at the defect.

Args:
    field (np.ndarray): 3D ...

### _estimate_core_radius(self, radial_profile) -> float
**Описание:** Estimate core radius from radial profile.

Physical Meaning:
    Estimates the radius of the core region where the field
    amplitude is highest and most coherent.

Args:
    radial_profile (Dict[str...

### visualize_power_law_analysis(self, analysis_result, output_path)
**Описание:** Visualize power law analysis results.

Physical Meaning:
    Creates visualization of the power law analysis showing
    the radial profile, log-log fit, and quality metrics.

Args:
    analysis_resul...

### run_power_law_variations(self, field, center, beta_range)
**Описание:** Run power law analysis for different beta values.

Physical Meaning:
    Analyzes power law behavior for different fractional orders
    β, validating the theoretical relationship A(r) ∝ r^(2β-3).

Ar...

## ./bhlff/models/level_b/visualization.py
Methods: 8

### __init__(self, style)
**Описание:** Initialize Level B visualizer.

Args:
    style (str): Matplotlib style for plots

### create_comprehensive_report(self, results, output_dir)
**Описание:** Create comprehensive visualization report.

Physical Meaning:
    Generates a complete set of visualizations for all Level B
    analysis results, providing a comprehensive view of the
    fundamental...

### _visualize_power_law_analysis(self, result, output_path)
**Описание:** Visualize power law analysis results.

### _visualize_node_analysis(self, result, output_path)
**Описание:** Visualize node analysis results.

### _visualize_topological_analysis(self, result, output_path)
**Описание:** Visualize topological charge analysis results.

### _visualize_zone_analysis(self, result, output_path)
**Описание:** Visualize zone separation analysis results.

### _create_summary_dashboard(self, results, output_path)
**Описание:** Create summary dashboard for all results.

### create_3d_visualization(self, field, center, output_path)
**Описание:** Create 3D visualization of the field.

Physical Meaning:
    Creates 3D visualization of the phase field showing
    the spatial structure and amplitude distribution.

Args:
    field (np.ndarray): 3D...

## ./bhlff/models/level_b/zone_analysis/boundary_detection.py
Methods: 17

### __init__(self, bvp_core)
**Описание:** Initialize boundary detector.

### identify_zone_boundaries(self, envelope)
**Описание:** Identify zone boundaries using full 7D analysis.

Physical Meaning:
    Identifies boundaries between different zones (core, transition, tail)
    using complete 7D analysis according to the 7D theory...

### _compute_level_sets(self, amplitude)
**Описание:** Compute level sets for boundary detection.

### _compute_phase_field_boundaries(self, amplitude)
**Описание:** Compute boundaries using phase field method.

### _analyze_boundary_topology(self, amplitude)
**Описание:** Analyze topology of boundaries.

### _compute_energy_landscape(self, amplitude)
**Описание:** Compute energy landscape for boundary analysis.

### _compute_boundary_length(self, level_set) -> float
**Описание:** Compute boundary length of level set.

### _compute_connectivity(self, level_set)
**Описание:** Compute connectivity properties of level set.

### _compute_phase_field_gradients(self, phase_field)
**Описание:** Compute gradients of phase field.

### _compute_field_gradients(self, field)
**Описание:** Compute gradients of field.

### _compute_curvature(self, field, gradients)
**Описание:** Compute curvature of level sets.

### _identify_critical_points(self, gradients)
**Описание:** Identify critical points where gradients are zero.

### _compute_topological_invariants(self, field, gradients)
**Описание:** Compute topological invariants of the field.

### _compute_energy_density(self, amplitude)
**Описание:** Compute local energy density.

### _compute_energy_gradients(self, energy_density)
**Описание:** Compute gradients of energy density.

### _identify_energy_barriers(self, energy_density, energy_gradients)
**Описание:** Identify energy barriers in the landscape.

### _identify_transition_regions(self, energy_density)
**Описание:** Identify transition regions in energy landscape.

## ./bhlff/models/level_b/zone_analysis/transition_analysis.py
Methods: 9

### __init__(self, bvp_core)
**Описание:** Initialize transition analyzer.

### identify_transition_regions(self, envelope)
**Описание:** Identify transition regions using full 7D analysis.

Physical Meaning:
    Identifies transition regions between different zones
    using complete 7D analysis including level set analysis,
    phase ...

### _identify_gradient_transitions(self, amplitude)
**Описание:** Identify transition regions using gradient analysis.

### _identify_phase_field_transitions(self, phase_field_boundaries)
**Описание:** Identify transition regions using phase field analysis.

### _identify_topological_transitions(self, topological_boundaries)
**Описание:** Identify transition regions using topological analysis.

### _identify_energy_transitions(self, energy_landscape)
**Описание:** Identify transition regions using energy landscape analysis.

### _merge_transition_regions(self, transition_regions)
**Описание:** Merge and filter transition regions.

### _are_regions_nearby(self, region1, region2, tolerance) -> bool
**Описание:** Check if two regions are nearby.

### _merge_nearby_regions(self, regions)
**Описание:** Merge nearby regions into a single region.

## ./bhlff/models/level_b/zone_analysis/zone_analysis.py
Methods: 13

### __init__(self, bvp_core)
**Описание:** Initialize zone analyzer.

Args:
    bvp_core (BVPCore): BVP core instance for analysis.

### identify_zone_boundaries(self, envelope)
**Описание:** Identify boundaries between different zones.

Physical Meaning:
    Identifies boundaries between different zones in the BVP field
    based on field properties and spatial gradients.

Mathematical Fo...

### classify_zones(self, envelope)
**Описание:** Classify spatial zones using full 7D analysis.

Physical Meaning:
    Classifies different spatial zones in the BVP field
    using complete 7D analysis including level set analysis,
    phase field m...

### _compute_adaptive_zone_thresholds(self, amplitude, level_sets)
**Описание:** Compute adaptive zone thresholds using level set analysis.

### _classify_core_zones(self, amplitude, thresholds)
**Описание:** Classify core zones using full analysis.

### _classify_tail_zones(self, amplitude, thresholds)
**Описание:** Classify tail zones using full analysis.

### _classify_transition_zones(self, amplitude, core_mask, tail_mask)
**Описание:** Classify transition zones using full analysis.

### _find_local_maxima(self, amplitude)
**Описание:** Find local maxima in the amplitude field.

### _find_local_minima(self, amplitude)
**Описание:** Find local minima in the amplitude field.

### _compute_coherence_mask(self, amplitude)
**Описание:** Compute coherence mask for zone classification.

### analyze_zone_properties(self, envelope)
**Описание:** Analyze properties of different zones.

Physical Meaning:
    Analyzes properties of different zones in the BVP field
    including amplitude, gradient, and coherence properties.

Mathematical Foundat...

### identify_transition_regions(self, envelope)
**Описание:** Identify transition regions using full 7D analysis.

Physical Meaning:
    Identifies transition regions between different zones
    using complete 7D analysis including level set analysis,
    phase ...

### __repr__(self) -> str
**Описание:** String representation of zone analyzer.

## ./bhlff/models/level_b/zone_analysis/zone_properties.py
Methods: 2

### __init__(self, bvp_core)
**Описание:** Initialize zone properties analyzer.

### analyze_zone_properties(self, envelope)
**Описание:** Analyze properties of different zones.

Physical Meaning:
    Analyzes properties of different zones in the BVP field
    including amplitude, gradient, and coherence properties.

## ./bhlff/models/level_b/zone_analyzer.py
Methods: 12

### __init__(self)
**Описание:** Initialize zone analyzer.

### separate_zones(self, field, center, thresholds)
**Описание:** Separate field into zones (core/transition/tail).

Physical Meaning:
    Quantitatively separates the phase field into three
    characteristic zones based on local indicators, allowing
    analysis o...

### _compute_zone_indicators(self, field, spatial_axes, phase_axes, time_axis)
**Описание:** Compute zone indicators N, S, C.

Physical Meaning:
    Computes local indicators that characterize the properties
    of the phase field and allow quantitative zone separation.

Mathematical Foundati...

### _compute_norm_gradient(self, field, spatial_axes, phase_axes, time_axis)
**Описание:** Compute norm of field gradient across 7D axes.

### _compute_second_derivative(self, field, spatial_axes, phase_axes, time_axis)
**Описание:** Compute magnitude of 7D Laplacian (sum of second derivatives).

### _compute_laplacian(self, field, spatial_axes, phase_axes, time_axis)
**Описание:** Compute 7D Laplacian (sum of second derivatives along all axes).

### _compute_coherence(self, field, spatial_axes, phase_axes, time_axis)
**Описание:** Compute coherence indicator as 7D amplitude gradient norm.

### _compute_zone_radius(self, mask, center) -> float
**Описание:** Compute effective radius of a zone.

### _compute_zone_statistics(self, field, core_mask, transition_mask, tail_mask)
**Описание:** Compute statistics for each zone.

### _assess_zone_separation_quality(self, core_mask, tail_mask, transition_mask, zone_stats)
**Описание:** Assess quality of zone separation.

### visualize_zone_analysis(self, analysis_result, output_path)
**Описание:** Visualize zone analysis results.

Physical Meaning:
    Creates visualization of the zone analysis showing
    zone maps, indicators, and separation quality.

Args:
    analysis_result (Dict[str, Any]...

### run_zone_analysis_variations(self, field, center, threshold_ranges)
**Описание:** Run zone analysis for different threshold values.

Physical Meaning:
    Analyzes zone separation sensitivity to threshold parameters,
    helping to determine optimal separation criteria.

Args:
    ...

## ./bhlff/models/level_b/zone_analyzer_utils.py
Methods: 2

### visualize_zone_analysis(analysis_result, output_path)
**Описание:** Render a 2x2 figure showing zone masks and indicators.

Args:
    analysis_result (Dict[str, Any]): Output of analyzer.separate_zones
    output_path (str): Path to save the figure

### run_zone_analysis_variations(separate_fn, field, center, threshold_ranges)
**Описание:** Execute zone analysis across threshold grids.

Args:
    separate_fn: Callable compatible with analyzer.separate_zones
    field (np.ndarray): Input field
    center (List[float]): Defect center [x, y...

## ./bhlff/models/level_c/abcd_model.py
Methods: 11

### __init__(self, resonators, bvp_core)
**Описание:** Initialize ABCD model.

Physical Meaning:
    Sets up the ABCD model for the given resonator chain,
    computing transmission matrices for each layer and
    preparing for system analysis.

Args:
   ...

### compute_transmission_matrix(self, frequency)
**Описание:** Compute 2x2 transmission matrix for given frequency.

Physical Meaning:
    Computes the overall transmission matrix T_total(ω) for the
    entire resonator chain at frequency ω, representing the
    ...

### find_resonance_conditions(self, frequency_range)
**Описание:** Find frequencies satisfying resonance conditions.

Physical Meaning:
    Finds all frequencies where det(T_total - I) = 0,
    which correspond to system resonance modes.

Mathematical Foundation:
   ...

### find_system_modes(self, frequency_range)
**Описание:** Find system resonance modes.

Physical Meaning:
    Identifies all system resonance modes in the given frequency
    range, computing their frequencies, quality factors, and
    coupling properties.

...

### compute_system_admittance(self, frequency) -> complex
**Описание:** Compute total system admittance.

Physical Meaning:
    Computes the complex admittance Y(ω) = I(ω)/V(ω) of the
    entire resonator chain, representing the system's
    response to external excitatio...

### compare_with_numerical(self, numerical_results)
**Описание:** Compare with numerical simulation results.

Physical Meaning:
    Compares ABCD model predictions with numerical simulation
    results, computing errors and validating the model accuracy.

Mathematic...

### _compute_layer_properties(self)
**Описание:** Compute properties for each layer.

### _compute_layer_matrix(self, layer, frequency)
**Описание:** Compute transmission matrix for single layer.

Physical Meaning:
    Computes the 2x2 transmission matrix for a single
    resonator layer at frequency ω.

Mathematical Foundation:
    For a layer wit...

### _compute_quality_factor(self, frequency) -> float
**Описание:** Compute quality factor for given frequency.

Physical Meaning:
    Computes the quality factor Q = ω / (2 * Im(ω)) which
    characterizes the resonance sharpness and energy storage.

### _compute_mode_amplitude_phase(self, frequency)
**Описание:** Compute mode amplitude and phase.

Physical Meaning:
    Computes the amplitude and phase of the resonance mode
    at the given frequency from eigenvector analysis.

### _compute_coupling_strength(self, frequency, all_frequencies) -> float
**Описание:** Compute coupling strength with other modes.

Physical Meaning:
    Computes the coupling strength between the mode at the
    given frequency and other system modes.

## ./bhlff/models/level_c/beating/basic/beating_basic_comparison.py
Methods: 6

### __init__(self, bvp_core)
**Описание:** Initialize comparison analyzer.

### compare_analyses(self, results1, results2)
**Описание:** Compare two beating analysis results.

Physical Meaning:
    Compares two sets of beating analysis results to
    identify differences, similarities, and consistency.

Args:
    results1 (Dict[str, An...

### _compare_beating_frequencies(self, freq1, freq2)
**Описание:** Compare beating frequencies between two analyses.

Physical Meaning:
    Compares beating frequencies to identify
    similarities and differences in frequency content.

Args:
    freq1 (List[float]):...

### _compare_interference_patterns(self, patterns1, patterns2)
**Описание:** Compare interference patterns between two analyses.

Physical Meaning:
    Compares interference patterns to identify
    similarities and differences in pattern characteristics.

Args:
    patterns1 ...

### _compare_mode_coupling(self, coupling1, coupling2)
**Описание:** Compare mode coupling between two analyses.

Physical Meaning:
    Compares mode coupling characteristics to identify
    similarities and differences in coupling behavior.

Args:
    coupling1 (Dict[...

### _compute_overall_comparison(self, comparison_results)
**Описание:** Compute overall comparison metrics.

Physical Meaning:
    Computes overall comparison metrics from individual
    comparison results to provide a summary assessment.

Args:
    comparison_results (Di...

## ./bhlff/models/level_c/beating/basic/beating_basic_core.py
Methods: 25

### __init__(self, bvp_core)
**Описание:** Нет докстринга

### analyze_beating_comprehensive(self, envelope)
**Описание:** Comprehensive beating analysis according to theoretical framework.

Physical Meaning:
    Performs full theoretical analysis of mode beating
    according to the 7D phase field theory, including
    i...

### analyze_beating_statistical(self, envelope)
**Описание:** Analyze mode beating with statistical analysis.

Physical Meaning:
    Analyzes mode beating using statistical methods
    for comprehensive understanding of beating patterns.

Args:
    envelope (np....

### compare_beating_analyses(self, results1, results2)
**Описание:** Compare two beating analysis results.

Physical Meaning:
    Compares two sets of beating analysis results to
    identify differences, similarities, and consistency.

Args:
    results1 (Dict[str, An...

### optimize_beating_parameters(self, envelope)
**Описание:** Optimize beating analysis parameters.

Physical Meaning:
    Optimizes parameters used in beating analysis
    to improve accuracy and reliability.

Args:
    envelope (np.ndarray): 7D envelope field ...

### _analyze_interference_theoretical(self, envelope)
**Описание:** Analyze interference patterns using theoretical framework.

Physical Meaning:
    Analyzes mode interference patterns according to the
    7D phase field theory, detecting characteristic
    interfere...

### _analyze_mode_coupling_theoretical(self, envelope)
**Описание:** Analyze mode coupling using theoretical framework.

Physical Meaning:
    Analyzes mode coupling strength and mechanisms
    according to the 7D phase field theory.

Mathematical Foundation:
    Coupl...

### _analyze_phase_coherence_theoretical(self, envelope)
**Описание:** Analyze phase coherence using theoretical framework.

Physical Meaning:
    Analyzes phase coherence between different mode
    components according to the 7D phase field theory.

Mathematical Foundat...

### analyze_interference_theoretical(self, envelope)
**Описание:** Нет докстринга

### analyze_mode_coupling_theoretical(self, envelope)
**Описание:** Нет докстринга

### analyze_phase_coherence_theoretical(self, envelope)
**Описание:** Нет докстринга

### calculate_beating_frequencies_theoretical(self, envelope)
**Описание:** Нет докстринга

### _calculate_beating_frequencies_theoretical(self, envelope)
**Описание:** Calculate beating frequencies using theoretical framework.

Physical Meaning:
    Calculates theoretical beating frequencies based on
    mode frequency differences according to the 7D phase field the...

### _validate_theoretical_consistency(self, envelope, analysis_results)
**Описание:** Validate theoretical consistency of analysis results.

Physical Meaning:
    Validates that analysis results are consistent with
    the 7D phase field theory predictions.

Args:
    envelope (np.ndar...

### _detect_spatial_interference_patterns(self, envelope_complex)
**Описание:** Detect spatial interference patterns.

Physical Meaning:
    Detects spatial interference patterns in the complex
    envelope field using theoretical criteria.

### _decompose_mode_components(self, envelope)
**Описание:** Decompose envelope into mode components.

Physical Meaning:
    Decomposes the envelope field into individual
    mode components using theoretical decomposition.

### _calculate_coupling_matrix(self, mode_components)
**Описание:** Calculate coupling matrix between mode components.

Physical Meaning:
    Calculates the coupling matrix elements between
    different mode components according to the theory.

### _calculate_phase_coherence(self, phase_field) -> float
**Описание:** Calculate phase coherence.

Physical Meaning:
    Calculates the phase coherence measure according
    to the theoretical definition.

### _analyze_phase_stability(self, phase_field) -> float
**Описание:** Analyze phase stability.

Physical Meaning:
    Analyzes the stability of phase variations
    in the field.

### _calculate_phase_correlation(self, phase_field) -> float
**Описание:** Calculate phase correlation.

Physical Meaning:
    Calculates the correlation between phase
    variations in different spatial regions.

### _analyze_beating_patterns(self, envelope, beating_frequencies)
**Описание:** Analyze beating patterns.

Physical Meaning:
    Analyzes the characteristic beating patterns
    in the envelope field.

### optimize_analysis(self, envelope, results)
**Описание:** Нет докстринга

### optimize_parameters(self, envelope, params)
**Описание:** Нет докстринга

### perform_statistical_analysis(self, envelope, results)
**Описание:** Нет докстринга

### compare_analyses(self, results1, results2)
**Описание:** Нет докстринга

## ./bhlff/models/level_c/beating/basic/beating_basic_optimization.py
Methods: 9

### __init__(self, bvp_core)
**Описание:** Initialize optimization analyzer.

### optimize_analysis(self, envelope, initial_results)
**Описание:** Optimize beating analysis results.

Physical Meaning:
    Optimizes beating analysis results using iterative
    refinement techniques.

Args:
    envelope (np.ndarray): 7D envelope field data.
    in...

### optimize_parameters(self, envelope, initial_params)
**Описание:** Optimize analysis parameters.

Physical Meaning:
    Optimizes parameters used in beating analysis
    to improve accuracy and reliability.

Args:
    envelope (np.ndarray): 7D envelope field data.
  ...

### validate_optimization(self, envelope, initial_params, optimized_params)
**Описание:** Validate optimization results.

Physical Meaning:
    Validates that optimization improves analysis
    performance compared to initial parameters.

Args:
    envelope (np.ndarray): 7D envelope field ...

### _optimize_frequencies(self, frequencies) -> list
**Описание:** Optimize beating frequencies.

### _optimize_coupling(self, coupling)
**Описание:** Optimize mode coupling analysis.

### _calculate_performance(self, envelope, params) -> float
**Описание:** Calculate performance metric for given parameters.

### _adjust_parameters(self, params, performance)
**Описание:** Adjust parameters based on performance.

### _check_convergence(self, current_params, initial_params) -> bool
**Описание:** Check if optimization has converged.

## ./bhlff/models/level_c/beating/basic/beating_basic_statistics.py
Methods: 9

### __init__(self, bvp_core)
**Описание:** Initialize statistics analyzer.

### perform_statistical_analysis(self, envelope, basic_results)
**Описание:** Perform statistical analysis on beating data.

Physical Meaning:
    Performs statistical analysis on beating data to
    provide comprehensive understanding of patterns.

Args:
    envelope (np.ndarr...

### _analyze_frequency_statistics(self, frequencies)
**Описание:** Analyze statistical properties of beating frequencies.

### _analyze_coupling_statistics(self, coupling)
**Описание:** Analyze statistical properties of mode coupling.

### _analyze_envelope_statistics(self, envelope)
**Описание:** Analyze statistical properties of the envelope field.

### _perform_hypothesis_testing(self, envelope, basic_results)
**Описание:** Perform hypothesis testing on beating data.

### _calculate_skewness(self, data) -> float
**Описание:** Calculate skewness of the data.

### _calculate_kurtosis(self, data) -> float
**Описание:** Calculate kurtosis of the data.

### calculate_confidence_intervals(self, data, confidence_level)
**Описание:** Calculate confidence intervals for statistical measures.

## ./bhlff/models/level_c/beating/beating_core_basic.py
Methods: 25

### __init__(self, bvp_core)
**Описание:** Initialize beating core analyzer.

Physical Meaning:
    Sets up the analyzer with the BVP core for accessing
    field data and computational resources.

Args:
    bvp_core (BVPCore): BVP core instan...

### analyze_beating(self, envelope)
**Описание:** Analyze mode beating in the envelope field.

Physical Meaning:
    Analyzes mode beating patterns in the 7D envelope field,
    identifying interference patterns and beating frequencies
    that indic...

### _analyze_frequency_domain(self, envelope)
**Описание:** Analyze frequency domain characteristics.

Physical Meaning:
    Performs FFT analysis to identify frequency components
    in the envelope field, which are essential for detecting
    beating pattern...

### _detect_interference_patterns(self, envelope)
**Описание:** Detect interference patterns in the envelope field.

Physical Meaning:
    Detects interference patterns that indicate mode beating,
    including spatial and temporal interference effects
    in the ...

### _calculate_beating_frequencies(self, frequency_analysis)
**Описание:** Calculate beating frequencies from frequency analysis.

Physical Meaning:
    Calculates beating frequencies by analyzing frequency
    differences between dominant modes, which represent
    the char...

### _analyze_mode_coupling(self, envelope, beating_frequencies)
**Описание:** Analyze mode coupling effects.

Physical Meaning:
    Analyzes mode coupling effects that give rise to beating,
    including coupling strength, coupling mechanisms, and
    mode interaction patterns....

### _calculate_beating_strength(self, envelope, beating_frequencies) -> float
**Описание:** Calculate the strength of beating effects.

Physical Meaning:
    Calculates the overall strength of beating effects
    in the envelope field, providing a quantitative
    measure of mode interaction...

### _find_dominant_frequencies(self, power_spectrum)
**Описание:** Find dominant frequencies in the power spectrum.

### _calculate_frequency_statistics(self, power_spectrum)
**Описание:** Calculate frequency statistics.

### _analyze_spatial_interference(self, envelope)
**Описание:** Analyze spatial interference patterns.

### _analyze_temporal_interference(self, envelope)
**Описание:** Analyze temporal interference patterns.

### _analyze_phase_interference(self, envelope)
**Описание:** Analyze phase interference patterns.

### _calculate_coupling_strength(self, envelope, beating_frequencies) -> float
**Описание:** Calculate mode coupling strength.

### _identify_coupling_mechanisms(self, envelope)
**Описание:** Identify coupling mechanisms.

### _analyze_mode_interactions(self, envelope, beating_frequencies)
**Описание:** Analyze mode interactions.

### _get_frequency_power(self, power_spectrum, frequency) -> float
**Описание:** Get power at specific frequency.

### _find_peaks(self, data)
**Описание:** Find peaks in data array.

### _index_to_frequency(self, index, shape) -> float
**Описание:** Convert array index to frequency.

### _frequency_to_index(self, frequency, shape) -> int
**Описание:** Convert frequency to array index.

### _calculate_spatial_correlation(self, envelope)
**Описание:** Calculate spatial correlation.

### _calculate_temporal_correlation(self, envelope)
**Описание:** Calculate temporal correlation.

### _calculate_phase_correlation(self, envelope)
**Описание:** Calculate phase correlation.

### _has_nonlinear_coupling(self, envelope) -> bool
**Описание:** Check for nonlinear coupling.

### _has_resonant_coupling(self, envelope) -> bool
**Описание:** Check for resonant coupling.

### _has_parametric_coupling(self, envelope) -> bool
**Описание:** Check for parametric coupling.

## ./bhlff/models/level_c/beating/beating_correlation.py
Methods: 8

### __init__(self, bvp_core)
**Описание:** Initialize beating correlation analyzer.

Args:
    bvp_core (BVPCore): BVP core instance for field access.

### calculate_correlation_analysis(self, envelope)
**Описание:** Calculate correlation analysis of the envelope field.

Physical Meaning:
    Calculates correlation measures between different
    parts of the envelope field to identify beating patterns.

Args:
    ...

### _calculate_autocorrelation(self, envelope)
**Описание:** Calculate autocorrelation of the envelope field.

Physical Meaning:
    Calculates the autocorrelation function to identify
    periodic patterns and beating frequencies.

Args:
    envelope (np.ndarr...

### _calculate_cross_correlation(self, envelope)
**Описание:** Calculate cross-correlation between different field components.

Physical Meaning:
    Calculates cross-correlation between different
    components of the envelope field.

Args:
    envelope (np.ndar...

### _calculate_correlation_statistics(self, autocorrelation, cross_correlation)
**Описание:** Calculate correlation statistics.

Physical Meaning:
    Calculates statistical measures of the correlation
    functions for analysis.

Args:
    autocorrelation (np.ndarray): Autocorrelation functio...

### calculate_variance_analysis(self, envelope)
**Описание:** Calculate variance analysis of the envelope field.

Physical Meaning:
    Calculates variance measures to identify regions
    of high variability that may indicate beating patterns.

Args:
    envelo...

### _calculate_local_variance(self, envelope)
**Описание:** Calculate local variance of the envelope field.

Physical Meaning:
    Calculates local variance to identify regions
    of high variability.

Args:
    envelope (np.ndarray): 7D envelope field data.
...

### _calculate_variance_statistics(self, overall_variance, local_variance)
**Описание:** Calculate variance statistics.

Physical Meaning:
    Calculates statistical measures of the variance
    for analysis.

Args:
    overall_variance (float): Overall variance.
    local_variance (np.nd...

## ./bhlff/models/level_c/beating/beating_patterns.py
Methods: 11

### __init__(self, bvp_core)
**Описание:** Initialize beating pattern detector.

Args:
    bvp_core (BVPCore): BVP core instance for field access.

### calculate_pattern_detection(self, envelope)
**Описание:** Calculate pattern detection in the envelope field.

Physical Meaning:
    Detects patterns in the envelope field that may
    indicate beating phenomena.

Args:
    envelope (np.ndarray): 7D envelope ...

### _detect_temporal_patterns(self, envelope)
**Описание:** Detect temporal patterns in the envelope field.

Physical Meaning:
    Detects temporal patterns that may indicate
    beating phenomena.

Args:
    envelope (np.ndarray): 7D envelope field data.

Ret...

### _detect_spatial_patterns(self, envelope)
**Описание:** Detect spatial patterns in the envelope field.

Physical Meaning:
    Detects spatial patterns that may indicate
    beating phenomena.

Args:
    envelope (np.ndarray): 7D envelope field data.

Retur...

### _detect_phase_patterns(self, envelope)
**Описание:** Detect phase patterns in the envelope field.

Physical Meaning:
    Detects phase patterns that may indicate
    beating phenomena.

Args:
    envelope (np.ndarray): 7D envelope field data.

Returns:
...

### _detect_pattern_in_data(self, data)
**Описание:** Detect patterns in a data array.

Physical Meaning:
    Detects patterns in the data that may indicate
    beating phenomena.

Args:
    data (np.ndarray): Input data array.

Returns:
    Dict[str, An...

### _calculate_pattern_statistics(self, temporal_patterns, spatial_patterns, phase_patterns)
**Описание:** Calculate pattern statistics.

Physical Meaning:
    Calculates statistical measures of the detected
    patterns for analysis.

Args:
    temporal_patterns (List[Dict[str, Any]]): Temporal patterns.
...

### calculate_statistical_measures(self, envelope)
**Описание:** Calculate statistical measures of the envelope field.

Physical Meaning:
    Calculates various statistical measures to characterize
    the envelope field and its beating properties.

Args:
    envel...

### _calculate_basic_statistics(self, envelope)
**Описание:** Calculate basic statistics of the envelope field.

Physical Meaning:
    Calculates basic statistical measures to characterize
    the envelope field.

Args:
    envelope (np.ndarray): 7D envelope fie...

### _calculate_higher_moments(self, envelope)
**Описание:** Calculate higher-order moments of the envelope field.

Physical Meaning:
    Calculates higher-order moments to characterize
    the distribution of the envelope field.

Args:
    envelope (np.ndarray...

### _calculate_spectral_statistics(self, envelope)
**Описание:** Calculate spectral statistics of the envelope field.

Physical Meaning:
    Calculates spectral statistics to characterize
    the frequency content of the envelope field.

Args:
    envelope (np.ndar...

## ./bhlff/models/level_c/beating/beating_spectrum.py
Methods: 11

### __init__(self, bvp_core)
**Описание:** Initialize beating spectrum analyzer.

Args:
    bvp_core (BVPCore): BVP core instance for field access.

### calculate_beating_spectrum(self, envelope)
**Описание:** Calculate beating spectrum from envelope field.

Physical Meaning:
    Calculates the beating spectrum by analyzing frequency
    differences and interference patterns in the envelope field.

Args:
  ...

### _calculate_frequency_spectrum(self, envelope)
**Описание:** Calculate frequency spectrum of the envelope field.

Physical Meaning:
    Calculates the frequency spectrum using FFT analysis
    to identify dominant frequencies and their amplitudes.

Args:
    en...

### _calculate_beating_frequencies(self, frequency_spectrum)
**Описание:** Calculate beating frequencies from frequency spectrum.

Physical Meaning:
    Calculates beating frequencies as differences between
    dominant frequencies, representing mode interactions.

Args:
   ...

### _calculate_interference_patterns(self, envelope, beating_frequencies)
**Описание:** Calculate interference patterns from beating frequencies.

Physical Meaning:
    Calculates interference patterns that result from
    beating between different frequency modes.

Args:
    envelope (n...

### _calculate_spectrum_statistics(self, frequency_spectrum, beating_frequencies)
**Описание:** Calculate spectrum statistics.

Physical Meaning:
    Calculates statistical measures of the frequency spectrum
    and beating frequencies for analysis.

Args:
    frequency_spectrum (Dict[str, Any])...

### _find_dominant_frequencies(self, frequency_magnitudes)
**Описание:** Find dominant frequencies in the spectrum.

Physical Meaning:
    Identifies the most significant frequencies in the
    spectrum based on their amplitudes.

Args:
    frequency_magnitudes (np.ndarray...

### _find_peaks(self, data)
**Описание:** Find peaks in the data.

Physical Meaning:
    Identifies local maxima in the data that represent
    significant frequency components.

Args:
    data (np.ndarray): Input data array.

Returns:
    Li...

### _calculate_frequency_statistics(self, frequency_magnitudes)
**Описание:** Calculate frequency statistics.

Physical Meaning:
    Calculates statistical measures of the frequency
    magnitudes for analysis.

Args:
    frequency_magnitudes (np.ndarray): Frequency magnitude s...

### _calculate_single_interference_pattern(self, envelope, frequency, amplitude)
**Описание:** Calculate interference pattern for a single beating frequency.

Physical Meaning:
    Calculates the interference pattern that results from
    beating at a specific frequency.

Args:
    envelope (np...

### _calculate_interference_statistics(self, interference_patterns)
**Описание:** Calculate interference pattern statistics.

Physical Meaning:
    Calculates statistical measures of the interference
    patterns for analysis.

Args:
    interference_patterns (Dict[str, Any]): Inte...

## ./bhlff/models/level_c/beating/ml/beating_ml_core.py
Methods: 4

### __init__(self, bvp_core)
**Описание:** Initialize machine learning beating core analyzer.

Args:
    bvp_core (BVPCore): BVP core instance for field access.

### analyze_beating_machine_learning(self, envelope)
**Описание:** Analyze mode beating using machine learning techniques.

Physical Meaning:
    Analyzes mode beating using machine learning methods
    for advanced pattern recognition and classification.

Args:
    ...

### _perform_machine_learning_analysis(self, envelope, basic_results)
**Описание:** Perform machine learning analysis on beating data.

Physical Meaning:
    Performs comprehensive machine learning analysis including
    pattern classification, frequency prediction, and coupling opti...

### _analyze_beating_basic(self, envelope)
**Описание:** Perform basic beating analysis.

Physical Meaning:
    Performs basic analysis of mode beating patterns
    without machine learning techniques.

Args:
    envelope (np.ndarray): 7D envelope field dat...

## ./bhlff/models/level_c/beating/ml/beating_ml_optimization.py
Methods: 15

### __init__(self, bvp_core)
**Описание:** Initialize optimization analyzer.

Args:
    bvp_core (BVPCore): BVP core instance for field access.

### optimize_ml_parameters(self, envelope)
**Описание:** Optimize machine learning parameters.

Physical Meaning:
    Optimizes machine learning parameters to improve
    the accuracy and reliability of ML-based analysis.

Args:
    envelope (np.ndarray): 7...

### _optimize_ml_parameters(self, envelope, initial_params)
**Описание:** Optimize machine learning parameters using iterative methods.

Physical Meaning:
    Uses iterative optimization methods to find optimal ML parameters
    that maximize analysis accuracy and reliabili...

### _validate_ml_optimization(self, envelope, initial_params, optimized_params)
**Описание:** Validate machine learning parameter optimization.

Physical Meaning:
    Validates that the optimized parameters improve analysis performance
    compared to the initial parameters.

Args:
    envelop...

### _calculate_ml_performance(self, envelope, params) -> float
**Описание:** Calculate machine learning performance metric.

Physical Meaning:
    Calculates a performance metric for the ML analysis based on
    the current parameter values.

Args:
    envelope (np.ndarray): 7...

### _adjust_parameters(self, params, performance)
**Описание:** Adjust parameters based on current performance.

Physical Meaning:
    Adjusts ML parameters to improve performance based on
    the current performance metric.

Args:
    params (Dict[str, float]): C...

### _check_convergence(self, current_params, initial_params) -> bool
**Описание:** Check if parameter optimization has converged.

Physical Meaning:
    Checks if the parameter optimization process has converged
    to a stable solution.

Args:
    current_params (Dict[str, float]):...

### optimize_classification_parameters(self, envelope)
**Описание:** Optimize classification-specific parameters.

Physical Meaning:
    Optimizes parameters specifically for pattern classification
    tasks in beating analysis.

Args:
    envelope (np.ndarray): 7D env...

### optimize_prediction_parameters(self, envelope)
**Описание:** Optimize prediction-specific parameters.

Physical Meaning:
    Optimizes parameters specifically for frequency and coupling
    prediction tasks in beating analysis.

Args:
    envelope (np.ndarray):...

### _optimize_classification_parameters(self, envelope, initial_params)
**Описание:** Optimize classification-specific parameters.

### _optimize_prediction_parameters(self, envelope, initial_params)
**Описание:** Optimize prediction-specific parameters.

### _validate_classification_optimization(self, envelope, initial_params, optimized_params)
**Описание:** Validate classification parameter optimization.

### _validate_prediction_optimization(self, envelope, initial_params, optimized_params)
**Описание:** Validate prediction parameter optimization.

### _calculate_classification_performance(self, envelope, params) -> float
**Описание:** Calculate classification performance metric.

### _calculate_prediction_performance(self, envelope, params) -> float
**Описание:** Calculate prediction performance metric.

## ./bhlff/models/level_c/beating/ml/beating_ml_patterns.py
Methods: 8

### __init__(self, bvp_core)
**Описание:** Initialize pattern classification analyzer.

Args:
    bvp_core (BVPCore): BVP core instance for field access.

### classify_beating_patterns(self, envelope)
**Описание:** Classify beating patterns using machine learning.

Physical Meaning:
    Classifies beating patterns in the envelope field
    using machine learning techniques for pattern recognition.

Args:
    env...

### _extract_pattern_features(self, envelope)
**Описание:** Extract features for pattern classification.

Physical Meaning:
    Extracts relevant features from the envelope field
    for machine learning-based pattern classification.

Args:
    envelope (np.nd...

### _classify_patterns_ml(self, features)
**Описание:** Classify patterns using machine learning.

Physical Meaning:
    Uses machine learning algorithms to classify beating patterns
    based on extracted features.

Args:
    features (Dict[str, Any]): Ex...

### _classify_patterns_simple(self, features)
**Описание:** Classify patterns using simple heuristics.

Physical Meaning:
    Uses simple heuristic methods to classify beating patterns
    when machine learning is not available.

Args:
    features (Dict[str, ...

### _calculate_symmetry_score(self, envelope) -> float
**Описание:** Calculate symmetry score of the envelope field.

### _calculate_regularity_score(self, envelope) -> float
**Описание:** Calculate regularity score of the envelope field.

### _calculate_complexity_score(self, envelope) -> float
**Описание:** Calculate complexity score of the envelope field.

## ./bhlff/models/level_c/beating/ml/beating_ml_prediction.py
Methods: 21

### __init__(self, bvp_core)
**Описание:** Initialize prediction analyzer.

Args:
    bvp_core (BVPCore): BVP core instance for field access.

### predict_beating_frequencies(self, envelope)
**Описание:** Predict beating frequencies using machine learning.

Physical Meaning:
    Predicts beating frequencies in the envelope field
    using machine learning techniques for frequency analysis.

Args:
    e...

### predict_mode_coupling(self, envelope)
**Описание:** Predict mode coupling using machine learning.

Physical Meaning:
    Predicts mode coupling effects in the envelope field
    using machine learning techniques for coupling analysis.

Args:
    envelo...

### _extract_frequency_features(self, envelope)
**Описание:** Extract features for frequency prediction.

Physical Meaning:
    Extracts relevant features from the envelope field
    for machine learning-based frequency prediction.

Args:
    envelope (np.ndarra...

### _extract_coupling_features(self, envelope)
**Описание:** Extract features for coupling prediction.

Physical Meaning:
    Extracts relevant features from the envelope field
    for machine learning-based mode coupling prediction.

Args:
    envelope (np.nda...

### _predict_frequencies_ml(self, features)
**Описание:** Predict frequencies using machine learning.

Physical Meaning:
    Uses machine learning algorithms to predict beating frequencies
    based on extracted features.

Args:
    features (Dict[str, Any])...

### _predict_frequencies_simple(self, features)
**Описание:** Predict frequencies using simple heuristics.

Physical Meaning:
    Uses simple heuristic methods to predict beating frequencies
    when machine learning is not available.

Args:
    features (Dict[s...

### _predict_coupling_ml(self, features)
**Описание:** Predict coupling using machine learning.

Physical Meaning:
    Uses machine learning algorithms to predict mode coupling
    based on extracted features.

Args:
    features (Dict[str, Any]): Extract...

### _predict_coupling_simple(self, features)
**Описание:** Predict coupling using simple heuristics.

Physical Meaning:
    Uses simple heuristic methods to predict mode coupling
    when machine learning is not available.

Args:
    features (Dict[str, Any])...

### _calculate_spectral_entropy(self, spectrum) -> float
**Описание:** Calculate spectral entropy.

### _calculate_frequency_spacing(self, indices, shape) -> float
**Описание:** Calculate average frequency spacing.

### _calculate_frequency_bandwidth(self, spectrum) -> float
**Описание:** Calculate frequency bandwidth.

### _calculate_autocorrelation(self, envelope) -> float
**Описание:** Calculate envelope autocorrelation.

### _calculate_laplacian(self, envelope)
**Описание:** Calculate Laplacian of the envelope.

### _calculate_spatial_correlation(self, envelope) -> float
**Описание:** Calculate spatial correlation.

### _calculate_frequency_coupling_strength(self, spectrum) -> float
**Описание:** Calculate frequency coupling strength.

### _calculate_mode_interaction_energy(self, spectrum) -> float
**Описание:** Calculate mode interaction energy.

### _calculate_coupling_symmetry(self, spectrum) -> float
**Описание:** Calculate coupling symmetry.

### _calculate_nonlinear_strength(self, envelope) -> float
**Описание:** Calculate nonlinear strength.

### _calculate_mode_mixing_degree(self, envelope) -> float
**Описание:** Calculate mode mixing degree.

### _calculate_coupling_efficiency(self, envelope) -> float
**Описание:** Calculate coupling efficiency.

## ./bhlff/models/level_c/beating/optimization/beating_validation_accuracy_optimization.py
Methods: 5

### __init__(self, bvp_core)
**Описание:** Initialize accuracy optimization analyzer.

### optimize_accuracy(self, results, initial_accuracy)
**Описание:** Optimize validation accuracy.

Physical Meaning:
    Optimizes validation accuracy to improve reliability
    of validation.

Args:
    results (Dict[str, Any]): Analysis results.
    initial_accuracy...

### validate_optimization(self, results, initial_accuracy, optimized_accuracy)
**Описание:** Validate accuracy optimization.

Physical Meaning:
    Validates that accuracy optimization improves
    validation reliability.

Args:
    results (Dict[str, Any]): Analysis results.
    initial_accu...

### _assess_complexity(self, results) -> float
**Описание:** Assess complexity of analysis results.

### _calculate_accuracy_score(self, results, accuracy_params) -> float
**Описание:** Calculate accuracy score for given parameters.

## ./bhlff/models/level_c/beating/optimization/beating_validation_optimization_core.py
Methods: 6

### __init__(self, bvp_core)
**Описание:** Initialize optimization-based beating validation analyzer.

Args:
    bvp_core (BVPCore): BVP core instance for field access.

### optimize_validation_parameters(self, results)
**Описание:** Optimize validation parameters for beating analysis.

Physical Meaning:
    Optimizes validation parameters to improve the accuracy
    and efficiency of beating analysis validation.

Args:
    result...

### optimize_validation_process(self, results)
**Описание:** Optimize validation process for beating analysis.

Physical Meaning:
    Optimizes the validation process to improve efficiency
    and accuracy of beating analysis validation.

Args:
    results (Dic...

### optimize_validation_accuracy(self, results)
**Описание:** Optimize validation accuracy for beating analysis.

Physical Meaning:
    Optimizes validation accuracy to improve the reliability
    of beating analysis validation.

Args:
    results (Dict[str, Any...

### optimize_validation_efficiency(self, results)
**Описание:** Optimize validation efficiency for beating analysis.

Physical Meaning:
    Optimizes validation efficiency to improve the speed
    and resource usage of beating analysis validation.

Args:
    resul...

### _assess_result_complexity(self, results) -> float
**Описание:** Assess complexity of analysis results.

Physical Meaning:
    Assesses the complexity of analysis results to determine
    appropriate optimization strategies.

Args:
    results (Dict[str, Any]): Ana...

## ./bhlff/models/level_c/beating/optimization/beating_validation_parameter_optimization.py
Methods: 6

### __init__(self, bvp_core)
**Описание:** Initialize parameter optimization analyzer.

### optimize_parameters(self, results, initial_params)
**Описание:** Optimize validation parameters.

Physical Meaning:
    Optimizes validation parameters to improve accuracy
    and efficiency.

Args:
    results (Dict[str, Any]): Analysis results.
    initial_params...

### validate_optimization(self, results, initial_params, optimized_params)
**Описание:** Validate parameter optimization.

Physical Meaning:
    Validates that parameter optimization improves
    validation performance.

Args:
    results (Dict[str, Any]): Analysis results.
    initial_pa...

### _calculate_performance(self, results, params) -> float
**Описание:** Calculate performance metric for given parameters.

### _adjust_parameters(self, params, performance)
**Описание:** Adjust parameters based on performance.

### _check_convergence(self, current_params, initial_params) -> bool
**Описание:** Check if optimization has converged.

## ./bhlff/models/level_c/beating/optimization/beating_validation_process_optimization.py
Methods: 8

### __init__(self, bvp_core)
**Описание:** Initialize process optimization analyzer.

### optimize_process(self, results, initial_process)
**Описание:** Optimize validation process.

Physical Meaning:
    Optimizes the validation process to improve efficiency
    and accuracy.

Args:
    results (Dict[str, Any]): Analysis results.
    initial_process ...

### optimize_efficiency(self, results, initial_efficiency)
**Описание:** Optimize validation efficiency.

Physical Meaning:
    Optimizes validation efficiency to improve speed
    and resource usage.

Args:
    results (Dict[str, Any]): Analysis results.
    initial_effic...

### validate_optimization(self, results, initial_process, optimized_process)
**Описание:** Validate process optimization.

Physical Meaning:
    Validates that process optimization improves
    validation performance.

Args:
    results (Dict[str, Any]): Analysis results.
    initial_proces...

### validate_efficiency_optimization(self, results, initial_efficiency, optimized_efficiency)
**Описание:** Validate efficiency optimization.

Physical Meaning:
    Validates that efficiency optimization improves
    validation performance.

Args:
    results (Dict[str, Any]): Analysis results.
    initial_...

### _assess_complexity(self, results) -> float
**Описание:** Assess complexity of analysis results.

### _calculate_efficiency(self, process) -> float
**Описание:** Calculate efficiency metric for process configuration.

### _calculate_performance(self, efficiency) -> float
**Описание:** Calculate performance metric for efficiency configuration.

## ./bhlff/models/level_c/beating/validation/beating_validation_comparison.py
Methods: 6

### __init__(self, bvp_core)
**Описание:** Initialize comparison validation analyzer.

### compare_results(self, results1, results2)
**Описание:** Compare two sets of beating analysis results.

Physical Meaning:
    Compares two sets of beating analysis results to
    identify differences and similarities.

Args:
    results1 (Dict[str, Any]): F...

### _compare_beating_frequencies(self, freq1, freq2)
**Описание:** Compare beating frequencies between two analyses.

### _compare_interference_patterns(self, patterns1, patterns2)
**Описание:** Compare interference patterns between two analyses.

### _compare_mode_coupling(self, coupling1, coupling2)
**Описание:** Compare mode coupling between two analyses.

### _compute_overall_comparison(self, comparison_results)
**Описание:** Compute overall comparison metrics.

## ./bhlff/models/level_c/beating/validation/beating_validation_core.py
Methods: 8

### __init__(self, bvp_core)
**Описание:** Initialize beating validation analyzer.

Args:
    bvp_core (BVPCore): BVP core instance for field access.

### validate_with_statistics(self, results)
**Описание:** Validate beating analysis results with statistical analysis.

Physical Meaning:
    Performs comprehensive statistical validation of beating
    analysis results to ensure their reliability and accura...

### compare_analysis_results(self, results1, results2)
**Описание:** Compare two sets of beating analysis results.

Physical Meaning:
    Compares two sets of beating analysis results to
    identify differences, similarities, and consistency.

Args:
    results1 (Dict...

### validate_analysis_consistency(self, results)
**Описание:** Validate consistency of beating analysis results.

Physical Meaning:
    Validates the internal consistency of beating analysis
    results to ensure they are physically reasonable.

Args:
    results...

### _validate_beating_frequencies(self, frequencies)
**Описание:** Validate beating frequencies.

Physical Meaning:
    Validates beating frequencies for physical reasonableness
    and statistical significance.

Args:
    frequencies (List[float]): List of beating f...

### _validate_interference_patterns(self, patterns)
**Описание:** Validate interference patterns.

Physical Meaning:
    Validates interference patterns for physical reasonableness
    and consistency.

Args:
    patterns (List[Dict[str, Any]]): List of interference...

### _validate_mode_coupling(self, coupling)
**Описание:** Validate mode coupling analysis.

Physical Meaning:
    Validates mode coupling analysis for physical reasonableness
    and consistency.

Args:
    coupling (Dict[str, Any]): Mode coupling analysis r...

### _check_frequency_physical_reasonableness(self, frequencies) -> bool
**Описание:** Check if frequencies are physically reasonable.

Physical Meaning:
    Checks if beating frequencies are within physically
    reasonable ranges for the system.

Args:
    frequencies (List[float]): L...

## ./bhlff/models/level_c/beating/validation/beating_validation_statistics.py
Methods: 2

### __init__(self, bvp_core)
**Описание:** Initialize statistical validation analyzer.

### compute_overall_statistical_validation(self, validation_results)
**Описание:** Compute overall statistical validation.

Physical Meaning:
    Computes overall statistical validation metrics
    from individual validation results.

Args:
    validation_results (Dict[str, Any]): I...

## ./bhlff/models/level_c/beating/validation_basic/beating_validation_frequencies.py
Methods: 6

### __init__(self, bvp_core)
**Описание:** Initialize beating frequency validation.

### validate_beating_frequencies_physical(self, frequencies)
**Описание:** Physical validation of beating frequencies.

Physical Meaning:
    Validates beating frequencies according to physical principles
    and theoretical constraints of the 7D phase field theory.

Mathema...

### _is_physically_valid_frequency(self, frequency) -> bool
**Описание:** Check if frequency is physically valid.

Physical Meaning:
    Validates that the frequency is within physically meaningful bounds
    according to the 7D phase field theory.

Args:
    frequency (flo...

### _is_within_theoretical_bounds(self, frequency) -> bool
**Описание:** Check if frequency is within theoretical bounds.

Physical Meaning:
    Validates that the frequency is within theoretical constraints
    for spatial, temporal, and phase frequencies in the 7D theory...

### _analyze_frequency_harmonics(self, frequencies)
**Описание:** Analyze frequency harmonics and relationships.

Physical Meaning:
    Analyzes harmonic relationships between frequencies
    according to the 7D phase field theory.

Args:
    frequencies (List[float...

### validate_beating_frequencies(self, frequencies)
**Описание:** Legacy method for backward compatibility.

Physical Meaning:
    Basic frequency validation for backward compatibility.
    For comprehensive validation, use validate_beating_frequencies_physical.

Ar...

## ./bhlff/models/level_c/beating/validation_basic/beating_validation_patterns.py
Methods: 9

### __init__(self, bvp_core)
**Описание:** Initialize beating pattern validation.

### validate_interference_patterns_physical(self, patterns)
**Описание:** Physical validation of interference patterns.

Physical Meaning:
    Validates interference patterns according to physical principles
    and theoretical constraints of the 7D phase field theory.

Mat...

### validate_interference_patterns(self, patterns)
**Описание:** Legacy method for backward compatibility.

Physical Meaning:
    Basic pattern validation for backward compatibility.
    For comprehensive validation, use validate_interference_patterns_physical.

Ar...

### _validate_single_pattern(self, pattern)
**Описание:** Validate a single interference pattern.

### _is_physically_valid_pattern(self, pattern) -> bool
**Описание:** Check if pattern is physically valid.

Physical Meaning:
    Validates that the pattern parameters are within physically
    meaningful bounds according to the 7D phase field theory.

Args:
    patter...

### _is_within_theoretical_bounds(self, pattern) -> bool
**Описание:** Check if pattern is within theoretical bounds.

Physical Meaning:
    Validates that the pattern is within theoretical constraints
    for interference patterns in the 7D theory.

Args:
    pattern (D...

### _analyze_pattern_coherence(self, patterns)
**Описание:** Analyze pattern coherence and relationships.

Physical Meaning:
    Analyzes coherence relationships between patterns
    according to the 7D phase field theory.

Args:
    patterns (List[Dict[str, An...

### _calculate_pattern_coherence(self, pattern1, pattern2) -> float
**Описание:** Calculate coherence between two patterns.

Physical Meaning:
    Calculates the coherence between two interference patterns
    based on their amplitude, phase, and frequency relationships.

Args:
   ...

### _classify_coherence_type(self, coherence) -> str
**Описание:** Classify coherence type based on coherence value.

Physical Meaning:
    Classifies the type of coherence relationship between patterns
    according to the 7D phase field theory.

Args:
    coherence...

## ./bhlff/models/level_c/boundaries.py
Methods: 16

### __init__(self, bvp_core)
**Описание:** Initialize boundary analyzer.

Args:
    bvp_core (BVPCore): BVP core framework instance.

### analyze_boundaries(self, envelope)
**Описание:** Perform comprehensive boundary analysis.

Physical Meaning:
    Analyzes all aspects of boundaries in the 7D phase field,
    including detection, classification, effects, and stability.

Mathematical...

### _analyze_level_set_boundaries(self, envelope)
**Описание:** Analyze boundaries using level set methods.

### _analyze_phase_field_boundaries(self, envelope)
**Описание:** Analyze boundaries using phase field methods.

### _analyze_topological_boundaries(self, envelope)
**Описание:** Analyze boundaries using topological methods.

### _analyze_boundary_energy(self, envelope)
**Описание:** Analyze boundary energy landscape.

### _find_level_set_boundary(self, level_set)
**Описание:** Find boundary of level set.

### _analyze_boundary_properties(self, boundary_mask, field)
**Описание:** Analyze properties of boundary.

### _find_critical_points(self, field)
**Описание:** Find critical points in the field.

### _analyze_topological_structure(self, critical_points, field)
**Описание:** Analyze topological structure of the field.

### _classify_topological_boundaries(self, critical_points, field)
**Описание:** Classify boundaries by topological properties.

### _analyze_energy_landscape(self, energy_density)
**Описание:** Analyze energy landscape.

### _find_energy_boundaries(self, energy_density)
**Описание:** Find boundaries in energy landscape.

### _analyze_boundary_stability(self, energy_boundaries, energy_density)
**Описание:** Analyze stability of boundaries.

### _estimate_boundary_curvature(self, boundary_mask) -> float
**Описание:** Estimate curvature of boundary.

### _create_boundary_summary(self, level_set_analysis, phase_field_analysis, topological_analysis, energy_analysis)
**Описание:** Create summary of boundary analysis.

## ./bhlff/models/level_c/boundary_analysis.py
Methods: 16

### __init__(self, bvp_core)
**Описание:** Initialize boundary analysis.

Args:
    bvp_core (BVPCore): BVP core framework instance.

### analyze_single_wall(self, domain, boundary_params)
**Описание:** Analyze single wall boundary effects (C1 test).

Physical Meaning:
    Performs comprehensive analysis of a single spherical
    boundary with admittance contrast, including resonance
    mode detecti...

### _create_boundary_geometry(self, domain, boundary_params, contrast) -> BoundaryGeometry
**Описание:** Create boundary geometry.

Physical Meaning:
    Creates a spherical boundary geometry with specified
    contrast and material properties.

### _analyze_admittance_spectrum(self, domain, boundary, frequency_range) -> AdmittanceSpectrum
**Описание:** Analyze admittance spectrum.

Physical Meaning:
    Computes the complex admittance Y(ω) spectrum over the
    frequency range, revealing resonance frequencies and
    system response characteristics....

### _analyze_radial_profiles(self, domain, boundary) -> RadialProfile
**Описание:** Analyze radial profiles.

Physical Meaning:
    Computes the radial distribution of field amplitude A(r),
    revealing the spatial structure of resonance modes and
    field concentration regions.

M...

### _find_resonance_modes(self, admittance_spectrum)
**Описание:** Find resonance modes in admittance spectrum.

Physical Meaning:
    Identifies resonance modes by finding peaks in the
    admittance spectrum above the threshold.

### _find_resonance_birth_threshold(self, contrast_results) -> float
**Описание:** Find resonance birth threshold.

Physical Meaning:
    Determines the minimum contrast value η* at which
    the first resonance mode appears.

### _create_boundary_summary(self, contrast_results, resonance_threshold)
**Описание:** Create boundary analysis summary.

Physical Meaning:
    Creates a comprehensive summary of the boundary analysis
    results, including resonance characteristics and threshold
    information.

### _validate_c1_results(self, contrast_results, resonance_threshold) -> bool
**Описание:** Validate C1 test results.

Physical Meaning:
    Validates that the C1 test results meet the acceptance
    criteria for boundary analysis.

### _solve_stationary_frequency(self, domain, boundary, frequency)
**Описание:** Solve stationary problem for given frequency.

Physical Meaning:
    Solves the stationary BVP envelope equation for the
    given frequency, including boundary effects.

### _create_source_field(self, domain, frequency)
**Описание:** Create source field for given frequency.

Physical Meaning:
    Creates a source field s(x) for the given frequency,
    representing external excitation of the system.

### _apply_boundary_conditions(self, field, boundary, frequency)
**Описание:** Apply boundary conditions to field.

Physical Meaning:
    Applies the boundary conditions corresponding to the
    boundary geometry and material contrast.

### _create_spherical_shell_mask(self, domain, center, radius, dr)
**Описание:** Create spherical shell mask.

Physical Meaning:
    Creates a mask for a spherical shell at the given radius,
    used for radial profile analysis.

### _find_peaks(self, signal, height)
**Описание:** Find peaks in signal above threshold.

Physical Meaning:
    Identifies peaks in the signal that exceed the specified
    height threshold.

### _find_local_maxima(self, radii, amplitudes)
**Описание:** Find local maxima in radial profile.

Physical Meaning:
    Identifies local maxima in the radial amplitude profile,
    indicating regions of field concentration.

### _compute_quality_factor(self, admittance_spectrum, peak_idx) -> float
**Описание:** Compute quality factor for resonance peak.

Physical Meaning:
    Computes the quality factor Q = ω / (2 * Δω) for the
    resonance peak, characterizing the resonance sharpness.

## ./bhlff/models/level_c/level_c_integration.py
Methods: 14

### __init__(self, bvp_core)
**Описание:** Initialize Level C integration.

Args:
    bvp_core (BVPCore): BVP core framework instance.

### run_all_tests(self, test_config) -> LevelCResults
**Описание:** Run all Level C tests.

Physical Meaning:
    Executes all Level C tests in sequence, providing
    comprehensive analysis of boundary effects, resonator
    chains, memory effects, and mode beating.
...

### _run_c1_test(self, test_config)
**Описание:** Run C1: Single wall boundary analysis.

Physical Meaning:
    Performs single wall boundary analysis to study
    resonance mode birth and admittance contrast effects.

### _run_c2_test(self, test_config)
**Описание:** Run C2: Resonator chain ABCD analysis.

Physical Meaning:
    Performs resonator chain analysis using ABCD model
    to study system resonance modes and coupling effects.

### _run_c3_test(self, test_config)
**Описание:** Run C3: Quench memory and pinning analysis.

Physical Meaning:
    Performs quench memory analysis to study memory
    effects, pinning, and field stabilization.

### _run_c4_test(self, test_config)
**Описание:** Run C4: Mode beating analysis.

Physical Meaning:
    Performs mode beating analysis to study dual-mode
    excitation, beating patterns, and drift velocity.

### _create_resonator_layers(self, c2_params)
**Описание:** Create resonator layers for C2 test.

Physical Meaning:
    Creates resonator layers based on C2 parameters
    for ABCD model analysis.

### _run_abcd_analysis(self, abcd_model, c2_params)
**Описание:** Run ABCD analysis for C2 test.

Physical Meaning:
    Performs ABCD model analysis to study system
    resonance modes and coupling effects.

### _validate_c1_results(self, c1_results) -> bool
**Описание:** Validate C1 test results.

Physical Meaning:
    Validates that C1 test results meet the acceptance
    criteria for boundary analysis.

### _validate_c2_results(self, c2_results) -> bool
**Описание:** Validate C2 test results.

Physical Meaning:
    Validates that C2 test results meet the acceptance
    criteria for resonator chain analysis.

### _validate_c3_results(self, c3_results) -> bool
**Описание:** Validate C3 test results.

Physical Meaning:
    Validates that C3 test results meet the acceptance
    criteria for quench memory analysis.

### _validate_c4_results(self, c4_results) -> bool
**Описание:** Validate C4 test results.

Physical Meaning:
    Validates that C4 test results meet the acceptance
    criteria for mode beating analysis.

### _validate_overall_results(self, c1_results, c2_results, c3_results, c4_results) -> str
**Описание:** Validate overall Level C results.

Physical Meaning:
    Validates that all Level C tests meet their
    acceptance criteria and determines overall status.

### create_test_configuration(self, domain_params, physics_params) -> TestConfiguration
**Описание:** Create test configuration for Level C tests.

Physical Meaning:
    Creates a comprehensive test configuration for
    all Level C tests based on domain and physics parameters.

## ./bhlff/models/level_c/memory/memory_analyzer.py
Methods: 17

### __init__(self, bvp_core)
**Описание:** Initialize memory analyzer.

Physical Meaning:
    Sets up the analyzer with the BVP core for accessing
    field data and computational resources.

Args:
    bvp_core (BVPCore): BVP core instance for...

### analyze_memory(self, envelope)
**Описание:** Analyze memory systems in the envelope field.

Physical Meaning:
    Analyzes memory systems in the 7D envelope field,
    identifying information storage, persistence patterns,
    and memory-field i...

### _analyze_temporal_correlations(self, envelope)
**Описание:** Analyze temporal correlations in the envelope field.

Physical Meaning:
    Analyzes temporal correlations that indicate memory
    effects, including autocorrelation and cross-correlation
    pattern...

### _detect_persistence_patterns(self, envelope)
**Описание:** Detect persistence patterns in the envelope field.

Physical Meaning:
    Detects persistence patterns that indicate memory
    effects, including temporal persistence, spatial
    persistence, and ph...

### _calculate_memory_capacity(self, envelope, temporal_analysis) -> float
**Описание:** Calculate memory capacity from temporal analysis.

Physical Meaning:
    Calculates the memory capacity based on temporal
    correlations and persistence patterns, providing
    a quantitative measur...

### _analyze_memory_interactions(self, envelope, persistence_patterns)
**Описание:** Analyze memory-field interactions.

Physical Meaning:
    Analyzes interactions between memory systems and
    the field, including information transfer, memory
    encoding, and retrieval mechanisms....

### _calculate_memory_strength(self, envelope, persistence_patterns) -> float
**Описание:** Calculate the strength of memory effects.

Physical Meaning:
    Calculates the overall strength of memory effects
    in the envelope field, providing a quantitative
    measure of memory system acti...

### _calculate_autocorrelation(self, envelope)
**Описание:** Calculate autocorrelation of the envelope field.

### _calculate_cross_correlation(self, envelope)
**Описание:** Calculate cross-correlation between different field components.

### _calculate_correlation_statistics(self, autocorrelation, cross_correlation)
**Описание:** Calculate correlation statistics.

### _analyze_temporal_persistence(self, envelope)
**Описание:** Analyze temporal persistence patterns.

### _analyze_spatial_persistence(self, envelope)
**Описание:** Analyze spatial persistence patterns.

### _analyze_phase_persistence(self, envelope)
**Описание:** Analyze phase persistence patterns.

### _find_correlation_length(self, autocorrelation) -> float
**Описание:** Find correlation length from autocorrelation.

### _calculate_interaction_strength(self, envelope, persistence_patterns) -> float
**Описание:** Calculate memory-field interaction strength.

### _analyze_information_transfer(self, envelope)
**Описание:** Analyze information transfer in the field.

### _calculate_encoding_efficiency(self, envelope, persistence_patterns) -> float
**Описание:** Calculate memory encoding efficiency.

## ./bhlff/models/level_c/memory/memory_utilities.py
Methods: 14

### calculate_memory_metrics(envelope)
**Описание:** Calculate memory metrics for the envelope field.

Physical Meaning:
    Calculates various memory metrics including capacity,
    efficiency, and strength based on field properties.

Mathematical Foun...

### analyze_memory_patterns(envelope, threshold)
**Описание:** Analyze memory patterns in the envelope field.

Physical Meaning:
    Analyzes memory patterns including temporal, spatial,
    and phase patterns that indicate memory effects.

Args:
    envelope (np...

### calculate_memory_interactions(envelope, patterns)
**Описание:** Calculate memory interactions between different patterns.

Physical Meaning:
    Calculates interactions between different memory
    patterns, including coupling strength and interaction
    mechanis...

### validate_memory_analysis(results) -> bool
**Описание:** Validate memory analysis results.

Physical Meaning:
    Validates memory analysis results to ensure they
    are physically meaningful and mathematically consistent.

Args:
    results (Dict[str, Any...

### _calculate_capacity(envelope) -> float
**Описание:** Calculate memory capacity from field properties.

### _calculate_efficiency(envelope) -> float
**Описание:** Calculate memory efficiency from field properties.

### _calculate_strength(envelope) -> float
**Описание:** Calculate memory strength from field properties.

### _calculate_persistence(envelope) -> float
**Описание:** Calculate memory persistence from field properties.

### _analyze_temporal_patterns(envelope, threshold)
**Описание:** Analyze temporal memory patterns.

### _analyze_spatial_patterns(envelope, threshold)
**Описание:** Analyze spatial memory patterns.

### _analyze_phase_patterns(envelope, threshold)
**Описание:** Analyze phase memory patterns.

### _calculate_pattern_interactions(patterns)
**Описание:** Calculate interactions between different patterns.

### _calculate_field_interactions(envelope, patterns)
**Описание:** Calculate interactions between field and patterns.

### _calculate_interaction_strength(patterns) -> float
**Описание:** Calculate overall interaction strength.

## ./bhlff/models/level_c/mode_beating_analysis.py
Methods: 21

### __init__(self, bvp_core)
**Описание:** Initialize mode beating analysis.

Args:
    bvp_core (BVPCore): BVP core framework instance.

### analyze_mode_beating(self, domain, beating_params)
**Описание:** Analyze mode beating effects (C4 test).

Physical Meaning:
    Performs comprehensive analysis of mode beating effects,
    including dual-mode excitation, beating patterns,
    and drift velocity ana...

### _analyze_background_beating(self, domain, dual_mode, time_params)
**Описание:** Analyze background beating without pinning.

Physical Meaning:
    Analyzes mode beating in the absence of pinning
    effects, providing baseline measurements for
    comparison with pinned systems.

### _analyze_pinned_beating(self, domain, dual_mode, time_params)
**Описание:** Analyze pinned beating with memory effects.

Physical Meaning:
    Analyzes mode beating in the presence of pinning
    effects, including memory-induced field stabilization.

### _compute_theoretical_analysis(self, dual_mode)
**Описание:** Compute theoretical analysis for dual-mode system.

Physical Meaning:
    Computes theoretical predictions for the dual-mode
    system, including wave vectors and drift velocity.

Mathematical Founda...

### _analyze_errors(self, background_results, pinned_results, theoretical_analysis)
**Описание:** Analyze errors between numerical and theoretical results.

Physical Meaning:
    Computes error metrics between numerical results
    and theoretical predictions for both background
    and pinned sys...

### _create_beating_summary(self, beating_results)
**Описание:** Create beating analysis summary.

Physical Meaning:
    Creates a comprehensive summary of the beating
    analysis results, including error metrics and
    suppression characteristics.

### _validate_c4_results(self, beating_results) -> bool
**Описание:** Validate C4 test results.

Physical Meaning:
    Validates that the C4 test results meet the acceptance
    criteria for mode beating analysis.

### _create_dual_mode_field(self, domain, dual_mode)
**Описание:** Create dual-mode field.

Physical Meaning:
    Creates a field configuration with dual-mode
    excitation for beating analysis.

### _create_dual_mode_field_with_pinning(self, domain, dual_mode)
**Описание:** Create dual-mode field with pinning effects.

Physical Meaning:
    Creates a field configuration with dual-mode
    excitation and pinning effects for beating analysis.

### _evolve_dual_mode_field(self, field, dual_mode, time_params)
**Описание:** Evolve dual-mode field in time.

Physical Meaning:
    Performs time evolution of the dual-mode field,
    including beating pattern development.

### _evolve_dual_mode_field_with_pinning(self, field, dual_mode, time_params)
**Описание:** Evolve dual-mode field with pinning effects.

Physical Meaning:
    Performs time evolution of the dual-mode field
    with pinning effects, including memory-induced
    field stabilization.

### _analyze_beating_patterns(self, time_evolution, dual_mode) -> BeatingPattern
**Описание:** Analyze beating patterns in field evolution.

Physical Meaning:
    Analyzes the beating patterns in the field evolution,
    including beating frequency and amplitude modulation.

### _analyze_drift_velocity(self, time_evolution) -> DriftVelocityAnalysis
**Описание:** Analyze drift velocity from field evolution.

Physical Meaning:
    Computes the drift velocity of field patterns by analyzing
    cross-correlation of effective intensity I_eff(x,t) over time.

### _create_dual_mode_source(self, dual_mode, time)
**Описание:** Create dual-mode source at given time.

Physical Meaning:
    Creates the dual-mode source s(x,t) = s₁(x) e^(-iω₁t) + s₂(x) e^(-iω₂t)
    at the specified time.

### _apply_evolution_operator(self, field, source, dt)
**Описание:** Apply evolution operator to field.

Physical Meaning:
    Applies the evolution operator to advance the field
    in time, including source terms.

### _compute_wave_vector(self, frequency) -> WaveVector
**Описание:** Compute wave vector for given frequency.

Physical Meaning:
    Computes the wave vector k = ω / c_φ for the
    given frequency.

### _apply_moving_average(self, data, window_size)
**Описание:** Apply moving average to data.

Physical Meaning:
    Applies moving average smoothing to reduce noise
    in the field evolution data.

### _compute_cross_correlation_2d(self, field1, field2)
**Описание:** Compute 2D cross-correlation between fields.

Physical Meaning:
    Computes cross-correlation between two field
    configurations to measure pattern similarity.

### _find_peak_shift(self, correlation) -> float
**Описание:** Find peak shift in correlation.

Physical Meaning:
    Finds the shift of the correlation peak,
    indicating pattern displacement.

### _compute_temporal_coherence(self, amplitude_evolution) -> float
**Описание:** Compute temporal coherence of amplitude evolution.

Physical Meaning:
    Computes the temporal coherence of the amplitude
    evolution, indicating pattern stability.

## ./bhlff/models/level_c/quench_memory_analysis.py
Methods: 22

### __init__(self, bvp_core)
**Описание:** Initialize quench memory analysis.

Args:
    bvp_core (BVPCore): BVP core framework instance.

### analyze_quench_memory(self, domain, memory_params)
**Описание:** Analyze quench memory and pinning effects (C3 test).

Physical Meaning:
    Performs comprehensive analysis of quench memory effects,
    including memory formation, pinning analysis, and drift
    ve...

### _evolve_with_memory(self, domain, memory, time_params)
**Описание:** Evolve field with memory effects.

Physical Meaning:
    Performs time evolution of the field with memory effects,
    including memory kernel application and quench detection.

Mathematical Foundatio...

### _analyze_drift_velocity(self, time_evolution) -> DriftAnalysis
**Описание:** Analyze drift velocity from field evolution.

Physical Meaning:
    Computes the drift velocity of field patterns by analyzing
    cross-correlation of effective intensity I_eff(x,t) over time.

Mathe...

### _analyze_cross_correlation(self, time_evolution)
**Описание:** Analyze cross-correlation of field evolution.

Physical Meaning:
    Computes cross-correlation analysis to understand
    pattern stability and temporal coherence.

### _analyze_jaccard_index(self, time_evolution)
**Описание:** Analyze Jaccard index for pattern stability.

Physical Meaning:
    Computes the Jaccard index to measure pattern
    stability and similarity over time.

### _find_freezing_threshold(self, memory_results) -> float
**Описание:** Find freezing threshold for memory parameters.

Physical Meaning:
    Determines the minimum memory strength γ* required
    to achieve field pinning (v_cell < 10⁻³ L/T₀).

### _create_memory_summary(self, memory_results, freezing_threshold)
**Описание:** Create memory analysis summary.

Physical Meaning:
    Creates a comprehensive summary of the memory analysis
    results, including freezing characteristics and stability
    metrics.

### _validate_c3_results(self, memory_results, freezing_threshold) -> bool
**Описание:** Validate C3 test results.

Physical Meaning:
    Validates that the C3 test results meet the acceptance
    criteria for quench memory analysis.

### _create_initial_field(self, domain)
**Описание:** Create initial field configuration.

Physical Meaning:
    Creates the initial field configuration for
    time evolution analysis.

### _create_memory_kernel(self, memory) -> MemoryKernel
**Описание:** Create memory kernel.

Physical Meaning:
    Creates the memory kernel K(t) that determines
    how past events influence current field evolution.

### _apply_memory_term(self, field_history, memory_kernel, memory)
**Описание:** Apply memory term to field.

Physical Meaning:
    Applies the memory term Γ_memory[a] = -γ ∫_0^t K(t-τ) a(τ) dτ
    to the current field based on field history.

### _apply_evolution_operator(self, field, memory_term, dt)
**Описание:** Apply evolution operator to field.

Physical Meaning:
    Applies the evolution operator to advance the field
    in time, including memory effects.

### _detect_quench_events(self, field, time)
**Описание:** Detect quench events in field.

Physical Meaning:
    Detects quench events based on amplitude, detuning,
    and gradient thresholds.

### _collect_quench_events(self, field_history)
**Описание:** Collect all quench events from field history.

Physical Meaning:
    Collects all quench events detected during the
    field evolution.

### _apply_moving_average(self, data, window_size)
**Описание:** Apply moving average to data.

Physical Meaning:
    Applies moving average smoothing to reduce noise
    in the field evolution data.

### _compute_cross_correlation_2d(self, field1, field2)
**Описание:** Compute 2D cross-correlation between fields.

Physical Meaning:
    Computes cross-correlation between two field
    configurations to measure pattern similarity.

### _find_peak_shift(self, correlation) -> float
**Описание:** Find peak shift in correlation.

Physical Meaning:
    Finds the shift of the correlation peak,
    indicating pattern displacement.

### _compute_jaccard_index(self, field_evolution) -> float
**Описание:** Compute Jaccard index for pattern stability.

Physical Meaning:
    Computes the Jaccard index to measure pattern
    stability and similarity over time.

### _compute_stability_score(self, field_evolution) -> float
**Описание:** Compute stability score for field evolution.

Physical Meaning:
    Computes a stability score based on the
    consistency of field patterns over time.

### _analyze_correlation_decay(self, correlation_matrix)
**Описание:** Analyze correlation decay over time.

Physical Meaning:
    Analyzes how correlation decays over time,
    indicating pattern stability.

### _analyze_pattern_stability(self, field_evolution)
**Описание:** Analyze pattern stability over time.

Physical Meaning:
    Analyzes the stability of field patterns
    over time evolution.

## ./bhlff/models/level_c/resonators/resonator_analysis.py
Methods: 19

### __init__(self, bvp_core)
**Описание:** Initialize resonator analysis.

Args:
    bvp_core (BVPCore): BVP core instance for field access.

### analyze_resonator_correlations(self, envelope)
**Описание:** Analyze resonator correlations.

Physical Meaning:
    Analyzes correlations between different resonator modes
    and spatial/temporal patterns in the envelope field.

Args:
    envelope (np.ndarray)...

### detect_resonator_patterns(self, envelope)
**Описание:** Detect resonator patterns in the envelope field.

Physical Meaning:
    Detects specific resonator patterns that indicate
    structured resonator behavior in the 7D phase field.

Args:
    envelope (...

### calculate_resonator_statistics(self, envelope)
**Описание:** Calculate resonator statistics.

Physical Meaning:
    Calculates statistical measures of resonator behavior,
    including amplitude distributions, frequency statistics,
    and resonator density.

A...

### _calculate_spatial_correlations(self, envelope)
**Описание:** Calculate spatial correlations.

### _calculate_temporal_correlations(self, envelope)
**Описание:** Calculate temporal correlations.

### _calculate_phase_correlations(self, envelope)
**Описание:** Calculate phase correlations.

### _calculate_cross_correlations(self, envelope)
**Описание:** Calculate cross-correlations between different dimension types.

### _detect_standing_wave_patterns(self, envelope)
**Описание:** Detect standing wave patterns.

### _detect_traveling_wave_patterns(self, envelope)
**Описание:** Detect traveling wave patterns.

### _detect_interference_patterns(self, envelope)
**Описание:** Detect interference patterns.

### _calculate_amplitude_statistics(self, envelope)
**Описание:** Calculate amplitude statistics.

### _calculate_frequency_statistics(self, envelope)
**Описание:** Calculate frequency statistics.

### _calculate_resonator_density(self, envelope) -> float
**Описание:** Calculate resonator density.

### _calculate_resonator_distribution(self, envelope)
**Описание:** Calculate resonator distribution.

### _has_standing_wave_characteristics(self, data) -> bool
**Описание:** Check for standing wave characteristics.

### _has_traveling_wave_characteristics(self, data) -> bool
**Описание:** Check for traveling wave characteristics.

### _has_interference_characteristics(self, data) -> bool
**Описание:** Check for interference characteristics.

### _find_local_maxima(self, data)
**Описание:** Find local maxima in the data.

## ./bhlff/models/level_c/resonators/resonator_analyzer.py
Methods: 15

### __init__(self, bvp_core)
**Описание:** Initialize resonator analyzer.

Physical Meaning:
    Sets up the analyzer with the BVP core for accessing
    field data and computational resources.

Args:
    bvp_core (BVPCore): BVP core instance ...

### analyze_resonators(self, envelope)
**Описание:** Analyze resonator structures in the envelope field.

Physical Meaning:
    Analyzes resonator structures in the 7D envelope field,
    identifying resonance frequencies, quality factors, and
    reson...

### _analyze_frequency_domain(self, envelope)
**Описание:** Analyze frequency domain characteristics.

Physical Meaning:
    Performs FFT analysis to identify frequency components
    in the envelope field, which are essential for detecting
    resonance struc...

### _detect_resonance_peaks(self, frequency_analysis)
**Описание:** Detect resonance peaks in the frequency spectrum.

Physical Meaning:
    Detects resonance peaks that indicate resonator structures,
    including their frequencies, amplitudes, and characteristics.

...

### _calculate_quality_factors(self, frequency_analysis, resonance_peaks)
**Описание:** Calculate quality factors for resonance peaks.

Physical Meaning:
    Calculates quality factors that characterize the sharpness
    and selectivity of resonance peaks, indicating resonator
    qualit...

### _analyze_resonator_interactions(self, envelope, resonance_peaks)
**Описание:** Analyze resonator-field interactions.

Physical Meaning:
    Analyzes interactions between resonator structures and
    the field, including coupling effects, energy transfer,
    and resonance enhanc...

### _calculate_resonance_strength(self, envelope, resonance_peaks) -> float
**Описание:** Calculate the strength of resonance effects.

Physical Meaning:
    Calculates the overall strength of resonance effects
    in the envelope field, providing a quantitative
    measure of resonator ac...

### _find_dominant_frequencies(self, power_spectrum)
**Описание:** Find dominant frequencies in the power spectrum.

### _calculate_frequency_statistics(self, power_spectrum)
**Описание:** Calculate frequency statistics.

### _find_peaks(self, data)
**Описание:** Find peaks in data array.

### _index_to_frequency(self, index, shape) -> float
**Описание:** Convert array index to frequency.

### _calculate_peak_width(self, power_spectrum, peak_index) -> float
**Описание:** Calculate peak width at half maximum.

### _calculate_interaction_strength(self, envelope, resonance_peaks) -> float
**Описание:** Calculate resonator-field interaction strength.

### _analyze_coupling_effects(self, envelope, resonance_peaks)
**Описание:** Analyze coupling effects between resonators and field.

### _calculate_energy_transfer(self, envelope, resonance_peaks) -> float
**Описание:** Calculate energy transfer between resonators and field.

## ./bhlff/models/level_c/resonators/resonator_spectrum.py
Methods: 13

### __init__(self, bvp_core)
**Описание:** Initialize resonator spectrum analyzer.

Args:
    bvp_core (BVPCore): BVP core instance for field access.

### calculate_resonance_spectrum(self, envelope)
**Описание:** Calculate resonance spectrum from envelope field.

Physical Meaning:
    Calculates the resonance spectrum by analyzing frequency
    characteristics and resonance patterns in the envelope field.

Arg...

### detect_resonance_modes(self, envelope)
**Описание:** Detect resonance modes in the envelope field.

Physical Meaning:
    Detects resonance modes that indicate resonator structures,
    including their frequencies, amplitudes, and characteristics.

Args...

### analyze_resonance_quality(self, envelope)
**Описание:** Analyze resonance quality factors.

Physical Meaning:
    Analyzes the quality factors of detected resonances,
    providing quantitative measures of resonance sharpness
    and stability.

Args:
    ...

### _calculate_resonance_characteristics(self, power_spectrum)
**Описание:** Calculate resonance characteristics from power spectrum.

### _find_resonance_peaks(self, resonance_spectrum)
**Описание:** Find resonance peaks in the spectrum.

### _calculate_resonance_statistics(self, resonance_spectrum)
**Описание:** Calculate resonance statistics.

### _analyze_spatial_resonance_modes(self, envelope)
**Описание:** Analyze spatial resonance modes.

### _analyze_temporal_resonance_modes(self, envelope)
**Описание:** Analyze temporal resonance modes.

### _analyze_phase_resonance_modes(self, envelope)
**Описание:** Analyze phase resonance modes.

### _calculate_quality_factor(self, peak, spectrum) -> float
**Описание:** Calculate quality factor for a resonance peak.

### _calculate_resonance_stability(self, envelope, peaks)
**Описание:** Calculate resonance stability metrics.

### _find_peaks_in_1d(self, data)
**Описание:** Find peaks in 1D data array.

## ./bhlff/models/level_d/bvp_integration.py
Methods: 10

### __init__(self, bvp_core)
**Описание:** Initialize Level D BVP integration.

Physical Meaning:
    Sets up integration between Level D models and BVP framework,
    providing access to BVP core functionality and specialized
    Level D anal...

### process_bvp_data(self, envelope)
**Описание:** Process BVP data for Level D operations.

Physical Meaning:
    Analyzes multimode superposition, field projections,
    and streamlines in BVP envelope to understand the
    complex multimode dynamic...

### _analyze_mode_superposition(self, envelope, threshold)
**Описание:** Analyze mode superposition patterns using BVP envelope.

Physical Meaning:
    Performs FFT analysis to decompose the BVP envelope into
    its constituent modes and identifies dominant frequency
    ...

### _analyze_field_projections(self, envelope, axes)
**Описание:** Analyze field projections onto different subspaces.

Physical Meaning:
    Projects the BVP envelope onto different subspaces to
    understand the field structure in spatial and phase
    dimensions ...

### _analyze_streamlines(self, envelope, resolution)
**Описание:** Analyze streamline patterns in the field.

Physical Meaning:
    Computes field gradients to analyze streamline patterns
    and flow characteristics in the BVP envelope, providing
    insights into t...

### _analyze_bvp_integration(self, envelope)
**Описание:** Analyze BVP-specific integration aspects.

Physical Meaning:
    Analyzes how the BVP envelope integrates with Level D
    models, including envelope modulation patterns, carrier
    frequency effects...

### _analyze_envelope_modulation(self, envelope)
**Описание:** Analyze envelope modulation patterns.

### _analyze_carrier_frequency_effects(self, envelope)
**Описание:** Analyze carrier frequency effects on Level D models.

### _analyze_nonlinear_interactions(self, envelope)
**Описание:** Analyze nonlinear interactions.

### _check_bvp_compliance(self, envelope)
**Описание:** Check BVP framework compliance.

## ./bhlff/models/level_d/level_d_models.py
Methods: 9

### __init__(self, domain, parameters)
**Описание:** Initialize Level D models.

Physical Meaning:
    Sets up the multimode superposition and field projection
    models for analyzing the unified phase field structure
    and its interaction windows.

...

### create_multi_mode_field(self, base_field, modes)
**Описание:** Create multi-mode field from base field and additional modes.

Physical Meaning:
    Constructs a multi-mode phase field by superposing
    different frequency components, representing the
    complex...

### analyze_mode_superposition(self, field, new_modes)
**Описание:** Analyze mode superposition on the frame.

Physical Meaning:
    Tests the stability of the phase field frame when
    adding new modes, ensuring topological robustness
    of the underlying field stru...

### project_field_windows(self, field, window_params)
**Описание:** Project fields onto different frequency-amplitude windows.

Physical Meaning:
    Separates the unified phase field into different
    interaction regimes based on frequency and amplitude
    characte...

### trace_phase_streamlines(self, field, center)
**Описание:** Trace phase streamlines around defects.

Physical Meaning:
    Computes streamlines of the phase gradient field,
    revealing the topological structure of phase flow
    around defects and singularit...

### analyze_multimode_field(self, field)
**Описание:** Comprehensive analysis of multimode field.

Physical Meaning:
    Performs complete analysis of a multimode field including
    superposition analysis, field projections, and streamline
    analysis t...

### _get_default_window_params(self)
**Описание:** Get default window parameters for field projections.

Returns:
    Dict: Default window parameters

### validate_field(self, field) -> bool
**Описание:** Validate field for Level D analysis.

Physical Meaning:
    Checks if the field is suitable for Level D analysis,
    including proper shape, finite values, and appropriate
    frequency content.

Arg...

### analyze_field(self, field)
**Описание:** Analyze field for Level D.

Physical Meaning:
    Performs comprehensive Level D analysis of the phase field,
    including multimode superposition, field projections, and
    streamline analysis.

Ma...

## ./bhlff/models/level_d/projections.py
Methods: 22

### __init__(self)
**Описание:** Initialize signature analyzer.

### project_em_field(self, field)
**Описание:** Project onto electromagnetic window.

Physical Meaning:
    Extracts the electromagnetic component of the phase
    field, corresponding to U(1) gauge interactions
    and phase gradient flows.

Mathe...

### project_strong_field(self, field)
**Описание:** Project onto strong interaction window.

Physical Meaning:
    Extracts the strong interaction component, corresponding
    to high-Q localized modes and steep amplitude gradients
    near the core.

...

### project_weak_field(self, field)
**Описание:** Project onto weak interaction window.

Physical Meaning:
    Extracts the weak interaction component, corresponding
    to chiral combinations and parity-breaking envelope
    functions with low Q and...

### project_field_windows(self, field, window_params)
**Описание:** Project fields onto different frequency-amplitude windows.

Physical Meaning:
    Separates the unified phase field into different
    interaction regimes based on frequency and amplitude
    characte...

### analyze_field_signatures(self, projections)
**Описание:** Analyze characteristic signatures of each field type.

Physical Meaning:
    Computes characteristic signatures for each interaction
    type, including localization, range, and anisotropy
    propert...

### project(self, field)
**Описание:** Project field onto weak window.

### _create_em_filter(self, shape)
**Описание:** Create EM window filter.

### _create_frequency_grid(self, shape)
**Описание:** Create frequency grid for filtering.

### _create_strong_filter(self, shape)
**Описание:** Create strong window filter.

### _apply_q_factor_filter(self, frequencies, q_factor)
**Описание:** Apply Q-factor filtering.

### _create_weak_filter(self, shape)
**Описание:** Create weak window filter.

### _apply_chiral_filter(self, chiral_factor)
**Описание:** Apply chiral filtering.

### _analyze_single_field_signature(self, field, field_type)
**Описание:** Analyze signature of a single field.

### _compute_localization(self, field) -> float
**Описание:** Compute field localization metric.

### _compute_range_characteristics(self, field)
**Описание:** Compute range characteristics.

### _compute_anisotropy(self, field) -> float
**Описание:** Compute field anisotropy.

### _compute_chirality(self, field) -> float
**Описание:** Compute field chirality.

### _compute_confinement(self, field) -> float
**Описание:** Compute field confinement.

### _compute_parity_violation(self, field) -> float
**Описание:** Compute parity violation.

### _compute_correlation_length(self, field) -> float
**Описание:** Compute correlation length.

### _compute_decay_rate(self, field) -> float
**Описание:** Compute decay rate.

## ./bhlff/models/level_d/streamlines.py
Methods: 18

### __init__(self, domain)
**Описание:** Initialize topology analyzer.

### trace_phase_streamlines(self, field, center)
**Описание:** Trace phase streamlines around defects.

Physical Meaning:
    Computes streamlines of the phase gradient field,
    revealing the topological structure of phase flow
    around defects and singularit...

### analyze_streamlines(self, field, resolution)
**Описание:** Analyze streamline patterns in the field.

Physical Meaning:
    Computes field gradients to analyze streamline patterns
    and flow characteristics in the field, providing
    insights into the fiel...

### compute_phase_gradient(self, phase)
**Описание:** Compute phase gradient field.

Physical Meaning:
    Computes the gradient of the phase field,
    representing the local direction of phase
    flow and its magnitude.

Mathematical Foundation:
    ∇...

### compute_field_gradients(self, field)
**Описание:** Compute field gradients.

Physical Meaning:
    Computes the gradient of the field in all
    spatial dimensions.

Args:
    field (np.ndarray): Input field

Returns:
    np.ndarray: Field gradients

### compute_divergence(self, gradients)
**Описание:** Compute divergence of gradient field.

Physical Meaning:
    Computes the divergence of the gradient field,
    representing sources and sinks in the flow.

Mathematical Foundation:
    ∇·v = ∂v_x/∂x ...

### compute_curl(self, gradients)
**Описание:** Compute curl of gradient field.

Physical Meaning:
    Computes the curl of the gradient field,
    representing rotational flow patterns.

Mathematical Foundation:
    ∇×v = (∂v_z/∂y - ∂v_y/∂z, ∂v_x/...

### trace_streamlines(self, gradient_field, center)
**Описание:** Trace streamlines in gradient field.

Physical Meaning:
    Traces streamlines that are tangent to the
    gradient field at each point, revealing
    the flow patterns of the field.

Mathematical Fou...

### _create_initial_points(self, center, num_points)
**Описание:** Create initial points for streamline tracing.

### _trace_single_streamline(self, gradient_field, initial_point, integration_steps, step_size)
**Описание:** Trace a single streamline.

### _interpolate_gradient(self, gradient_field, point)
**Описание:** Interpolate gradient at given point.

### _is_out_of_bounds(self, point) -> bool
**Описание:** Check if point is out of bounds.

### analyze_streamline_topology(self, streamlines)
**Описание:** Analyze topology of streamlines.

Physical Meaning:
    Analyzes the topological structure of streamlines,
    including winding numbers, topology classes, and
    stability indices.

Args:
    stream...

### _compute_winding_numbers(self, streamlines)
**Описание:** Compute winding numbers for streamlines.

### _compute_single_winding_number(self, streamline) -> float
**Описание:** Compute winding number for a single streamline.

### _compute_topology_class(self, streamlines) -> str
**Описание:** Compute topology class of streamlines.

### _compute_stability_index(self, streamlines) -> float
**Описание:** Compute stability index of streamlines.

### _compute_streamline_density(self, field, resolution) -> float
**Описание:** Compute streamline density.

## ./bhlff/models/level_d/superposition.py
Methods: 16

### __init__(self, domain)
**Описание:** Initialize stability analyzer.

### create_multi_mode_field(self, base_field, modes)
**Описание:** Create multi-mode field from base field and additional modes.

Physical Meaning:
    Constructs a multi-mode phase field by superposing
    different frequency components, representing the
    complex...

### analyze_frame_stability(self, before, after)
**Описание:** Analyze frame stability using Jaccard index.

Physical Meaning:
    Computes the Jaccard index between frame structures
    before and after mode addition to quantify stability.

Mathematical Foundati...

### compute_jaccard_index(self, map1, map2) -> float
**Описание:** Compute Jaccard index for frame comparison.

Physical Meaning:
    Measures the similarity between two frame maps
    using the Jaccard index, which quantifies the
    overlap of non-zero regions.

Ma...

### _create_single_mode_field(self, frequency, amplitude, phase, spatial_mode)
**Описание:** Create single mode field.

Physical Meaning:
    Creates a single frequency mode with specified
    amplitude, phase, and spatial structure.

Args:
    frequency (float): Mode frequency
    amplitude ...

### _create_coordinate_grids(self)
**Описание:** Create coordinate grids for the domain.

### _create_bvp_envelope_modulation(self, coords, frequency)
**Описание:** Create BVP envelope modulation spatial mode.

### _create_default_spatial_mode(self, coords, frequency)
**Описание:** Create default spatial mode.

### analyze_superposition(self, field, threshold)
**Описание:** Analyze multimode superposition patterns.

Physical Meaning:
    Performs FFT analysis to decompose the field into
    its constituent modes and identifies dominant frequency
    components and their ...

### analyze_mode_superposition(self, field, new_modes)
**Описание:** Analyze mode superposition on the frame.

Physical Meaning:
    Tests the stability of the phase field frame when
    adding new modes, ensuring topological robustness.

Args:
    field (np.ndarray): ...

### _extract_dominant_frequencies(self, fft_field, mask)
**Описание:** Extract dominant frequencies from FFT field.

### _extract_mode_amplitudes(self, fft_field, mask)
**Описание:** Extract mode amplitudes from FFT field.

### _extract_mode_phases(self, fft_field, mask)
**Описание:** Extract mode phases from FFT field.

### _compute_superposition_quality(self, power_spectrum, mask) -> float
**Описание:** Compute superposition quality metric.

### extract_frame(self, field)
**Описание:** Extract frame structure from field.

### compute_stability_metrics(self, frame_before, frame_after)
**Описание:** Compute additional stability metrics.

## ./bhlff/models/level_e/bvp_integration.py
Methods: 12

### __init__(self, bvp_core)
**Описание:** Initialize Level E BVP integration.

Physical Meaning:
    Sets up integration between Level E models and BVP framework,
    providing access to BVP core functionality, quench detection,
    and speci...

### process_bvp_data(self, envelope)
**Описание:** Process BVP data for Level E operations.

Physical Meaning:
    Analyzes soliton structures, topological defects, their dynamics,
    interactions, and formation processes in BVP envelope to understan...

### _analyze_solitons(self, envelope, threshold)
**Описание:** Analyze soliton structures in BVP envelope.

Physical Meaning:
    Identifies and characterizes soliton structures in the BVP envelope,
    including their positions, amplitudes, widths, and stability...

### _analyze_defects(self, envelope, threshold)
**Описание:** Analyze topological defects in BVP envelope.

Physical Meaning:
    Identifies and classifies topological defects in the BVP envelope,
    including their types, positions, charges, and topological pr...

### _analyze_defect_dynamics(self, envelope, time_window)
**Описание:** Analyze defect dynamics and evolution.

Physical Meaning:
    Analyzes the temporal evolution of topological defects,
    including their motion, creation, annihilation, and
    transformation process...

### _analyze_defect_interactions(self, envelope, interaction_radius)
**Описание:** Analyze interactions between defects.

Physical Meaning:
    Analyzes interactions between multiple topological defects,
    including their mutual influence, collision processes,
    and collective b...

### _analyze_defect_formation(self, envelope)
**Описание:** Analyze defect formation processes.

Physical Meaning:
    Analyzes the formation mechanisms of topological defects,
    including nucleation processes, instability development,
    and formation path...

### _analyze_bvp_integration(self, envelope)
**Описание:** Analyze BVP-specific integration aspects.

Physical Meaning:
    Analyzes how the BVP envelope integrates with Level E
    models, including quench detection, envelope modulation
    effects on defect...

### _analyze_quench_detection(self, envelope)
**Описание:** Analyze quench detection in BVP envelope.

### _analyze_envelope_defect_coupling(self, envelope)
**Описание:** Analyze coupling between envelope and defects.

### _analyze_nonlinear_defect_effects(self, envelope)
**Описание:** Analyze nonlinear effects on defect formation.

### _check_bvp_compliance(self, envelope)
**Описание:** Check BVP framework compliance.

## ./bhlff/models/level_e/defect_core.py
Methods: 9

### __init__(self, domain, physics_params)
**Описание:** Initialize defect model.

Physical Meaning:
    Sets up the computational framework for topological defect
    calculations in the 7D phase field theory, including
    fractional Laplacian operators a...

### _setup_defect_operators(self)
**Описание:** Setup operators for defect calculations.

Physical Meaning:
    Initializes the mathematical operators required for
    defect dynamics, including fractional Laplacian and
    interaction potentials.

### _setup_fractional_laplacian(self)
**Описание:** Setup fractional Laplacian operator.

Physical Meaning:
    Computes the spectral representation of the fractional
    Laplacian operator μ(-Δ)^β required for defect dynamics
    in the 7D phase field...

### _setup_interaction_potential(self)
**Описание:** Setup interaction potential between defects.

Physical Meaning:
    Initializes the interaction potential between topological
    defects, which governs their mutual forces and dynamics.
    The poten...

### create_defect(self, position, charge)
**Описание:** Create topological defect at specified position.

Physical Meaning:
    Generates a field configuration with topological defect
    of specified charge at the given position, creating
    localized ph...

### _create_amplitude_profile(self, r, charge)
**Описание:** Create amplitude profile for defect.

Physical Meaning:
    Generates the amplitude profile that determines the
    spatial extent and shape of the topological defect.
    The profile ensures smooth t...

### compute_defect_charge(self, field, center) -> float
**Описание:** Compute topological charge of defect.

Physical Meaning:
    Calculates the topological charge by integrating the
    phase gradient around a closed loop surrounding the
    defect core.

Mathematical...

### _interpolate_phase_along_path(self, phase, x_path, y_path)
**Описание:** Interpolate phase along integration path.

Physical Meaning:
    Computes the phase values along a circular path around
    the defect for topological charge calculation.

### _compute_fractional_green_normalization(self, beta) -> float
**Описание:** Compute normalization constant for fractional Green function.

Physical Meaning:
    Calculates the normalization constant C_β for the fractional
    Green function G_β such that (-Δ)^β G_β = δ in R³....

## ./bhlff/models/level_e/defect_dynamics.py
Methods: 15

### __init__(self, domain, physics_params)
**Описание:** Initialize defect dynamics calculator.

Physical Meaning:
    Sets up the computational framework for defect dynamics
    including force calculations, gyroscopic effects, and
    dissipative processe...

### _setup_dynamics_parameters(self)
**Описание:** Setup parameters for defect dynamics.

Physical Meaning:
    Initializes the physical parameters required for
    defect dynamics calculations using energy-based dynamics.

### simulate_defect_motion(self, initial_position, time_steps, field)
**Описание:** Simulate defect motion using energy-based dynamics.

Physical Meaning:
    Computes defect motion based on energy gradients
    rather than classical mass-based dynamics.

### _find_defect_position(self, field)
**Описание:** Find current position of defect in field.

Physical Meaning:
    Locates the defect center by finding the position
    of minimum amplitude in the field configuration.

Args:
    field: Complex field ...

### _compute_defect_force(self, position, field)
**Описание:** Compute effective force on defect.

Physical Meaning:
    Calculates the effective force acting on the defect
    due to the gradient of the effective potential.

Mathematical Foundation:
    F = -∇U_...

### _compute_effective_potential(self, field)
**Описание:** Compute effective potential from field configuration.

Physical Meaning:
    Calculates the effective potential that governs
    defect motion based on the field configuration.

### _compute_potential_gradient(self, potential, position)
**Описание:** Compute gradient of potential at defect position.

Physical Meaning:
    Calculates the spatial gradient of the effective
    potential at the defect position.

### _interpolate_potential(self, potential, position) -> float
**Описание:** Interpolate potential at arbitrary position.

Physical Meaning:
    Computes the potential value at an arbitrary position
    using interpolation from the grid values.

Args:
    potential: Potential ...

### _compute_gyroscopic_force(self, velocity)
**Описание:** Compute gyroscopic force.

Physical Meaning:
    Calculates the gyroscopic force G × ẋ that arises
    from the topological structure of the defect.

Mathematical Foundation:
    F_gyro = G × ẋ where ...

### _compute_dissipative_force(self, velocity)
**Описание:** Compute dissipative force.

Physical Meaning:
    Calculates the dissipative force D ẋ that represents
    energy loss due to friction and other dissipative processes.

Mathematical Foundation:
    F_...

### _compute_energy_landscape(self, field)
**Описание:** Compute energy landscape from field configuration.

Physical Meaning:
    Calculates the energy landscape that governs
    defect motion based on 7D phase field theory.

### _compute_energy_gradients(self, energy_landscape)
**Описание:** Compute energy gradients for defect motion.

Physical Meaning:
    Calculates energy gradients that drive
    defect motion in 7D phase field theory.

### _integrate_energy_dynamics(self, initial_position, energy_gradients, time_steps)
**Описание:** Integrate energy-based dynamics for defect motion.

Physical Meaning:
    Integrates the energy-based equations of motion
    to obtain defect trajectory.

### _interpolate_energy_gradient(self, energy_gradients, position)
**Описание:** Interpolate energy gradient at arbitrary position.

Physical Meaning:
    Computes the energy gradient at an arbitrary position
    using interpolation from the grid values.

### _interpolate_scalar_field(self, field, x, y, z) -> float
**Описание:** Interpolate scalar field at arbitrary position.

## ./bhlff/models/level_e/defect_implementations.py
Methods: 14

### __init__(self, domain, physics_params, initial_defects)
**Описание:** Initialize multi-defect system.

Physical Meaning:
    Sets up a system of multiple defects with specified
    initial positions and charges.

Args:
    domain: Computational domain
    physics_params...

### _setup_vortex_parameters(self)
**Описание:** Setup parameters specific to vortex defects.

Physical Meaning:
    Initializes vortex-specific parameters including
    circulation, core size, and velocity field properties.

### create_vortex_profile(self, position)
**Описание:** Create vortex field profile.

Physical Meaning:
    Generates a field configuration with vortex structure
    at the specified position, including phase winding
    and amplitude profile.

Mathematica...

### _create_vortex_amplitude(self, r)
**Описание:** Create amplitude profile for vortex.

Physical Meaning:
    Generates the amplitude profile that determines the
    spatial extent of the vortex, with zero amplitude
    at the core and smooth transit...

### _setup_multi_defect_parameters(self)
**Описание:** Setup parameters for multi-defect system.

Physical Meaning:
    Initializes parameters specific to multi-defect
    systems including interaction ranges, annihilation
    criteria, and collective dyn...

### _setup_interaction_potential(self)
**Описание:** Setup interaction potential for multi-defect system.

Physical Meaning:
    Initializes the interaction potential between multiple
    defects, including Green functions and screening effects.

### compute_interaction_forces(self)
**Описание:** Compute interaction forces between all defects.

Physical Meaning:
    Calculates the forces acting on each defect due to
    interactions with all other defects in the system.

Returns:
    Array of ...

### _compute_pair_force(self, defect_i, defect_j)
**Описание:** Compute force between defect pair.

Physical Meaning:
    Calculates the force between two specific defects
    based on their charges and separation.

Args:
    defect_i: First defect configuration
 ...

### simulate_defect_annihilation(self, defect_pair)
**Описание:** Simulate annihilation of defect-antidefect pair.

Physical Meaning:
    Models the annihilation process where a defect and
    antidefect approach and annihilate, releasing energy
    and creating top...

### add_defect(self, position, charge)
**Описание:** Add new defect to the system.

Physical Meaning:
    Adds a new topological defect to the multi-defect
    system with specified position and charge.

Args:
    position: Position of new defect
    ch...

### remove_defect(self, defect_id)
**Описание:** Remove defect from the system.

Physical Meaning:
    Removes a defect from the multi-defect system,
    typically after annihilation or other processes.

Args:
    defect_id: ID of defect to remove

### get_system_energy(self) -> float
**Описание:** Compute total energy of the multi-defect system.

Physical Meaning:
    Calculates the total energy of the system including
    individual defect energies and interaction energies.

Returns:
    Total...

### evolve_system(self, time_step)
**Описание:** Evolve the multi-defect system in time.

Physical Meaning:
    Advances the system in time, updating defect positions
    and handling interactions and annihilation processes.

Args:
    time_step: Ti...

### _check_and_handle_annihilation(self)
**Описание:** Check for and handle defect annihilation.

Physical Meaning:
    Checks if any defect pairs are close enough for
    annihilation and handles the annihilation process.

## ./bhlff/models/level_e/defect_interactions.py
Methods: 11

### __init__(self, domain, physics_params)
**Описание:** Initialize defect interactions calculator.

Physical Meaning:
    Sets up the computational framework for defect interactions
    including Green functions, interaction potentials, and
    annihilatio...

### _setup_interaction_parameters(self)
**Описание:** Setup parameters for defect interactions.

Physical Meaning:
    Initializes the physical parameters required for
    defect interaction calculations including interaction
    strength, range, and Gre...

### compute_interaction_forces(self, positions, charges)
**Описание:** Compute interaction forces between defects.

Physical Meaning:
    Calculates the forces acting on each defect due to
    interactions with all other defects in the system.

Mathematical Foundation:
 ...

### _compute_pair_force(self, r_ij, r_magnitude, charge_i, charge_j)
**Описание:** Compute force between defect pair.

Physical Meaning:
    Calculates the force between two defects based on
    their charges and separation.

Mathematical Foundation:
    F = -qᵢqⱼ ∇G(r) where G is t...

### _compute_green_function(self, r)
**Описание:** Compute fractional Green function and its gradient.

Physical Meaning:
    Calculates the fractional Green function G_β(r) and its gradient
    for defect interactions. The Green function represents
 ...

### simulate_defect_annihilation(self, defect_pair, positions, charges)
**Описание:** Simulate annihilation of defect-antidefect pair.

Physical Meaning:
    Models the process where a defect and antidefect approach
    and annihilate, releasing energy and creating topological
    tran...

### _compute_annihilation_energy(self, charge1, charge2, separation) -> float
**Описание:** Compute energy released during annihilation using fractional Green function.

Physical Meaning:
    Calculates the energy released when a defect-antidefect
    pair annihilates, based on their charges...

### _compute_energy_release_rate(self, annihilation_energy) -> float
**Описание:** Compute rate of energy release during annihilation.

Physical Meaning:
    Calculates how quickly energy is released during
    the annihilation process.

### _compute_relaxation_time(self, separation) -> float
**Описание:** Compute field relaxation time after annihilation.

Physical Meaning:
    Calculates the time required for the field to relax
    to its new configuration after defect annihilation.

### compute_interaction_potential(self, positions, charges) -> float
**Описание:** Compute total interaction potential energy.

Physical Meaning:
    Calculates the total potential energy of the defect
    system due to all pairwise interactions.

Mathematical Foundation:
    U = (1...

### _compute_fractional_green_normalization(self, beta) -> float
**Описание:** Compute normalization constant for fractional Green function.

Physical Meaning:
    Calculates the normalization constant C_β for the fractional
    Green function G_β such that (-Δ)^β G_β = δ in R³....

## ./bhlff/models/level_e/discretization_effects.py
Methods: 23

### __init__(self, reference_config)
**Описание:** Initialize discretization analyzer.

Args:
    reference_config: Reference configuration for comparison

### _setup_convergence_metrics(self)
**Описание:** Setup metrics for convergence analysis.

### analyze_grid_convergence(self, grid_sizes)
**Описание:** Analyze convergence with grid refinement.

Physical Meaning:
    Investigates how results change as the computational
    grid is refined, establishing convergence rates and
    optimal grid sizes.

M...

### _create_grid_config(self, grid_size)
**Описание:** Create configuration with specified grid size.

### _run_simulation(self, config)
**Описание:** Run simulation with given configuration.

Physical Meaning:
    Executes the 7D phase field simulation with specified
    discretization parameters and returns key observables.

### _compute_metrics(self, output)
**Описание:** Compute convergence metrics from simulation output.

### _analyze_convergence(self, results)
**Описание:** Analyze convergence behavior.

Physical Meaning:
    Computes convergence rates and identifies optimal
    grid sizes for different observables.

### _compute_convergence_rate(self, grid_sizes, values) -> float
**Описание:** Compute convergence rate for a metric.

Mathematical Foundation:
    p = log(|e_h1|/|e_h2|)/log(h1/h2) where e_h is the error
    at grid spacing h.

### _assess_convergence_quality(self, values)
**Описание:** Assess quality of convergence.

### _analyze_overall_convergence(self, convergence_rates)
**Описание:** Analyze overall convergence behavior.

### _recommend_grid_size(self, convergence_analysis) -> int
**Описание:** Recommend optimal grid size based on convergence analysis.

### analyze_domain_size_effects(self, domain_sizes)
**Описание:** Analyze effects of finite domain size.

Physical Meaning:
    Investigates how the finite computational domain
    affects results, particularly for long-range
    interactions and boundary effects.

...

### _create_domain_config(self, domain_size)
**Описание:** Create configuration with specified domain size.

### _analyze_domain_effects(self, results)
**Описание:** Analyze effects of domain size on results.

### _analyze_domain_dependence(self, domain_sizes, values)
**Описание:** Analyze dependence of metric on domain size.

### _analyze_overall_domain_effects(self, domain_effects)
**Описание:** Analyze overall domain size effects.

### analyze_time_step_stability(self, time_steps)
**Описание:** Analyze stability with respect to time step.

Physical Meaning:
    Investigates numerical stability of time integration
    schemes and optimal time step selection.

Args:
    time_steps: List of tim...

### _create_time_step_config(self, dt)
**Описание:** Create configuration with specified time step.

### _analyze_time_step_stability(self, results)
**Описание:** Analyze time step stability.

### _analyze_metric_stability(self, time_steps, values)
**Описание:** Analyze stability of a metric with respect to time step.

### _analyze_overall_stability(self, stability_metrics)
**Описание:** Analyze overall time step stability.

### save_results(self, results, filename)
**Описание:** Save discretization analysis results to file.

Args:
    results: Analysis results dictionary
    filename: Output filename

### _make_serializable(self, obj) -> Any
**Описание:** Convert numpy arrays to lists for JSON serialization.

## ./bhlff/models/level_e/failure_detection.py
Methods: 18

### __init__(self, config)
**Описание:** Initialize failure detector.

Args:
    config: Configuration dictionary

### _setup_logging(self)
**Описание:** Setup logging for failure detection.

### _setup_failure_criteria(self)
**Описание:** Setup criteria for failure detection.

### detect_failures(self)
**Описание:** Detect all types of failures in the system.

Physical Meaning:
    Comprehensive analysis of system failures including
    passivity violations, singular modes, energy conservation
    violations, and...

### _check_passivity_violation(self)
**Описание:** Check for passivity violations.

Physical Meaning:
    Verifies that the system remains passive (Re Y_out ≥ 0),
    which is a fundamental physical requirement for energy
    conservation and stabilit...

### _get_impedance_data(self)
**Описание:** Get impedance data for passivity checking.

### _check_singular_mode(self)
**Описание:** Check for singular modes.

Physical Meaning:
    Detects singular modes where λ = 0 with ŝ(0) ≠ 0,
    which can lead to numerical instabilities and
    unphysical behavior.

### _get_mode_data(self)
**Описание:** Get mode data for singular mode checking.

### _check_energy_conservation(self)
**Описание:** Check for energy conservation violations.

Physical Meaning:
    Verifies that energy is conserved within acceptable
    tolerances, which is fundamental for physical consistency.

### _get_energy_data(self)
**Описание:** Get energy data for conservation checking.

### _check_topological_charge(self)
**Описание:** Check for topological charge violations.

Physical Meaning:
    Verifies that topological charge remains integer-valued
    within acceptable tolerances, which is fundamental for
    topological consi...

### _get_topological_charge_data(self)
**Описание:** Get topological charge data for checking.

### _check_numerical_stability(self)
**Описание:** Check for numerical stability issues.

Physical Meaning:
    Detects numerical instabilities such as NaN values,
    infinite values, and excessive growth rates.

### _get_numerical_data(self)
**Описание:** Get numerical data for stability checking.

### _assess_overall_failures(self, failures)
**Описание:** Assess overall failure status.

### analyze_failure_boundaries(self, parameter_ranges)
**Описание:** Analyze boundaries where failures occur.

Physical Meaning:
    Identifies parameter ranges where the system fails,
    establishing boundaries of applicability.

### save_results(self, results, filename)
**Описание:** Save failure detection results to file.

Args:
    results: Detection results dictionary
    filename: Output filename

### _make_serializable(self, obj) -> Any
**Описание:** Convert numpy arrays to lists for JSON serialization.

## ./bhlff/models/level_e/level_e_experiments.py
Methods: 23

### __init__(self, config)
**Описание:** Initialize Level E experiments.

Args:
    config: Configuration dictionary with experiment parameters

### _setup_logging(self)
**Описание:** Setup logging for experiments.

### _setup_analyzers(self)
**Описание:** Setup analysis components.

### _setup_experiment_parameters(self)
**Описание:** Setup experiment parameters.

### run_full_analysis(self)
**Описание:** Execute complete Level E analysis suite.

Physical Meaning:
    Performs comprehensive investigation of system stability
    and sensitivity, providing complete characterization of
    the 7D phase fi...

### _run_sensitivity_analysis(self)
**Описание:** Run sensitivity analysis (E1).

### _run_robustness_testing(self)
**Описание:** Run robustness testing (E2).

### _run_discretization_analysis(self)
**Описание:** Run discretization analysis (E3).

### _run_failure_detection(self)
**Описание:** Run failure detection (E4).

### _run_phase_mapping(self)
**Описание:** Run phase mapping (E5).

### _run_performance_analysis(self)
**Описание:** Run performance analysis (E6).

### _assess_overall_results(self, results)
**Описание:** Assess overall results of Level E experiments.

### _generate_recommendations(self, results)
**Описание:** Generate recommendations based on results.

### run_soliton_experiments(self)
**Описание:** Run specialized soliton experiments.

Physical Meaning:
    Performs detailed analysis of soliton solutions including
    stability analysis, energy computation, and topological
    charge verificatio...

### _test_baryon_solitons(self)
**Описание:** Test baryon soliton solutions.

### _test_skyrmion_solitons(self)
**Описание:** Test skyrmion soliton solutions.

### _test_soliton_interactions(self)
**Описание:** Test soliton-soliton interactions.

### run_defect_experiments(self)
**Описание:** Run specialized defect experiments.

Physical Meaning:
    Performs detailed analysis of topological defects including
    dynamics simulation, interaction analysis, and formation processes.

### _test_single_defects(self)
**Описание:** Test single defect properties.

### _test_defect_pairs(self)
**Описание:** Test defect pair interactions.

### _test_multi_defect_systems(self)
**Описание:** Test multi-defect system dynamics.

### save_results(self, results, filename)
**Описание:** Save Level E experiment results to file.

Args:
    results: Experiment results dictionary
    filename: Output filename

### _make_serializable(self, obj) -> Any
**Описание:** Convert numpy arrays to lists for JSON serialization.

## ./bhlff/models/level_e/performance_analysis.py
Methods: 27

### __init__(self, config)
**Описание:** Initialize performance analyzer.

Args:
    config: Configuration dictionary

### _setup_performance_metrics(self)
**Описание:** Setup performance metrics for analysis.

### _setup_benchmark_cases(self)
**Описание:** Setup benchmark test cases.

### analyze_performance(self)
**Описание:** Perform comprehensive performance analysis.

Physical Meaning:
    Analyzes computational performance across different
    problem sizes and configurations, providing optimization
    recommendations....

### _analyze_scaling_behavior(self)
**Описание:** Analyze computational scaling behavior.

### _test_grid_scaling(self, grid_sizes)
**Описание:** Test scaling with grid size.

### _test_domain_scaling(self, domain_sizes)
**Описание:** Test scaling with domain size.

### _test_time_scaling(self, time_ranges)
**Описание:** Test scaling with time range.

### _analyze_overall_scaling(self, scaling_results)
**Описание:** Analyze overall scaling behavior.

### _compute_overall_assessment(self, assessments) -> str
**Описание:** Compute overall assessment from individual assessments.

### _analyze_accuracy_cost_tradeoffs(self)
**Описание:** Analyze accuracy vs cost trade-offs.

### _compute_actual_accuracy(self, simulation_result) -> float
**Описание:** Compute actual accuracy achieved in simulation.

### _analyze_tradeoffs(self, results)
**Описание:** Analyze accuracy vs cost trade-offs.

### _generate_optimization_recommendations(self, results, optimal_index)
**Описание:** Generate optimization recommendations.

### _run_benchmark_tests(self)
**Описание:** Run benchmark tests for regression testing.

### _benchmark_single_soliton(self)
**Описание:** Benchmark single soliton simulation.

### _benchmark_defect_pair(self)
**Описание:** Benchmark defect pair simulation.

### _benchmark_multi_defect_system(self)
**Описание:** Benchmark multi-defect system simulation.

### _analyze_memory_usage(self)
**Описание:** Analyze memory usage patterns.

### _optimize_parameters(self)
**Описание:** Optimize parameters for best performance.

### _compute_efficiency_score(self, simulation_result, execution_time, memory_usage) -> float
**Описание:** Compute efficiency score for parameter combination.

### _generate_parameter_recommendations(self, optimization_results, optimal_index)
**Описание:** Generate parameter optimization recommendations.

### _run_simulation(self, config)
**Описание:** Run simulation with given configuration.

Physical Meaning:
    Executes the 7D phase field simulation with specified
    parameters and returns key observables.

### save_results(self, results, filename)
**Описание:** Save performance analysis results to file.

Args:
    results: Analysis results dictionary
    filename: Output filename

### _make_serializable(self, obj) -> Any
**Описание:** Convert numpy arrays to lists for JSON serialization.

### power_law(t, a, b)
**Описание:** Нет докстринга

### memory_law(N, a, b)
**Описание:** Нет докстринга

## ./bhlff/models/level_e/phase_mapping.py
Methods: 26

### __init__(self, config)
**Описание:** Initialize phase mapper.

Args:
    config: Configuration dictionary

### _setup_classification_metrics(self)
**Описание:** Setup metrics for regime classification.

### _setup_regime_classifiers(self)
**Описание:** Setup classifiers for different regimes.

### map_phases(self)
**Описание:** Map system phases in parameter space.

Physical Meaning:
    Creates a comprehensive map of system behavior regimes
    in parameter space, identifying transition boundaries
    and regime characteris...

### _generate_parameter_grid(self)
**Описание:** Generate parameter grid for phase mapping.

### _classify_parameter_space(self, parameter_grid)
**Описание:** Classify each point in parameter space.

### _classify_single_point(self, params)
**Описание:** Classify a single parameter point.

### _simulate_parameter_point(self, params)
**Описание:** Simulate single parameter point.

Physical Meaning:
    Runs simulation with given parameters and returns
    key observables for regime classification.

Mathematical Foundation:
    Solves the 7D pha...

### _classify_power_law(self, simulation_result) -> float
**Описание:** Classify power law regime.

### _classify_resonator(self, simulation_result) -> float
**Описание:** Classify resonator regime.

### _classify_frozen(self, simulation_result) -> float
**Описание:** Classify frozen regime.

### _classify_leaky(self, simulation_result) -> float
**Описание:** Classify leaky regime.

### _analyze_regime_boundaries(self, parameter_grid, classifications)
**Описание:** Analyze boundaries between regimes.

### _find_regime_boundary(self, regime_data, regime1, regime2)
**Описание:** Find boundary between two regimes.

### _compute_regime_separation(self, regime1_data, regime2_data) -> float
**Описание:** Compute separation between two regimes.

### _find_boundary_points(self, regime1_data, regime2_data)
**Описание:** Find boundary points between regimes.

### _compute_regime_statistics(self, classifications)
**Описание:** Compute statistics for each regime.

### _create_phase_diagram(self, parameter_grid, classifications)
**Описание:** Create phase diagram visualization data.

### _create_2d_slice(self, classifications, param1, param2, param3_value)
**Описание:** Create 2D slice of phase diagram.

### classify_resonances(self, resonance_data)
**Описание:** Classify resonances as emergent vs fundamental.

Physical Meaning:
    Applies criteria from 7d-00-18.md to distinguish between
    emergent resonances (arising from interactions) and
    fundamental ...

### _compute_universality(self, resonance) -> float
**Описание:** Compute universality score for resonance.

### _compute_shape_quality(self, resonance) -> float
**Описание:** Compute shape quality score for resonance.

### _compute_ecology_score(self, resonance) -> float
**Описание:** Compute ecology score for resonance.

### _summarize_classifications(self, classifications)
**Описание:** Summarize classification results.

### save_results(self, results, filename)
**Описание:** Save phase mapping results to file.

Args:
    results: Mapping results dictionary
    filename: Output filename

### _make_serializable(self, obj) -> Any
**Описание:** Convert numpy arrays to lists for JSON serialization.

## ./bhlff/models/level_e/robustness_tests.py
Methods: 28

### __init__(self, base_config)
**Описание:** Initialize robustness tester.

Args:
    base_config: Base configuration for testing

### _setup_perturbation_generators(self)
**Описание:** Setup generators for different types of perturbations.

### test_noise_robustness(self, noise_levels)
**Описание:** Test robustness to BVP-modulation noise.

Physical Meaning:
    Investigates system response to random perturbations
    in the BVP envelope configuration, simulating environmental
    noise and measu...

### _add_bvp_modulation_noise(self, noise_level)
**Описание:** Add BVP-modulation noise to configurations.

Physical Meaning:
    Adds random perturbations to the BVP envelope configuration
    to simulate environmental noise and measurement uncertainties.

### test_parameter_uncertainty(self, uncertainty_ranges)
**Описание:** Test robustness to parameter uncertainties.

Physical Meaning:
    Investigates how uncertainties in physical parameters
    affect system behavior and stability.

Args:
    uncertainty_ranges: Dictio...

### _generate_parameter_variations(self, param_name, uncertainty)
**Описание:** Generate parameter variations for uncertainty testing.

### _run_simulations_with_param_variations(self, param_name, variations)
**Описание:** Run simulations with parameter variations.

### test_geometry_perturbations(self, perturbation_types)
**Описание:** Test robustness to geometry perturbations.

Physical Meaning:
    Investigates system response to changes in domain
    geometry, boundary conditions, and spatial structure.

Args:
    perturbation_ty...

### _generate_geometry_perturbations(self, perturbation_type)
**Описание:** Generate geometry perturbations.

### _add_boundary_jitter(self, config)
**Описание:** Add random jitter to boundary positions.

### _deform_domain(self, config)
**Описание:** Deform domain geometry.

### _distort_grid(self, config)
**Описание:** Distort computational grid.

### _run_simulations(self, configs)
**Описание:** Run simulations for multiple configurations.

### _simulate_single_case(self, config)
**Описание:** Simulate single configuration case.

Physical Meaning:
    Runs a single simulation with given configuration and returns
    key observables for robustness analysis.

### _compute_degradation(self, outputs, noise_level)
**Описание:** Compute degradation metrics.

Physical Meaning:
    Quantifies how much the system performance degrades
    under noise perturbations compared to the baseline.

### _get_baseline_outputs(self)
**Описание:** Get baseline outputs for comparison.

### _check_passivity(self, outputs)
**Описание:** Check for passivity violations.

Physical Meaning:
    Verifies that the system remains passive (Re Y_out ≥ 0)
    under perturbations, which is a fundamental physical requirement.

### _check_topology(self, outputs)
**Описание:** Check topological stability.

Physical Meaning:
    Verifies that topological invariants (like topological charge)
    remain stable under perturbations.

### _analyze_parameter_sensitivity(self, outputs, variations)
**Описание:** Analyze sensitivity to parameter variations.

### _analyze_geometry_sensitivity(self, outputs, perturbation_type)
**Описание:** Analyze sensitivity to geometry perturbations.

### _generate_gaussian_noise(self, shape, amplitude)
**Описание:** Generate Gaussian noise.

### _generate_uniform_noise(self, shape, amplitude)
**Описание:** Generate uniform noise.

### _generate_colored_noise(self, shape, amplitude, color)
**Описание:** Generate colored noise with power spectrum ~ f^(-color).

### _generate_uniform_param_perturbation(self, param_name, uncertainty) -> float
**Описание:** Generate uniform parameter perturbation.

### _generate_gaussian_param_perturbation(self, param_name, uncertainty) -> float
**Описание:** Generate Gaussian parameter perturbation.

### _generate_systematic_param_perturbation(self, param_name, uncertainty) -> float
**Описание:** Generate systematic parameter perturbation.

### save_results(self, results, filename)
**Описание:** Save robustness test results to file.

Args:
    results: Test results dictionary
    filename: Output filename

### _make_serializable(self, obj) -> Any
**Описание:** Convert numpy arrays to lists for JSON serialization.

## ./bhlff/models/level_e/sensitivity_analysis.py
Methods: 15

### __init__(self, parameter_ranges)
**Описание:** Initialize Sobol analyzer.

Args:
    parameter_ranges: Dictionary mapping parameter names to (min, max) ranges

### generate_lhs_samples(self, n_samples)
**Описание:** Generate Latin Hypercube samples.

Physical Meaning:
    Creates efficient sampling of parameter space ensuring
    good coverage with minimal computational cost.

Args:
    n_samples: Number of sampl...

### compute_sobol_indices(self, samples, outputs)
**Описание:** Compute Sobol sensitivity indices.

Physical Meaning:
    Calculates first-order and total-order Sobol indices
    to rank parameter importance.

Mathematical Foundation:
    S_i = Var[E[Y|X_i]]/Var[Y...

### _compute_first_order_index(self, samples, outputs, param_idx) -> float
**Описание:** Compute first-order Sobol index for parameter.

### _compute_total_order_index(self, samples, outputs, param_idx) -> float
**Описание:** Compute total-order Sobol index for parameter.

### analyze_parameter_sensitivity(self, n_samples)
**Описание:** Perform complete sensitivity analysis.

Physical Meaning:
    Executes full sensitivity analysis workflow including
    sampling, simulation, and index computation.

Args:
    n_samples: Number of sam...

### _run_simulations(self, samples)
**Описание:** Run simulations for parameter samples.

Physical Meaning:
    Executes the 7D phase field simulations for each parameter
    combination to generate output data for sensitivity analysis.

### _simulate_single_case(self, params) -> float
**Описание:** Simulate single parameter case.

Physical Meaning:
    Runs a single simulation with given parameters and returns
    a key observable (e.g., power law exponent, quality factor).

### _rank_parameters(self, sobol_indices)
**Описание:** Rank parameters by their total-order Sobol indices.

Args:
    sobol_indices: Dictionary with Sobol indices

Returns:
    List of (parameter_name, total_order_index) tuples sorted by importance

### _compute_stability_metrics(self, sobol_indices)
**Описание:** Compute stability metrics for sensitivity analysis.

Physical Meaning:
    Evaluates the stability and reliability of the sensitivity
    analysis results, including convergence and consistency checks...

### analyze_mass_complexity_correlation(self, samples, outputs)
**Описание:** Analyze correlation between mass and complexity.

Physical Meaning:
    Investigates the "mass = complexity" thesis by analyzing
    the correlation between particle mass and field complexity
    in t...

### _compute_mass_metrics(self, samples, mass_params)
**Описание:** Compute mass-related metrics from parameters.

### _compute_complexity_metrics(self, samples, complexity_params)
**Описание:** Compute complexity-related metrics from parameters.

### save_results(self, results, filename)
**Описание:** Save sensitivity analysis results to file.

Args:
    results: Analysis results dictionary
    filename: Output filename

### _make_serializable(self, obj) -> Any
**Описание:** Convert numpy arrays to lists for JSON serialization.

## ./bhlff/models/level_e/soliton_core.py
Methods: 10

### __init__(self, domain, physics_params)
**Описание:** Initialize soliton model.

Physical Meaning:
    Sets up the computational framework for finding and analyzing
    stable soliton solutions in the 7D phase field.

Args:
    domain: Computational doma...

### _setup_field_operators(self)
**Описание:** Setup field operators for soliton calculations.

Physical Meaning:
    Initializes the mathematical operators needed for computing
    the energy functional and its derivatives in the 7D phase field.

### _setup_fractional_laplacian(self)
**Описание:** Setup fractional Laplacian operator.

### _setup_skyrme_terms(self)
**Описание:** Setup Skyrme interaction terms.

### _setup_wzw_term(self)
**Описание:** Setup Wess-Zumino-Witten term for baryon number conservation.

Physical Meaning:
    Initializes the WZW term that ensures baryon number conservation
    and provides the correct quantum statistics fo...

### _setup_topological_charge(self)
**Описание:** Setup topological charge calculation for 7D U(1)^3 phase winding.

Physical Meaning:
    Initializes the calculation of topological charge which represents
    the baryon number of the soliton via U(1...

### find_soliton_solution(self, initial_guess)
**Описание:** Find soliton solution using iterative methods.

Physical Meaning:
    Searches for stable localized field configurations that minimize
    the energy functional while preserving topological charge.

M...

### compute_soliton_energy(self, soliton) -> float
**Описание:** Compute total energy of soliton configuration.

Physical Meaning:
    Calculates the total energy of the soliton including kinetic,
    Skyrme, and WZW contributions.

Mathematical Foundation:
    E =...

### analyze_soliton_stability(self, soliton)
**Описание:** Analyze stability of soliton solution.

Physical Meaning:
    Investigates the response of the soliton to small perturbations
    to determine if it represents a stable minimum of the energy
    funct...

### compute_topological_charge(self, soliton) -> float
**Описание:** Compute topological charge of soliton via 7D U(1)^3 phase winding.

Physical Meaning:
    Calculates the baryon number B via U(1)^3 winding over φ-coordinates
    which represents the topological char...

## ./bhlff/models/level_e/soliton_energy.py
Methods: 9

### __init__(self, domain, physics_params)
**Описание:** Initialize energy calculator.

Args:
    domain: Computational domain
    physics_params: Physical parameters

### compute_total_energy(self, field) -> float
**Описание:** Compute total energy of soliton configuration.

Physical Meaning:
    Calculates the total energy of the soliton including kinetic,
    Skyrme, and WZW contributions.

Mathematical Foundation:
    E =...

### compute_kinetic_energy(self, field) -> float
**Описание:** Compute kinetic energy contribution.

Physical Meaning:
    Calculates the kinetic energy contribution from the time
    derivative of the field configuration, representing the
    energy associated w...

### compute_skyrme_energy(self, field) -> float
**Описание:** Compute Skyrme energy contribution.

Physical Meaning:
    Calculates the Skyrme energy contribution from the
    quartic terms in the field derivatives, providing
    stability against collapse.

Mat...

### compute_wzw_energy(self, field) -> float
**Описание:** Compute WZW energy contribution for 7D U(1)^3 phase field.

Physical Meaning:
    Calculates the Wess-Zumino-Witten energy contribution
    for 7D U(1)^3 phase patterns on VBP substrate that ensures
 ...

### compute_energy_gradient(self, field)
**Описание:** Compute gradient of energy functional.

Physical Meaning:
    Calculates the first derivative of the energy functional
    with respect to the field configuration.

### _compute_kinetic_gradient(self, field)
**Описание:** Compute gradient of kinetic energy term.

### _compute_skyrme_gradient(self, field)
**Описание:** Compute gradient of Skyrme terms.

### _compute_wzw_gradient(self, field)
**Описание:** Compute gradient of WZW term.

## ./bhlff/models/level_e/soliton_implementations.py
Methods: 9

### __init__(self, domain, physics_params, charge)
**Описание:** Initialize skyrmion soliton.

Physical Meaning:
    Sets up a soliton with specified topological charge,
    representing multi-baryon states or exotic configurations.

Args:
    domain: Computational...

### _setup_fr_constraints(self)
**Описание:** Setup Finkelstein-Rubinstein constraints for fermionic statistics.

Physical Meaning:
    Implements the Finkelstein-Rubinstein constraints that ensure
    the soliton has fermionic statistics by requ...

### apply_fr_constraints(self, field)
**Описание:** Apply Finkelstein-Rubinstein constraints to field.

Physical Meaning:
    Applies the FR constraints to ensure fermionic statistics
    by enforcing the sign change under 2π rotation.

Args:
    field...

### _apply_rotation(self, field, angle)
**Описание:** Apply rotation to field configuration.

Physical Meaning:
    Rotates the field configuration by the specified angle
    around the rotation axis.

Args:
    field: Field configuration
    angle: Rota...

### _interpolate_field(self, field, position)
**Описание:** Interpolate field value at given position.

Physical Meaning:
    Computes the field value at a non-integer position using
    trilinear interpolation.

Args:
    field: Field configuration
    positi...

### _setup_charge_specific_terms(self)
**Описание:** Setup terms specific to topological charge.

Physical Meaning:
    Initializes charge-specific terms and constraints that
    depend on the topological charge of the soliton.

Mathematical Foundation:...

### _setup_baryon_terms(self)
**Описание:** Setup terms for B=1 soliton.

### _setup_multi_baryon_terms(self)
**Описание:** Setup terms for B>1 soliton.

### _setup_antibaryon_terms(self)
**Описание:** Setup terms for B<0 soliton.

## ./bhlff/models/level_e/soliton_optimization.py
Methods: 10

### __init__(self, domain, physics_params)
**Описание:** Initialize optimizer.

Args:
    domain: Computational domain
    physics_params: Physical parameters

### find_solution(self, initial_guess)
**Описание:** Find soliton solution using iterative methods.

Physical Meaning:
    Searches for stable localized field configurations that minimize
    the energy functional while preserving topological charge.

M...

### _solve_stationary_equation(self, initial_guess)
**Описание:** Solve stationary equation using Newton-Raphson method.

Physical Meaning:
    Finds field configuration that minimizes the energy
    functional, representing a stable soliton solution.

Mathematical ...

### _compute_energy_gradient(self, field)
**Описание:** Compute gradient of energy functional.

Physical Meaning:
    Calculates the first derivative of the energy functional
    with respect to the field configuration.

### _compute_energy_hessian(self, field)
**Описание:** Compute Hessian of energy functional.

Physical Meaning:
    Calculates the second derivative of the energy functional
    for Newton-Raphson iterations.

### _compute_energy_functional(self, field) -> float
**Описание:** Compute energy functional for optimization.

Physical Meaning:
    Computes the total energy of the field configuration
    for optimization algorithms.

### _update_with_line_search(self, U, delta_U, F)
**Описание:** Update solution with line search for optimal step size.

Physical Meaning:
    Finds optimal step size to ensure energy decrease
    and convergence of the Newton-Raphson method.

### _compute_kinetic_gradient(self, field)
**Описание:** Compute gradient of kinetic energy term.

### _compute_skyrme_gradient(self, field)
**Описание:** Compute gradient of Skyrme terms.

### _compute_wzw_gradient(self, field)
**Описание:** Compute gradient of WZW term.

## ./bhlff/models/level_e/soliton_stability.py
Methods: 8

### __init__(self, domain, physics_params)
**Описание:** Initialize stability analyzer.

Args:
    domain: Computational domain
    physics_params: Physical parameters

### analyze_stability(self, soliton)
**Описание:** Analyze stability of soliton solution.

Physical Meaning:
    Investigates the response of the soliton to small perturbations
    to determine if it represents a stable minimum of the energy
    funct...

### _compute_energy_hessian(self, field)
**Описание:** Compute Hessian of energy functional.

Physical Meaning:
    Calculates the second derivative of the energy functional
    for stability analysis.

### _compute_energy_functional(self, field) -> float
**Описание:** Compute energy functional for Hessian calculation.

Physical Meaning:
    Computes the total energy of the field configuration
    for numerical differentiation.

### _analyze_eigenmodes(self, eigenvalues, eigenvectors)
**Описание:** Analyze eigenmodes for understanding perturbation types.

Physical Meaning:
    Classifies eigenmodes by their physical meaning (translational,
    rotational, deformational).

### _analyze_mode_symmetry(self, eigenvector) -> str
**Описание:** Analyze symmetry of eigenmode.

Physical Meaning:
    Determines the type of symmetry of the perturbation
    (translational, rotational, deformational).

### _is_translational_mode(self, eigenvector) -> bool
**Описание:** Check for translational mode.

### _is_rotational_mode(self, eigenvector) -> bool
**Описание:** Check for rotational mode.

## ./bhlff/models/level_f/bvp_integration.py
Methods: 11

### __init__(self, bvp_core)
**Описание:** Initialize Level F BVP integration.

Physical Meaning:
    Sets up integration between Level F models and BVP framework,
    providing access to BVP core functionality, impedance calculation,
    and ...

### process_bvp_data(self, envelope)
**Описание:** Process BVP data for Level F operations.

Physical Meaning:
    Analyzes multi-particle systems, collective modes, phase transitions,
    and nonlinear collective effects in BVP envelope to understand...

### _analyze_multi_particle_systems(self, envelope, threshold)
**Описание:** Analyze multi-particle systems in BVP envelope.

Physical Meaning:
    Identifies and analyzes multiple particle systems in the BVP envelope,
    including particle positions, interactions, and collec...

### _analyze_collective_modes(self, envelope, threshold)
**Описание:** Analyze collective modes in BVP envelope.

Physical Meaning:
    Identifies and analyzes collective oscillation modes in the BVP envelope,
    including their frequencies, amplitudes, and coupling pro...

### _analyze_phase_transitions(self, envelope, threshold)
**Описание:** Analyze phase transitions in BVP envelope.

Physical Meaning:
    Identifies and analyzes phase transitions in the BVP envelope,
    including transition points, critical behavior, and order parameter...

### _analyze_nonlinear_effects(self, envelope, threshold)
**Описание:** Analyze nonlinear collective effects in BVP envelope.

Physical Meaning:
    Analyzes nonlinear collective effects in the BVP envelope,
    including nonlinear coupling, mode mixing, and nonlinear
   ...

### _analyze_bvp_integration(self, envelope)
**Описание:** Analyze BVP-specific integration aspects.

Physical Meaning:
    Analyzes how the BVP envelope integrates with Level F
    models, including impedance calculation, envelope modulation
    effects on c...

### _analyze_impedance_calculation(self, envelope)
**Описание:** Analyze impedance calculation in BVP envelope.

### _analyze_envelope_collective_coupling(self, envelope)
**Описание:** Analyze coupling between envelope and collective modes.

### _analyze_nonlinear_collective_effects(self, envelope)
**Описание:** Analyze nonlinear effects on collective behavior.

### _check_bvp_compliance(self, envelope)
**Описание:** Check BVP framework compliance.

## ./bhlff/models/level_f/collective.py
Methods: 20

### __init__(self, system, excitation_params)
**Описание:** Initialize collective excitations model.

Physical Meaning:
    Sets up the model for studying collective excitations
    in the multi-particle system.

Args:
    system (MultiParticleSystem): Multi-p...

### excite_system(self, external_field)
**Описание:** Excite the system with external field.

Physical Meaning:
    Applies external field to the system and
    computes the response.

Args:
    external_field (np.ndarray): External field F(x,t)

Returns...

### analyze_response(self, response)
**Описание:** Analyze system response to excitation.

Physical Meaning:
    Extracts collective mode frequencies and
    amplitudes from the response.

Args:
    response (np.ndarray): System response R(x,t)

Retur...

### compute_dispersion_relations(self)
**Описание:** Compute dispersion relations for collective modes.

Physical Meaning:
    Calculates ω(k) relations for collective
    excitations in the system.

Returns:
    Dict containing:
        - wave_vectors:...

### compute_susceptibility(self, frequencies)
**Описание:** Compute susceptibility function χ(ω).

Physical Meaning:
    Calculates the linear response susceptibility
    for collective excitations.

Args:
    frequencies (np.ndarray): Frequency array

Returns...

### _setup_analysis_parameters(self)
**Описание:** Setup analysis parameters for collective excitations.

Physical Meaning:
    Initializes parameters needed for dispersion
    relation analysis and response computation.

### _harmonic_excitation(self, external_field)
**Описание:** Apply harmonic excitation to the system.

Physical Meaning:
    Applies harmonic external field and computes
    the steady-state response.

### _impulse_excitation(self, external_field)
**Описание:** Apply impulse excitation to the system.

Physical Meaning:
    Applies impulse external field and computes
    the transient response.

### _frequency_sweep_excitation(self, external_field)
**Описание:** Apply frequency sweep excitation to the system.

Physical Meaning:
    Applies frequency sweep external field and
    computes the response across frequency range.

### _apply_excitation(self, external_field, excitation)
**Описание:** Apply excitation to the system and compute response.

Physical Meaning:
    Applies the external excitation to the system
    and computes the resulting response.

### _compute_external_force(self, external_field, excitation_amplitude)
**Описание:** Compute external force on particles.

Physical Meaning:
    Calculates the external force acting
    on each particle due to the external field.

### _find_spectral_peaks(self, spectrum, frequencies)
**Описание:** Find spectral peaks in the response.

Physical Meaning:
    Identifies resonant frequencies in the
    system response spectrum.

### _analyze_step_resonator_transmission(self, response)
**Описание:** Analyze energy exchange through step resonator boundaries.

Physical Meaning:
    Computes transmission/reflection coefficients for collective modes
    through semi-transparent step resonator boundar...

### _compute_boundary_energy_flux(self, field)
**Описание:** Compute energy flux through step resonator boundaries.

Physical Meaning:
    Calculates energy exchange through semi-transparent
    step resonator boundaries using 7D BVP theory.

### _compute_participation_ratios(self, response)
**Описание:** Compute participation ratios for collective modes.

Physical Meaning:
    Calculates how much each particle participates
    in the collective response.

### _compute_quality_factors(self, peaks, transmission_analysis)
**Описание:** Compute quality factors for collective modes.

Physical Meaning:
    Calculates quality factors based on transmission
    coefficients through step resonator boundaries.

### _solve_dispersion_equation(self, k) -> float
**Описание:** Solve dispersion equation for given wave vector.

Physical Meaning:
    Solves the dispersion equation ω(k) for collective
    modes in the multi-particle system.

### _compute_group_velocity(self, k, omega) -> float
**Описание:** Compute group velocity for given k and ω.

Physical Meaning:
    Calculates the group velocity v_g = dω/dk
    for the dispersion relation.

### _fit_dispersion_relation(self, k_values, frequencies)
**Описание:** Fit dispersion relation to data.

Physical Meaning:
    Fits a mathematical model to the computed
    dispersion relation data.

### analyze(self, data)
**Описание:** Analyze data for this model.

Physical Meaning:
    Performs comprehensive analysis of collective excitations,
    including response analysis and dispersion relations.

Args:
    data (Any): Input da...

## ./bhlff/models/level_f/multi_particle.py
Methods: 29

### __init__(self, domain, particles, interaction_range, interaction_strength)
**Описание:** Initialize multi-particle system.

Physical Meaning:
    Sets up a system of multiple particles with specified
    interactions and computational domain.

Args:
    domain (Domain): Computational doma...

### compute_effective_potential(self)
**Описание:** Compute effective potential for the system.

Physical Meaning:
    Calculates the total effective potential including
    single-particle, pair-wise, and higher-order interactions.

Mathematical Found...

### find_collective_modes(self)
**Описание:** Find collective modes of the system.

Physical Meaning:
    Identifies collective excitations that involve
    coordinated motion of multiple particles.

Returns:
    Dict containing:
        - freque...

### analyze_correlations(self)
**Описание:** Analyze correlation functions.

Physical Meaning:
    Computes spatial and temporal correlations
    between particle positions and phases.

Returns:
    Dict containing:
        - spatial_correlation...

### check_stability(self)
**Описание:** Check stability of the multi-particle system.

Physical Meaning:
    Analyzes the stability of the system by checking
    the eigenvalues of the dynamics matrix.

Returns:
    Dict containing stabilit...

### _setup_interaction_matrices(self)
**Описание:** Setup interaction matrices for efficient computation.

Physical Meaning:
    Pre-computes interaction matrices to optimize
    collective mode calculations.

### _compute_single_particle_potential(self, particle)
**Описание:** Compute single-particle potential contribution.

Physical Meaning:
    Calculates the potential energy contribution
    from a single particle.

### _compute_pair_interaction(self, particle_i, particle_j)
**Описание:** Compute pair-wise interaction potential.

Physical Meaning:
    Calculates the interaction potential between
    two particles.

### _compute_higher_order_interactions(self)
**Описание:** Compute higher-order (three-body, etc.) interactions.

Physical Meaning:
    Calculates three-body and higher-order interaction
    contributions to the effective potential.

### _compute_three_body_interaction(self, particle_i, particle_j, particle_k)
**Описание:** Compute three-body interaction potential.

Physical Meaning:
    Calculates the three-body interaction contribution
    to the effective potential.

### _compute_dynamics_matrix(self)
**Описание:** Compute dynamics matrix using energy-based approach.

Physical Meaning:
    Computes the dynamics matrix for collective modes
    from energy and phase coherence matrices.

### _compute_energy_matrix(self)
**Описание:** Compute energy matrix from field configurations.

### _compute_phase_coherence_matrix(self)
**Описание:** Compute phase coherence matrix between particles.

### _compute_particle_energy(self, particle) -> float
**Описание:** Compute 7D phase field energy for particle.

### _compute_7d_bvp_energy(self, phase_field) -> float
**Описание:** Compute energy using 7D BVP theory.

Physical Meaning:
    Calculates the energy of the phase field
    using the 7D fractional Laplacian operator.

### _get_phase_field_around_particle(self, particle)
**Описание:** Get 7D phase field around a particle.

Physical Meaning:
    Extracts the phase field configuration
    in the vicinity of a particle.

### _compute_phase_coherence(self, particle_i, particle_j) -> float
**Описание:** Compute phase coherence between two particles.

### _compute_self_stiffness(self, particle) -> float
**Описание:** Compute self-stiffness using 7D phase field dynamics.

Physical Meaning:
    Calculates the self-stiffness coefficient
    based on 7D phase field energy rather than classical mechanics.

### _compute_phase_field_energy(self, particle) -> float
**Описание:** Compute 7D phase field energy for particle.

Physical Meaning:
    Calculates the energy of the 7D phase field
    around a particle using 7D BVP theory.

### _compute_coherence_length(self, particle) -> float
**Описание:** Compute coherence length for particle.

### _compute_interaction_energy(self, particle_i, particle_j) -> float
**Описание:** Compute interaction energy between particles using 7D BVP theory.

Physical Meaning:
    Calculates the interaction energy between particles
    based on 7D phase field coherence rather than classical...

### _compute_interaction_stiffness(self, particle_i, particle_j) -> float
**Описание:** Compute interaction stiffness using 7D phase field dynamics.

Physical Meaning:
    Calculates interaction stiffness based on
    7D phase field coherence between particles.

### _compute_7d_phase_coherence(self, particle_i, particle_j) -> float
**Описание:** Compute 7D phase coherence between particles.

Physical Meaning:
    Calculates the phase coherence between two particles
    using 7D phase field theory.

### _compute_participation_ratios(self, eigenvectors)
**Описание:** Compute participation ratios for collective modes.

Physical Meaning:
    Calculates how much each particle participates
    in each collective mode.

### _compute_spatial_correlations(self)
**Описание:** Compute spatial correlation functions.

Physical Meaning:
    Calculates spatial correlations between
    particle positions.

### _compute_phase_correlations(self)
**Описание:** Compute phase correlation matrix.

Physical Meaning:
    Calculates correlations between particle phases.

### _compute_temporal_correlations(self)
**Описание:** Compute temporal correlation functions.

Physical Meaning:
    Calculates temporal correlations in the system
    (placeholder for time evolution).

### analyze(self, data)
**Описание:** Analyze data for this model.

Physical Meaning:
    Performs comprehensive analysis of the multi-particle system,
    including effective potential, collective modes, and correlations.

Args:
    data...

### _extract_spherical_field(self, center, radius)
**Описание:** Extract field in spherical region.

Physical Meaning:
    Extracts the phase field in a spherical
    region around the given center point.

## ./bhlff/models/level_f/nonlinear.py
Methods: 30

### __init__(self, system, nonlinear_params)
**Описание:** Initialize nonlinear effects model.

Physical Meaning:
    Sets up the model for studying nonlinear effects
    in the multi-particle system.

Args:
    system (MultiParticleSystem): Multi-particle sy...

### add_nonlinear_interactions(self, nonlinear_params)
**Описание:** Add nonlinear interactions to the system.

Physical Meaning:
    Introduces nonlinear terms into the effective
    potential and equations of motion.

Args:
    nonlinear_params (Dict): Nonlinear inte...

### find_nonlinear_modes(self)
**Описание:** Find nonlinear modes in the system.

Physical Meaning:
    Identifies nonlinear collective modes that
    arise from nonlinear interactions.

Returns:
    Dict containing:
        - frequencies: ω_n (...

### find_soliton_solutions(self)
**Описание:** Find solitonic solutions in the system.

Physical Meaning:
    Identifies solitonic solutions that arise
    from nonlinear interactions.

Returns:
    Dict containing:
        - solitons: list of sol...

### check_nonlinear_stability(self)
**Описание:** Check stability of nonlinear solutions.

Physical Meaning:
    Analyzes stability of nonlinear modes and
    solitonic solutions.

Returns:
    Dict containing:
        - linear_stability: linear stab...

### _setup_nonlinear_terms(self)
**Описание:** Setup nonlinear terms for the system.

Physical Meaning:
    Initializes nonlinear interaction terms
    based on the specified nonlinearity type.

### _setup_cubic_nonlinearity(self)
**Описание:** Setup cubic nonlinearity terms.

Physical Meaning:
    Initializes cubic nonlinear terms of the form
    g * |ψ|³ in the effective potential.

### _setup_quartic_nonlinearity(self)
**Описание:** Setup quartic nonlinearity terms.

Physical Meaning:
    Initializes quartic nonlinear terms of the form
    g * |ψ|⁴ in the effective potential.

### _setup_sine_gordon_nonlinearity(self)
**Описание:** Setup sine-Gordon nonlinearity terms.

Physical Meaning:
    Initializes sine-Gordon nonlinear terms of the form
    λ * sin(φ) in the effective potential.

### _add_nonlinear_potential(self)
**Описание:** Add nonlinear potential to the system.

Physical Meaning:
    Adds nonlinear potential terms to the
    effective potential of the system.

Mathematical Foundation:
    V_nonlinear = λ₁|φ|² + λ₂|φ|⁴ +...

### _add_nonlinear_dynamics(self)
**Описание:** Add nonlinear dynamics to the system.

Physical Meaning:
    Adds nonlinear terms to the equations
    of motion using 7D BVP theory.

Mathematical Foundation:
    ∂²φ/∂t² + ω₀²φ + λ₁φ + λ₂φ³ + λ₃φ⁵ =...

### _compute_nonlinear_corrections(self, linear_modes)
**Описание:** Compute nonlinear corrections to linear modes.

Physical Meaning:
    Calculates nonlinear corrections to the
    linear collective modes.

### _find_bifurcation_points(self)
**Описание:** Find bifurcation points in the system.

Physical Meaning:
    Identifies bifurcation points where
    the system behavior changes qualitatively.

### _analyze_nonlinear_stability(self)
**Описание:** Analyze stability of nonlinear modes.

Physical Meaning:
    Analyzes the stability of nonlinear
    collective modes.

### _find_sine_gordon_solitons(self)
**Описание:** Find sine-Gordon soliton solutions.

Physical Meaning:
    Identifies kink and antikink solitons
    in the sine-Gordon model.

### _find_cubic_solitons(self)
**Описание:** Find cubic nonlinearity soliton solutions.

Physical Meaning:
    Identifies solitons in the cubic
    nonlinear Schrödinger equation.

### _find_quartic_solitons(self)
**Описание:** Find quartic nonlinearity soliton solutions.

Physical Meaning:
    Identifies solitons in the quartic
    nonlinear system.

### _analyze_soliton_properties(self, solitons)
**Описание:** Analyze properties of soliton solutions.

Physical Meaning:
    Analyzes the properties of found
    soliton solutions.

### _compute_soliton_profile(self, soliton)
**Описание:** Compute soliton profile.

Physical Meaning:
    Calculates the spatial profile of
    the soliton solution.

### _analyze_linear_stability(self)
**Описание:** Analyze linear stability of nonlinear solutions.

Physical Meaning:
    Performs linear stability analysis
    of nonlinear solutions.

### _compute_growth_rates(self)
**Описание:** Compute instability growth rates.

Physical Meaning:
    Calculates growth rates for unstable
    modes in the system.

### _identify_stability_regions(self)
**Описание:** Identify stability regions in parameter space.

Physical Meaning:
    Identifies regions of parameter space
    where nonlinear solutions are stable.

### analyze(self, data)
**Описание:** Analyze data for this model.

Physical Meaning:
    Performs comprehensive analysis of nonlinear effects,
    including nonlinear modes and soliton solutions.

Args:
    data (Any): Input data to anal...

### _compute_effective_potential(self) -> callable
**Описание:** Compute effective potential including nonlinear terms.

### _compute_nonlinear_force(self, phi)
**Описание:** Compute nonlinear force terms.

### _compute_boundary_energy_exchange(self, field) -> float
**Описание:** Compute energy exchange through step resonator boundaries.

Physical Meaning:
    Calculates energy exchange between field and environment
    through semi-transparent step resonator boundaries.

### _compute_driving_force(self, t) -> float
**Описание:** Compute driving force.

### _formulate_equations_of_motion(self) -> callable
**Описание:** Formulate complete equations of motion without damping.

### potential(phi)
**Описание:** Нет докстринга

### equations(t, y)
**Описание:** Нет докстринга

## ./bhlff/models/level_f/transitions.py
Methods: 20

### __init__(self, system)
**Описание:** Initialize phase transitions model.

Physical Meaning:
    Sets up the model for studying phase transitions
    in the multi-particle system.

Args:
    system (MultiParticleSystem): Multi-particle sy...

### parameter_sweep(self, parameter, values)
**Описание:** Perform parameter sweep to study phase transitions.

Physical Meaning:
    Varies a system parameter and monitors
    the system state for phase transitions.

Args:
    parameter (str): Parameter to v...

### compute_order_parameters(self)
**Описание:** Compute order parameters for the system.

Physical Meaning:
    Calculates order parameters that characterize
    different phases of the system.

Returns:
    Dict containing:
        - topological_o...

### identify_critical_points(self, phase_diagram)
**Описание:** Identify critical points in phase diagram.

Physical Meaning:
    Finds critical points where phase transitions
    occur based on order parameter behavior.

Args:
    phase_diagram (List[Dict]): Phas...

### analyze_phase_stability(self)
**Описание:** Analyze stability of different phases.

Physical Meaning:
    Analyzes the stability of different phases
    in the system.

Returns:
    Dict containing stability analysis

### _setup_analysis_parameters(self)
**Описание:** Setup analysis parameters for phase transitions.

Physical Meaning:
    Initializes parameters needed for analysis
    of phase transitions.

### _update_system_parameter(self, parameter, value)
**Описание:** Update system parameter.

Physical Meaning:
    Updates the specified parameter in the system
    to the given value.

### _equilibrate_system(self)
**Описание:** Equilibrate system to new parameter values.

Physical Meaning:
    Allows the system to reach equilibrium
    under the new parameter values.

### _analyze_system_state(self)
**Описание:** Analyze current system state.

Physical Meaning:
    Analyzes the current state of the system
    including particle positions, phases, and energies.

### _compute_topological_order(self) -> float
**Описание:** Compute topological order parameter.

Physical Meaning:
    Calculates the total topological charge
    as a measure of topological order.

### _compute_phase_coherence(self) -> float
**Описание:** Compute phase coherence order parameter.

Physical Meaning:
    Calculates the phase coherence |⟨e^{iφ}⟩|
    as a measure of phase ordering.

### _compute_spatial_order(self) -> float
**Описание:** Compute spatial order parameter.

Physical Meaning:
    Calculates spatial correlation as a measure
    of spatial ordering.

### _compute_energy_density(self) -> float
**Описание:** Compute average energy density.

Physical Meaning:
    Calculates the average energy density
    of the system.

### _find_discontinuities(self, param_values, order_values)
**Описание:** Find discontinuities in order parameter.

Physical Meaning:
    Identifies first-order phase transitions
    by finding discontinuities in order parameters.

### _find_critical_points(self, param_values, order_values)
**Описание:** Find critical points in order parameter.

Physical Meaning:
    Identifies second-order phase transitions
    by finding critical points in order parameters.

### _compute_critical_exponents(self, param_values, order_values, critical_point)
**Описание:** Compute critical exponents for phase transition.

Physical Meaning:
    Calculates critical exponents α, β, γ, δ
    for the phase transition.

### _check_phase_stability(self, state)
**Описание:** Check stability of current phase.

Physical Meaning:
    Analyzes the stability of the current
    phase of the system.

### _analyze_phase_boundaries(self)
**Описание:** Analyze phase boundaries.

Physical Meaning:
    Identifies boundaries between different
    phases in parameter space.

### _identify_stability_regions(self)
**Описание:** Identify stability regions in parameter space.

Physical Meaning:
    Identifies regions of parameter space
    where different phases are stable.

### analyze(self, data)
**Описание:** Analyze data for this model.

Physical Meaning:
    Performs comprehensive analysis of phase transitions,
    including order parameters and critical points.

Args:
    data (Any): Input data to analy...

## ./bhlff/models/level_g/analysis.py
Methods: 20

### __init__(self, evolution_results, observational_data)
**Описание:** Initialize cosmological analysis.

Physical Meaning:
    Sets up the cosmological analysis with evolution results
    and optional observational data for comparison.

Args:
    evolution_results: Cosm...

### _setup_analysis_parameters(self)
**Описание:** Setup analysis parameters.

Physical Meaning:
    Initializes parameters for cosmological analysis,
    including statistical methods and comparison metrics.

### analyze_structure_formation(self)
**Описание:** Analyze structure formation process.

Physical Meaning:
    Analyzes the process of structure formation from
    phase field evolution and gravitational effects.

Returns:
    Structure formation anal...

### _analyze_structure_evolution(self)
**Описание:** Analyze structure evolution over time.

Physical Meaning:
    Analyzes how structure evolves over cosmological time,
    including growth rates and characteristic scales.

Returns:
    Structure evolu...

### _compute_growth_rate(self, rms_evolution) -> float
**Описание:** Compute structure growth rate.

Physical Meaning:
    Computes the rate at which structure grows during
    cosmological evolution.

Args:
    rms_evolution: RMS evolution over time

Returns:
    Grow...

### _compute_characteristic_timescale(self, rms_evolution) -> float
**Описание:** Compute characteristic timescale.

Physical Meaning:
    Computes the characteristic timescale for structure
    formation from the evolution data.

Args:
    rms_evolution: RMS evolution over time

R...

### _compute_formation_timescales(self)
**Описание:** Compute formation timescales.

Physical Meaning:
    Computes various timescales for structure formation,
    including characteristic formation times.

Returns:
    Formation timescales

### _compute_initial_growth_time(self, structure_formation) -> float
**Описание:** Compute initial growth time.

Physical Meaning:
    Computes the time for initial structure growth
    from the formation data.

Args:
    structure_formation: Structure formation data

Returns:
    I...

### _compute_maturation_time(self, structure_formation) -> float
**Описание:** Compute maturation time.

Physical Meaning:
    Computes the time for structure maturation
    from the formation data.

Args:
    structure_formation: Structure formation data

Returns:
    Maturatio...

### _compute_equilibrium_time(self, structure_formation) -> float
**Описание:** Compute equilibrium time.

Physical Meaning:
    Computes the time when structure reaches equilibrium
    from the formation data.

Args:
    structure_formation: Structure formation data

Returns:
  ...

### _compute_structure_statistics(self)
**Описание:** Compute structure statistics.

Physical Meaning:
    Computes statistical properties of structure formation,
    including mean, variance, and correlation properties.

Returns:
    Structure statistic...

### _compute_skewness(self, values) -> float
**Описание:** Compute skewness of values.

Physical Meaning:
    Computes the skewness (third moment) of the
    structure values.

Args:
    values: List of values

Returns:
    Skewness

### _compute_kurtosis(self, values) -> float
**Описание:** Compute kurtosis of values.

Physical Meaning:
    Computes the kurtosis (fourth moment) of the
    structure values.

Args:
    values: List of values

Returns:
    Kurtosis

### _analyze_correlations(self)
**Описание:** Analyze correlations in structure formation.

Physical Meaning:
    Analyzes correlations between different structure
    metrics and evolution parameters.

Returns:
    Correlation analysis

### _compute_correlation(self, x, y) -> float
**Описание:** Compute correlation coefficient.

Physical Meaning:
    Computes the Pearson correlation coefficient
    between two sets of values.

Args:
    x: First set of values
    y: Second set of values

Retu...

### compare_with_observations(self)
**Описание:** Compare results with observational data.

Physical Meaning:
    Compares the theoretical results with observational
    data to validate the model.

Returns:
    Comparison results

### _compare_structure_formation(self)
**Описание:** Compare structure formation with observations.

Physical Meaning:
    Compares the theoretical structure formation
    with observational data.

Returns:
    Structure formation comparison

### _compare_parameters(self)
**Описание:** Compare parameters with observations.

Physical Meaning:
    Compares the theoretical parameters with
    observational constraints.

Returns:
    Parameter comparison

### _compare_statistics(self)
**Описание:** Compare statistics with observations.

Physical Meaning:
    Compares the theoretical statistics with
    observational statistics.

Returns:
    Statistical comparison

### _compute_goodness_of_fit(self)
**Описание:** Compute goodness of fit metrics.

Physical Meaning:
    Computes various goodness of fit metrics
    to assess model quality.

Returns:
    Goodness of fit metrics

## ./bhlff/models/level_g/astrophysics.py
Methods: 17

### __init__(self, object_type, object_params)
**Описание:** Initialize astrophysical object model.

Physical Meaning:
    Creates a model for a specific type of astrophysical object
    with given physical parameters.

Args:
    object_type: Type of object ('s...

### _setup_object_model(self)
**Описание:** Setup object model based on type.

Physical Meaning:
    Initializes the phase field model for the specific
    astrophysical object type.

### _setup_star_model(self)
**Описание:** Setup star model.

Physical Meaning:
    Creates a phase field model for a star with
    exponential radial profile and phase structure.

### _create_star_phase_profile(self)
**Описание:** Create phase profile for star.

Physical Meaning:
    Creates the phase field profile for a star:
    a(r) = A₀ exp(-r/R_s) cos(φ(r))

Returns:
    Star phase field profile

### _setup_galaxy_model(self)
**Описание:** Setup galaxy model.

Physical Meaning:
    Creates a phase field model for a galaxy with
    spiral structure and collective phase patterns.

### _create_galaxy_phase_profile(self)
**Описание:** Create phase profile for galaxy.

Physical Meaning:
    Creates the phase field profile for a galaxy:
    a(r,θ) = A(r) exp(i(mθ + φ(r)))

Returns:
    Galaxy phase field profile

### _setup_black_hole_model(self)
**Описание:** Setup black hole model.

Physical Meaning:
    Creates a phase field model for a black hole with
    extreme phase defect and strong curvature.

### _create_black_hole_phase_profile(self)
**Описание:** Create phase profile for black hole.

Physical Meaning:
    Creates the phase field profile for a black hole:
    a(r) = A₀ (r/r_s)^(-α) exp(iφ(r))

Returns:
    Black hole phase field profile

### create_star_model(self, stellar_params)
**Описание:** Create star model with given parameters.

Physical Meaning:
    Creates a star model with specified stellar parameters
    and phase field configuration.

Args:
    stellar_params: Stellar parameters
...

### create_galaxy_model(self, galactic_params)
**Описание:** Create galaxy model with given parameters.

Physical Meaning:
    Creates a galaxy model with specified galactic parameters
    and spiral structure.

Args:
    galactic_params: Galactic parameters

R...

### create_black_hole_model(self, bh_params)
**Описание:** Create black hole model with given parameters.

Physical Meaning:
    Creates a black hole model with specified parameters
    and extreme phase defect.

Args:
    bh_params: Black hole parameters

Re...

### analyze_phase_properties(self)
**Описание:** Analyze phase properties of the object.

Physical Meaning:
    Analyzes the phase field properties of the astrophysical
    object, including topological characteristics.

Returns:
    Phase propertie...

### _compute_phase_correlation_length(self) -> float
**Описание:** Compute phase correlation length.

Physical Meaning:
    Computes the characteristic length scale over which
    the phase field is correlated.

Returns:
    Correlation length

### compute_observable_properties(self)
**Описание:** Compute observable properties of the object.

Physical Meaning:
    Computes observable properties that can be compared
    with astronomical observations.

Returns:
    Observable properties

### _compute_effective_radius(self) -> float
**Описание:** Compute effective radius of the object.

Physical Meaning:
    Computes the effective radius where the phase field
    amplitude drops to 1/e of its maximum value.

Returns:
    Effective radius

### _compute_phase_energy(self) -> float
**Описание:** Compute phase field energy.

Physical Meaning:
    Computes the total energy associated with the
    phase field configuration.

Returns:
    Phase field energy

### _compute_defect_density(self) -> float
**Описание:** Compute topological defect density.

Physical Meaning:
    Computes the density of topological defects in
    the phase field configuration.

Returns:
    Defect density

## ./bhlff/models/level_g/bvp_integration.py
Methods: 11

### __init__(self, bvp_core)
**Описание:** Initialize Level G BVP integration.

Physical Meaning:
    Sets up integration between Level G models and BVP framework,
    providing access to BVP core functionality and specialized
    Level G anal...

### process_bvp_data(self, envelope)
**Описание:** Process BVP data for Level G operations.

Physical Meaning:
    Analyzes cosmological evolution, large-scale structure formation,
    astrophysical object formation, and gravitational effects in BVP
 ...

### _analyze_cosmological_evolution(self, envelope, scale)
**Описание:** Analyze cosmological evolution in BVP envelope.

Physical Meaning:
    Analyzes the cosmological evolution of the BVP envelope,
    including scale factor evolution, field evolution,
    and cosmologi...

### _analyze_large_scale_structure(self, envelope, threshold)
**Описание:** Analyze large-scale structure in BVP envelope.

Physical Meaning:
    Analyzes the large-scale structure formation in the BVP envelope,
    including structure identification, clustering analysis,
   ...

### _analyze_astrophysical_objects(self, envelope, threshold)
**Описание:** Analyze astrophysical objects in BVP envelope.

Physical Meaning:
    Analyzes the formation and evolution of astrophysical objects
    in the BVP envelope, including stars, galaxies, and other
    as...

### _analyze_gravitational_effects(self, envelope, threshold)
**Описание:** Analyze gravitational effects in BVP envelope.

Physical Meaning:
    Analyzes gravitational effects on the BVP envelope,
    including gravitational lensing, gravitational waves,
    and gravitationa...

### _analyze_bvp_integration(self, envelope)
**Описание:** Analyze BVP-specific integration aspects.

Physical Meaning:
    Analyzes how the BVP envelope integrates with Level G
    models, including cosmological parameters, envelope modulation
    effects on...

### _analyze_cosmological_parameters(self, envelope)
**Описание:** Analyze cosmological parameters in BVP envelope.

### _analyze_envelope_cosmological_coupling(self, envelope)
**Описание:** Analyze coupling between envelope and cosmological evolution.

### _analyze_gravitational_envelope_effects(self, envelope)
**Описание:** Analyze gravitational effects on envelope.

### _check_bvp_compliance(self, envelope)
**Описание:** Check BVP framework compliance.

## ./bhlff/models/level_g/cosmology.py
Methods: 14

### __init__(self, initial_conditions, cosmology_params)
**Описание:** Initialize cosmological model.

Physical Meaning:
    Sets up the cosmological model with initial conditions
    and cosmological parameters for universe evolution.

Args:
    initial_conditions: Init...

### compute_effective_metric_from_vbp_envelope(self, envelope_invariants)
**Описание:** Compute effective metric from VBP envelope dynamics.

Physical Meaning:
    Computes the effective metric g_eff[Θ] using only VBP envelope
    invariants (no spacetime curvature, no scale factors).

M...

### compute_scale_factor(self, t) -> float
**Описание:** Compute scale factor for cosmological evolution.

Physical Meaning:
    Computes a simple scale factor for cosmological evolution
    based on the envelope effective metric parameters.
    
Args:
    ...

### _setup_evolution_parameters(self)
**Описание:** Setup evolution parameters.

Physical Meaning:
    Initializes parameters for cosmological evolution,
    including time steps and physical constants.

### evolve_universe(self, time_range)
**Описание:** Evolve universe from initial to final time.

Physical Meaning:
    Evolves the universe from initial conditions through
    cosmological time, computing phase field evolution
    and structure formati...

### _initialize_phase_field(self)
**Описание:** Initialize phase field from initial conditions.

Physical Meaning:
    Creates initial phase field configuration based on
    cosmological initial conditions.

Returns:
    Initial phase field configu...

### _evolve_phase_field_step(self, t, dt, scale_factor)
**Описание:** Evolve phase field for one time step.

Physical Meaning:
    Advances the phase field configuration by one time step
    using the cosmological evolution equation.

Mathematical Foundation:
    ∂²a/∂t...

### _compute_hubble_parameter(self, t) -> float
**Описание:** Compute Hubble parameter at time t.

Physical Meaning:
    Computes the Hubble parameter H(t) for the expanding
    universe at cosmological time t.

Mathematical Foundation:
    H(t) = H0 * sqrt(Ω_Λ)...

### _analyze_structure_at_time(self, t)
**Описание:** Analyze structure formation at given time.

Physical Meaning:
    Analyzes the formation of large-scale structure
    from phase field evolution at cosmological time t.

Args:
    t: Cosmological time...

### _compute_correlation_length(self) -> float
**Описание:** Compute correlation length of phase field.

Physical Meaning:
    Computes the characteristic length scale over which
    the phase field is correlated.

Returns:
    Correlation length

### _count_topological_defects(self) -> int
**Описание:** Count topological defects in phase field.

Physical Meaning:
    Counts the number of topological defects (vortices,
    monopoles, etc.) in the current phase field configuration.

Returns:
    Number...

### analyze_structure_formation(self)
**Описание:** Analyze large-scale structure formation.

Physical Meaning:
    Analyzes the overall process of structure formation
    throughout cosmological evolution.

Returns:
    Structure formation analysis

### _compute_structure_growth_rate(self) -> float
**Описание:** Compute structure growth rate.

Physical Meaning:
    Computes the rate at which large-scale structure
    grows during cosmological evolution.

Returns:
    Structure growth rate

### compute_cosmological_parameters(self)
**Описание:** Compute cosmological parameters from evolution.

Physical Meaning:
    Computes derived cosmological parameters from
    the evolution results.

Returns:
    Dictionary of cosmological parameters

## ./bhlff/models/level_g/evolution.py
Methods: 16

### __init__(self, initial_conditions, cosmology_params)
**Описание:** Initialize cosmological evolution model.

Physical Meaning:
    Sets up the cosmological evolution model with initial
    conditions and cosmological parameters.

Args:
    initial_conditions: Initial...

### _setup_evolution_parameters(self)
**Описание:** Setup evolution parameters.

Physical Meaning:
    Initializes parameters for cosmological evolution,
    including time steps and physical constants.

### evolve_cosmology(self, time_range)
**Описание:** Evolve cosmology from initial to final time.

Physical Meaning:
    Evolves the cosmology from initial conditions through
    cosmological time, computing phase field evolution
    and structure forma...

### _compute_scale_factor(self, t) -> float
**Описание:** Compute scale factor at time t.

Physical Meaning:
    Computes the scale factor a(t) for the expanding
    universe at cosmological time t.

Mathematical Foundation:
    a(t) = a0 * exp(H0 * t) for Λ...

### _compute_hubble_parameter(self, t) -> float
**Описание:** Compute Hubble parameter at time t.

Physical Meaning:
    Computes the Hubble parameter H(t) for the expanding
    universe at cosmological time t.

Mathematical Foundation:
    H(t) = H0 * sqrt(Ω_Λ)...

### _initialize_phase_field(self)
**Описание:** Initialize phase field from initial conditions.

Physical Meaning:
    Creates initial phase field configuration based on
    cosmological initial conditions.

Returns:
    Initial phase field configu...

### _evolve_phase_field_step(self, t, dt, scale_factor)
**Описание:** Evolve phase field for one time step.

Physical Meaning:
    Advances the phase field configuration by one time step
    using the cosmological evolution equation.

Mathematical Foundation:
    ∂²a/∂t...

### _analyze_structure_at_time(self, t, phase_field)
**Описание:** Analyze structure formation at given time.

Physical Meaning:
    Analyzes the formation of large-scale structure
    from phase field evolution at cosmological time t.

Args:
    t: Cosmological time...

### _compute_correlation_length(self, phase_field) -> float
**Описание:** Compute correlation length of phase field.

Physical Meaning:
    Computes the characteristic length scale over which
    the phase field is correlated.

Args:
    phase_field: Phase field configurati...

### _count_topological_defects(self, phase_field) -> int
**Описание:** Count topological defects in phase field.

Physical Meaning:
    Counts the number of topological defects (vortices,
    monopoles, etc.) in the current phase field configuration.

Args:
    phase_fie...

### _compute_structure_growth_rate(self, phase_field) -> float
**Описание:** Compute structure growth rate.

Physical Meaning:
    Computes the rate at which large-scale structure
    grows from the phase field evolution.

Args:
    phase_field: Phase field configuration

Retu...

### _compute_cosmological_parameters(self, t, scale_factor)
**Описание:** Compute cosmological parameters at time t.

Physical Meaning:
    Computes derived cosmological parameters from
    the evolution at time t.

Args:
    t: Cosmological time
    scale_factor: Current s...

### analyze_cosmological_evolution(self)
**Описание:** Analyze cosmological evolution results.

Physical Meaning:
    Analyzes the overall cosmological evolution process,
    including structure formation and parameter evolution.

Returns:
    Cosmologica...

### _compute_structure_formation_rate(self) -> float
**Описание:** Compute structure formation rate.

Physical Meaning:
    Computes the rate at which large-scale structure
    forms during cosmological evolution.

Returns:
    Structure formation rate

### _analyze_parameter_evolution(self)
**Описание:** Analyze cosmological parameter evolution.

Physical Meaning:
    Analyzes the evolution of cosmological parameters
    throughout the cosmological evolution.

Returns:
    Parameter evolution analysis

### _compute_parameter_trends(self, cosmological_params)
**Описание:** Compute parameter trends.

Physical Meaning:
    Computes the trends in cosmological parameters
    throughout the evolution.

Args:
    cosmological_params: List of cosmological parameters

Returns:
...

## ./bhlff/models/level_g/gravity.py
Methods: 8

### __init__(self, system, gravity_params)
**Описание:** Initialize VBP envelope gravitational effects model.

Physical Meaning:
    Sets up the gravitational effects model with specialized
    calculators for envelope curvature, phase envelope balance,
   ...

### _setup_gravitational_parameters(self)
**Описание:** Setup VBP envelope gravitational parameters.

Physical Meaning:
    Initializes gravitational parameters for VBP envelope dynamics
    including phase velocity, bridge parameters, and coupling constan...

### compute_effective_metric(self)
**Описание:** Compute effective metric from VBP envelope.

Physical Meaning:
    Computes the effective metric g_eff[Θ] from the
    VBP envelope dynamics. This metric describes the
    geometry of the VBP envelope...

### _get_phase_field_from_system(self)
**Описание:** Get phase field from system.

Physical Meaning:
    Extracts the phase field configuration from the
    system for gravitational calculations.

### _create_default_phase_field(self)
**Описание:** Create default phase field for testing.

Physical Meaning:
    Creates a simple phase field configuration for
    gravitational calculations when no field is available.

### analyze_envelope_curvature(self)
**Описание:** Analyze VBP envelope curvature.

Physical Meaning:
    Computes and analyzes all aspects of VBP envelope
    curvature including envelope curvature descriptors,
    anisotropy measures, and focusing r...

### compute_gravitational_waves(self)
**Описание:** Compute gravitational waves from VBP envelope dynamics.

Physical Meaning:
    Calculates gravitational waves generated by the
    VBP envelope dynamics. Waves propagate at c_T=c_φ
    and follow GW-1...

### compute_envelope_effects(self)
**Описание:** Compute all VBP envelope gravitational effects.

Physical Meaning:
    Calculates all gravitational effects from VBP envelope
    dynamics including envelope curvature, gravitational waves,
    and ef...

## ./bhlff/models/level_g/gravity_curvature.py
Methods: 9

### __init__(self, domain, params)
**Описание:** Initialize curvature calculator.

Physical Meaning:
    Sets up the computational framework for curvature
    calculations with appropriate numerical parameters.

Args:
    domain: Computational domai...

### _setup_curvature_parameters(self)
**Описание:** Setup parameters for curvature calculations.

Physical Meaning:
    Initializes numerical parameters for curvature
    calculations including resolution and precision.

### compute_envelope_curvature(self, phase_field)
**Описание:** Compute VBP envelope curvature descriptors.

Physical Meaning:
    Calculates envelope curvature descriptors from the phase field Θ(x,φ,t).
    The curvature describes the distortion of the VBP envelo...

### _compute_phase_gradients(self, phase_field)
**Описание:** Compute gradients of the phase field.

Physical Meaning:
    Calculates the gradients of the phase field Θ(x,φ,t) with respect to
    spatial coordinates x and phase coordinates φ. These gradients are...

### _compute_effective_metric(self, phase_field, phase_gradients)
**Описание:** Compute effective metric from phase field.

Physical Meaning:
    Calculates the effective metric g_eff[Θ] from the phase field configuration.
    This metric describes the geometry of the VBP envelop...

### _compute_envelope_invariants(self, phase_gradients, g_eff)
**Описание:** Compute envelope curvature invariants.

Physical Meaning:
    Calculates scalar invariants of the envelope curvature that are
    independent of coordinate system choice. These replace classical
    c...

### _compute_anisotropy_index(self, g_eff) -> float
**Описание:** Compute anisotropy index of the effective metric.

Physical Meaning:
    Calculates the anisotropy index which measures deviation from
    isotropic envelope configuration. This is a key descriptor
  ...

### _compute_focusing_rate(self, phase_gradients, g_eff) -> float
**Описание:** Compute focusing rate of the envelope.

Physical Meaning:
    Calculates the focusing rate which describes how the envelope
    focuses or defocuses wavefronts. This is related to the
    energy argum...

### compute_envelope_invariants(self, phase_field)
**Описание:** Compute envelope curvature invariants from phase field.

Physical Meaning:
    Calculates scalar invariants of the envelope curvature that are
    independent of coordinate system choice. These replac...

## ./bhlff/models/level_g/gravity_einstein.py
Methods: 10

### __init__(self, domain, params)
**Описание:** Initialize phase envelope balance solver.

Physical Meaning:
    Sets up the computational framework for solving
    phase envelope balance equations with VBP envelope dynamics.

Args:
    domain: Com...

### _setup_envelope_parameters(self)
**Описание:** Setup parameters for phase envelope balance equations.

Physical Meaning:
    Initializes physical constants and numerical
    parameters for phase envelope balance solution.

### solve_phase_envelope_balance(self, phase_field)
**Описание:** Solve phase envelope balance equation for VBP envelope dynamics.

Physical Meaning:
    Solves the phase envelope balance equation D[Θ] = source where
    the balance operator D includes time memory (...

### _build_balance_operator(self, phase_field)
**Описание:** Build balance operator D for phase envelope equation.

Physical Meaning:
    Constructs the balance operator D[Θ] = source that includes
    time memory (Γ,K) and spatial (−Δ)^β terms with c_φ(a,k), χ...

### _build_memory_kernels(self, phase_field)
**Описание:** Build time memory kernels (Γ,K) for envelope dynamics.

Physical Meaning:
    Constructs memory kernels that describe the temporal evolution
    of the VBP envelope. These kernels encode the memory ef...

### _build_spatial_operator(self, phase_field)
**Описание:** Build spatial fractional Laplacian operator.

Physical Meaning:
    Constructs the spatial operator (−Δ)^β that describes
    the fractional diffusion in the VBP envelope dynamics.

Mathematical Found...

### _build_bridge_terms(self, phase_field)
**Описание:** Build bridge terms (χ/κ) for envelope dynamics.

Physical Meaning:
    Constructs the bridge terms that connect the phase field
    to the effective metric through the χ/κ parameter.

Mathematical Fou...

### _solve_envelope_balance(self, balance_operator, phase_field)
**Описание:** Solve envelope balance equation.

Physical Meaning:
    Solves the envelope balance equation D[Θ] = source using
    the constructed balance operator and phase field configuration.

Mathematical Found...

### _apply_balance_operator(self, balance_operator, solution)
**Описание:** Apply balance operator to solution.

Physical Meaning:
    Applies the balance operator D[Θ] to the current solution
    to compute the residual for the envelope balance equation.

Args:
    balance_o...

### _compute_effective_metric_from_solution(self, solution)
**Описание:** Compute effective metric from envelope solution.

Physical Meaning:
    Computes the effective metric g_eff[Θ] from the envelope solution.
    This metric describes the geometry of the VBP envelope an...

## ./bhlff/models/level_g/gravity_waves.py
Methods: 12

### __init__(self, domain, params)
**Описание:** Initialize gravitational waves calculator.

Physical Meaning:
    Sets up the computational framework for gravitational
    wave calculations with appropriate parameters.

Args:
    domain: Computatio...

### _setup_wave_parameters(self)
**Описание:** Setup parameters for VBP gravitational wave calculations.

Physical Meaning:
    Initializes parameters for gravitational wave
    calculations from VBP envelope dynamics including
    frequency range...

### compute_gravitational_waves(self, envelope_solution)
**Описание:** Compute gravitational waves from VBP envelope dynamics.

Physical Meaning:
    Calculates gravitational waves generated by the
    VBP envelope dynamics, including strain tensor,
    amplitude, and fr...

### _compute_strain_tensor_from_envelope(self, envelope_solution)
**Описание:** Compute gravitational wave strain tensor from VBP envelope dynamics.

Physical Meaning:
    Calculates the strain tensor h_μν from the VBP envelope dynamics.
    The strain represents the gravitationa...

### _compute_envelope_oscillations(self, envelope_solution)
**Описание:** Compute envelope oscillations from VBP solution.

Physical Meaning:
    Calculates the oscillations of the VBP envelope that
    generate gravitational waves. These oscillations
    represent the dyna...

### _compute_wave_amplitude_with_gw1_law(self, strain_tensor) -> float
**Описание:** Compute gravitational wave amplitude with GW-1 law.

Physical Meaning:
    Calculates the gravitational wave amplitude following
    the GW-1 amplitude law: |h|∝a^{-1} when Γ=K=0.
    This law describ...

### _compute_frequency_spectrum(self, strain_tensor)
**Описание:** Compute frequency spectrum of gravitational waves from VBP envelope.

Physical Meaning:
    Calculates the frequency spectrum of gravitational
    waves from VBP envelope dynamics using Fourier analys...

### _compute_polarization_modes(self, strain_tensor)
**Описание:** Compute polarization modes of gravitational waves from VBP envelope.

Physical Meaning:
    Calculates the polarization modes of gravitational waves
    from the VBP envelope dynamics. In 7D theory, a...

### _compute_wave_energy(self, strain_tensor) -> float
**Описание:** Compute energy carried by gravitational waves from VBP envelope.

Physical Meaning:
    Calculates the energy density of gravitational
    waves from the VBP envelope dynamics. The energy
    is carri...

### _compute_time_derivative(self, strain_tensor) -> float
**Описание:** Compute time derivative of strain tensor from VBP envelope.

Physical Meaning:
    Calculates the time derivative of the strain
    tensor from VBP envelope dynamics for energy computation.

### _compute_spatial_gradient(self, strain_tensor) -> float
**Описание:** Compute spatial gradient of strain tensor from VBP envelope.

Physical Meaning:
    Calculates the spatial gradient of the strain
    tensor from VBP envelope dynamics for energy computation.

### compute_detection_sensitivity(self, amplitude, frequency)
**Описание:** Compute detection sensitivity for VBP gravitational waves.

Physical Meaning:
    Calculates the detectability of gravitational waves
    from VBP envelope dynamics based on amplitude and frequency.
 ...

## ./bhlff/models/level_g/structure.py
Methods: 18

### __init__(self, initial_fluctuations, evolution_params)
**Описание:** Initialize large-scale structure model.

Physical Meaning:
    Sets up the large-scale structure model with initial
    density fluctuations and evolution parameters.

Args:
    initial_fluctuations: ...

### _setup_structure_parameters(self)
**Описание:** Setup structure parameters.

Physical Meaning:
    Initializes parameters for large-scale structure
    formation and evolution.

### evolve_structure(self, time_range)
**Описание:** Evolve large-scale structure formation.

Physical Meaning:
    Evolves the large-scale structure from initial
    fluctuations through cosmological time.

Mathematical Foundation:
    Integrates the d...

### _evolve_density_field(self, t, dt)
**Описание:** Evolve density field for one time step.

Physical Meaning:
    Advances the density field by one time step using
    the continuity equation and gravitational effects.

Mathematical Foundation:
    ∂ρ...

### _evolve_velocity_field(self, t, dt)
**Описание:** Evolve velocity field for one time step.

Physical Meaning:
    Advances the velocity field by one time step using
    the Euler equation and gravitational effects.

Mathematical Foundation:
    ∂v/∂t...

### _evolve_potential_field(self, t, dt)
**Описание:** Evolve gravitational potential field.

Physical Meaning:
    Advances the gravitational potential by one time step
    using the Poisson equation.

Mathematical Foundation:
    ∇²Φ = 4πGρ

Args:
    t...

### _compute_velocity_divergence(self)
**Описание:** Compute velocity field divergence.

Physical Meaning:
    Computes the divergence of the velocity field
    for the continuity equation.

Returns:
    Velocity divergence

### _compute_gravitational_acceleration(self)
**Описание:** Compute gravitational acceleration.

Physical Meaning:
    Computes the gravitational acceleration from
    the gravitational potential.

Returns:
    Gravitational acceleration

### _solve_poisson_equation(self, density)
**Описание:** Solve Poisson equation for gravitational potential.

Physical Meaning:
    Solves the Poisson equation ∇²Φ = 4πGρ to find
    the gravitational potential.

Mathematical Foundation:
    ∇²Φ = 4πGρ

Arg...

### _analyze_structure_at_time(self, t)
**Описание:** Analyze structure at given time.

Physical Meaning:
    Analyzes the large-scale structure at cosmological
    time t, including density peaks and correlations.

Args:
    t: Cosmological time

Return...

### _compute_density_correlation_length(self) -> float
**Описание:** Compute density correlation length.

Physical Meaning:
    Computes the characteristic length scale over which
    the density field is correlated.

Returns:
    Correlation length

### _count_density_peaks(self) -> int
**Описание:** Count density peaks (galaxy candidates).

Physical Meaning:
    Counts the number of density peaks that could
    correspond to galaxy formation sites.

Returns:
    Number of density peaks

### _compute_cluster_mass(self) -> float
**Описание:** Compute total cluster mass.

Physical Meaning:
    Computes the total mass in high-density regions
    that could correspond to galaxy clusters.

Returns:
    Total cluster mass

### analyze_galaxy_formation(self)
**Описание:** Analyze galaxy formation process.

Physical Meaning:
    Analyzes the process of galaxy formation from
    density fluctuations and gravitational collapse.

Returns:
    Galaxy formation analysis

### _count_total_galaxies(self) -> int
**Описание:** Count total number of galaxies.

Physical Meaning:
    Counts the total number of galaxies formed during
    structure evolution.

Returns:
    Total number of galaxies

### _compute_galaxy_mass_distribution(self)
**Описание:** Compute galaxy mass distribution.

Physical Meaning:
    Computes the distribution of galaxy masses formed
    during structure evolution.

Returns:
    Galaxy mass distribution

### _compute_formation_timescale(self) -> float
**Описание:** Compute galaxy formation timescale.

Physical Meaning:
    Computes the characteristic timescale for galaxy
    formation from density fluctuations.

Returns:
    Formation timescale

### _compute_galaxy_correlation(self)
**Описание:** Compute galaxy correlation function.

Physical Meaning:
    Computes the correlation function between galaxies
    formed during structure evolution.

Returns:
    Galaxy correlation function

## ./bhlff/models/level_g/validation.py
Methods: 19

### __init__(self, inversion_results, validation_criteria, experimental_data)
**Описание:** Initialize particle validation.

Physical Meaning:
    Sets up the particle validation with inversion results
    and validation criteria.

Args:
    inversion_results: Results from parameter inversio...

### _setup_inversion_parameters(self)
**Описание:** Setup inversion parameters.

Physical Meaning:
    Initializes parameters for particle inversion,
    including optimization settings and loss functions.

### invert_parameters(self)
**Описание:** Invert model parameters from observables.

Physical Meaning:
    Reconstructs the fundamental model parameters from
    observable particle properties using optimization.

Mathematical Foundation:
   ...

### _initialize_parameters(self)
**Описание:** Initialize parameters from priors.

Physical Meaning:
    Initializes the model parameters from prior
    distributions for optimization.

Returns:
    Initial parameter values

### _optimize_parameters(self, initial_params)
**Описание:** Optimize parameters using loss function.

Physical Meaning:
    Optimizes the model parameters to minimize the
    loss function with respect to observables.

Args:
    initial_params: Initial paramet...

### _compute_loss(self, params) -> float
**Описание:** Compute loss function.

Physical Meaning:
    Computes the loss function that measures the
    discrepancy between model predictions and observables.

Mathematical Foundation:
    L(θ) = Σ_k w_k d_k(m...

### _compute_model_predictions(self, params)
**Описание:** Compute model predictions for given parameters.

Physical Meaning:
    Computes the model predictions for observable
    metrics given the model parameters.

Args:
    params: Model parameters

Return...

### _compute_distance_metric(self, obs_value, mod_value, metric_name) -> float
**Описание:** Compute distance metric between observed and model values.

Physical Meaning:
    Computes the distance between observed and model
    values for a specific metric.

Args:
    obs_value: Observed valu...

### _compute_regularization(self, params) -> float
**Описание:** Compute regularization term.

Physical Meaning:
    Computes the regularization term to prevent
    overfitting and ensure physical constraints.

Args:
    params: Model parameters

Returns:
    Regul...

### _compute_gradients(self, params)
**Описание:** Compute gradients of loss function.

Physical Meaning:
    Computes the gradients of the loss function
    with respect to the model parameters.

Args:
    params: Model parameters

Returns:
    Param...

### _get_convergence_info(self)
**Описание:** Get convergence information.

Physical Meaning:
    Returns information about the convergence of
    the optimization process.

Returns:
    Convergence information

### _compute_parameter_uncertainties(self, params)
**Описание:** Compute parameter uncertainties.

Physical Meaning:
    Computes the uncertainties in the optimized
    parameters using statistical methods.

Args:
    params: Optimized parameters

Returns:
    Para...

### _setup_validation_parameters(self)
**Описание:** Setup validation parameters.

Physical Meaning:
    Initializes parameters for particle validation,
    including validation thresholds and criteria.

### validate_parameters(self)
**Описание:** Validate inverted parameters.

Physical Meaning:
    Validates the inverted parameters against
    experimental data and physical constraints.

Returns:
    Validation results

### _validate_parameters(self)
**Описание:** Validate parameter values.

Physical Meaning:
    Validates that the inverted parameters are
    within reasonable physical ranges.

Returns:
    Parameter validation results

### _validate_energy_balance(self)
**Описание:** Validate energy balance.

Physical Meaning:
    Validates that the energy balance is conserved
    in the phase field configuration.

Returns:
    Energy balance validation results

### _validate_physical_constraints(self)
**Описание:** Validate physical constraints.

Physical Meaning:
    Validates that the inverted parameters satisfy
    physical constraints and conservation laws.

Returns:
    Physical constraint validation result...

### _validate_experimental_data(self)
**Описание:** Validate against experimental data.

Physical Meaning:
    Validates that the inverted parameters reproduce
    experimental observations within uncertainties.

Returns:
    Experimental validation re...

### _compute_overall_validation(self)
**Описание:** Compute overall validation result.

Physical Meaning:
    Computes the overall validation result based on
    all validation tests.

Returns:
    Overall validation results

## ./bhlff/models/levels/bvp_integration_base.py
Methods: 5

### __init__(self, bvp_core)
**Описание:** Initialize BVP level integration base.

Physical Meaning:
    Sets up the integration interface with the BVP core framework,
    establishing the connection between BVP and level-specific operations.
...

### integrate_level(self, envelope)
**Описание:** Integrate BVP envelope with specific level.

Physical Meaning:
    Transforms BVP envelope data according to level-specific
    requirements while maintaining BVP framework compliance.

Args:
    enve...
**Декораторы:** abstractmethod

### validate_envelope(self, envelope) -> bool
**Описание:** Validate BVP envelope data.

Physical Meaning:
    Ensures that the BVP envelope data is physically meaningful
    and mathematically consistent before processing.

Args:
    envelope (np.ndarray): BV...

### get_bvp_constants(self)
**Описание:** Get BVP constants and configuration.

Physical Meaning:
    Retrieves the BVP constants and configuration parameters
    needed for level-specific operations.

Returns:
    Dict[str, Any]: BVP constan...

### __repr__(self) -> str
**Описание:** String representation of integration base.

## ./bhlff/models/levels/bvp_integration_core.py
Methods: 13

### __init__(self, bvp_core)
**Описание:** Initialize BVP integration core.

Physical Meaning:
    Sets up the integration core with the BVP framework,
    establishing the central coordination point for all
    level integrations.

Args:
    ...

### integrate_level_a(self, envelope)
**Описание:** Integrate BVP with Level A (basic solvers).

Physical Meaning:
    Integrates BVP framework with Level A basic solvers,
    ensuring that fundamental solver operations are
    properly coordinated wit...

### integrate_level_b(self, envelope)
**Описание:** Integrate BVP with Level B (fundamental properties).

Physical Meaning:
    Integrates BVP framework with Level B fundamental
    properties analysis, ensuring that power law behavior
    and fundamen...

### integrate_level_c(self, envelope)
**Описание:** Integrate BVP with Level C (boundaries and cells).

Physical Meaning:
    Integrates BVP framework with Level C boundary
    and cell analysis, ensuring that boundary effects,
    resonators, and memo...

### _integrate_bvp_with_level_a(self, envelope, validation_results)
**Описание:** Integrate BVP with Level A operations.

### _integrate_bvp_with_level_b(self, envelope, power_law_results)
**Описание:** Integrate BVP with Level B operations.

### _integrate_bvp_with_level_c(self, envelope)
**Описание:** Integrate BVP with Level C operations.

### _calculate_bvp_metrics(self, bvp_solution)
**Описание:** Calculate BVP solution metrics.

### _calculate_integration_metrics(self, envelope, bvp_results)
**Описание:** Calculate integration metrics.

### _calculate_integration_quality(self, envelope, bvp_solution) -> float
**Описание:** Calculate integration quality metric.

### _calculate_convergence_metrics(self, bvp_solution)
**Описание:** Calculate convergence metrics.

### _integrate_power_law_analysis(self, bvp_solution, power_law_results)
**Описание:** Integrate power law analysis with BVP solution.

### _compare_power_law_results(self, original_results, solution_results)
**Описание:** Compare power law results.

## ./bhlff/solvers/base/abstract_solver.py
Methods: 8

### __init__(self, domain, parameters)
**Описание:** Initialize abstract solver.

Physical Meaning:
    Sets up the solver with computational domain and physics
    parameters, preparing for numerical solution of phase field
    equations.

Args:
    do...

### solve(self, source)
**Описание:** Solve the phase field equation for given source.

Physical Meaning:
    Computes the phase field configuration that satisfies
    the governing equation with the given source term,
    representing th...
**Декораторы:** abstractmethod

### solve_time_evolution(self, initial_field, source, time_steps, dt)
**Описание:** Solve time evolution of the phase field.

Physical Meaning:
    Computes the time evolution of the phase field from initial
    conditions under the influence of a time-dependent source,
    represent...
**Декораторы:** abstractmethod

### validate_input(self, field, name)
**Описание:** Validate input field shape and properties.

Physical Meaning:
    Ensures that input fields have the correct shape and properties
    for the computational domain, preventing numerical errors and
    ...

### compute_residual(self, field, source)
**Описание:** Compute residual of the governing equation.

Physical Meaning:
    Computes the residual r = L_β a - s to measure how well
    the field satisfies the governing equation, used for
    convergence chec...

### get_energy(self, field) -> float
**Описание:** Compute energy of the field configuration.

Physical Meaning:
    Computes the total energy of the field configuration,
    representing the energy content of the phase field
    system in the current...

### is_initialized(self) -> bool
**Описание:** Check if solver is initialized.

Physical Meaning:
    Returns whether the solver has been properly initialized
    and is ready for computations.

Returns:
    bool: True if solver is initialized, Fa...

### __repr__(self) -> str
**Описание:** String representation of the solver.

## ./bhlff/solvers/integrators/bvp_evolution_computer.py
Methods: 6

### __init__(self, domain, config)
**Описание:** Initialize BVP evolution computer.

Physical Meaning:
    Sets up the BVP evolution computer with carrier frequency
    and modulation parameters for evolution computation.

Args:
    domain (Domain):...

### compute_bvp_evolution(self, field)
**Описание:** Compute BVP evolution terms.

Physical Meaning:
    Computes the BVP-specific evolution terms F_BVP(a, t) for
    the temporal evolution of the phase field configuration.

Mathematical Foundation:
   ...

### _compute_bvp_terms(self, field)
**Описание:** Compute core BVP evolution terms.

Physical Meaning:
    Computes the core BVP evolution terms representing the
    fundamental dynamics of the BVP field configuration.

Mathematical Foundation:
    C...

### _compute_bvp_nonlinear_terms_spectral(self, field_spectral)
**Описание:** Compute BVP nonlinear terms in spectral space.

Physical Meaning:
    Computes nonlinear self-interaction terms of the BVP field
    in spectral space for efficient computation.

Mathematical Foundati...

### _compute_modulation_terms(self, field)
**Описание:** Compute BVP modulation terms.

Physical Meaning:
    Computes modulation terms representing the effects of the
    high-frequency carrier on the BVP field evolution.

Mathematical Foundation:
    Comp...

### setup_spectral_evolution_matrix(self)
**Описание:** Setup spectral evolution matrix.

Physical Meaning:
    Pre-computes the spectral evolution matrix for efficient
    linear evolution computation.

Mathematical Foundation:
    Computes the spectral r...

## ./bhlff/solvers/integrators/bvp_integration_schemes.py
Methods: 6

### __init__(self, domain, config)
**Описание:** Initialize BVP integration schemes.

Physical Meaning:
    Sets up the BVP integration schemes with domain and configuration
    for temporal integration.

Args:
    domain (Domain): Computational dom...

### rk4_step(self, field, dt, evolution_func)
**Описание:** Fourth-order Runge-Kutta step.

Physical Meaning:
    Implements fourth-order Runge-Kutta time integration for
    BVP-modulated evolution with high accuracy.

Mathematical Foundation:
    RK4 scheme:...

### euler_step(self, field, dt, evolution_func)
**Описание:** Forward Euler step.

Physical Meaning:
    Implements forward Euler time integration for BVP-modulated
    evolution with first-order accuracy.

Mathematical Foundation:
    Euler scheme:
    a_new = ...

### crank_nicolson_step(self, field, dt, evolution_func)
**Описание:** Crank-Nicolson step.

Physical Meaning:
    Implements Crank-Nicolson time integration for BVP-modulated
    evolution with second-order accuracy and improved stability.

Mathematical Foundation:
    ...

### adaptive_step(self, field, dt, evolution_func, tolerance) -> tuple
**Описание:** Adaptive time step with error estimation.

Physical Meaning:
    Implements adaptive time stepping for BVP-modulated evolution
    with automatic error control and step size adjustment.

Mathematical ...

### get_scheme_info(self, scheme_name)
**Описание:** Get information about integration scheme.

Physical Meaning:
    Returns information about the specified integration scheme
    including order of accuracy and stability properties.

Args:
    scheme_...

## ./bhlff/solvers/integrators/bvp_modulation_integrator_core.py
Methods: 9

### __init__(self, domain, config)
**Описание:** Initialize BVP-modulated integrator.

Physical Meaning:
    Sets up the BVP-modulated integrator with carrier frequency
    and modulation parameters for temporal evolution.

Args:
    domain (Domain)...

### _setup_bvp_parameters(self)
**Описание:** Setup BVP integrator parameters.

Physical Meaning:
    Initializes the BVP integrator parameters from configuration
    including carrier frequency and modulation properties.

### step(self, field, dt)
**Описание:** Perform one time integration step.

Physical Meaning:
    Advances the field configuration by one time step using the
    specified integration scheme for BVP-modulated evolution.

Mathematical Founda...

### get_integrator_type(self) -> str
**Описание:** Get the integrator type.

Physical Meaning:
    Returns the type of integrator being used.

Returns:
    str: Integrator type.

### get_carrier_frequency(self) -> float
**Описание:** Get the carrier frequency.

Physical Meaning:
    Returns the high-frequency carrier frequency used in
    BVP modulation.

Returns:
    float: Carrier frequency.

### get_modulation_strength(self) -> float
**Описание:** Get the modulation strength.

Physical Meaning:
    Returns the strength of BVP modulation.

Returns:
    float: Modulation strength.

### get_integration_scheme(self) -> str
**Описание:** Get the integration scheme.

Physical Meaning:
    Returns the time integration scheme being used.

Returns:
    str: Integration scheme name.

### get_scheme_info(self)
**Описание:** Get information about the current integration scheme.

Physical Meaning:
    Returns detailed information about the current integration
    scheme including order of accuracy and stability properties....

### __repr__(self) -> str
**Описание:** String representation of the BVP-modulated integrator.

## ./bhlff/solvers/integrators/time_integrator.py
Methods: 9

### __init__(self, domain, config, bvp_core)
**Описание:** Initialize time integrator with BVP framework integration.

Physical Meaning:
    Sets up the time integrator with computational domain and
    configuration parameters for temporal evolution,
    wit...

### step(self, field, dt)
**Описание:** Perform one time step.

Physical Meaning:
    Advances the phase field configuration by one time step,
    computing the temporal evolution of the field.

Mathematical Foundation:
    Solves ∂a/∂t = F...
**Декораторы:** abstractmethod

### get_integrator_type(self) -> str
**Описание:** Get the integrator type.

Physical Meaning:
    Returns the type of time integrator being used.

Returns:
    str: Integrator type.

Raises:
    NotImplementedError: Must be implemented by subclasses.
**Декораторы:** abstractmethod

### get_domain(self) -> Domain
**Описание:** Get the computational domain.

Physical Meaning:
    Returns the computational domain for the integrator.

Returns:
    Domain: Computational domain.

### get_config(self)
**Описание:** Get the integrator configuration.

Physical Meaning:
    Returns the configuration parameters for the integrator.

Returns:
    Dict[str, Any]: Integrator configuration.

### detect_quenches(self, envelope)
**Описание:** Detect quench events in BVP envelope.

Physical Meaning:
    Detects quench events when local thresholds are reached
    in the BVP envelope using the integrated quench detection system.

Args:
    en...

### get_bvp_core(self)
**Описание:** Get the integrated BVP core.

Physical Meaning:
    Returns the BVP framework integration if available.

Returns:
    Optional[BVPCore]: BVP core integration or None.

### set_bvp_core(self, bvp_core)
**Описание:** Set the BVP core integration.

Physical Meaning:
    Updates the BVP framework integration and reinitializes
    the quench detection system.

Args:
    bvp_core (BVPCore): BVP framework integration.

### __repr__(self) -> str
**Описание:** String representation of the integrator.

## ./bhlff/testing/automated_reporting.py
Methods: 55

### set_physics_summary(self, summary)
**Описание:** Set physics validation summary.

### add_level_analysis(self, level, analysis)
**Описание:** Add level-specific analysis.

### set_quality_summary(self, summary)
**Описание:** Set quality metrics summary.

### set_performance_summary(self, summary)
**Описание:** Set performance metrics summary.

### set_validation_status(self, status)
**Описание:** Set validation status.

### set_physics_trends(self, trends)
**Описание:** Set physics trend analysis.

### set_convergence_analysis(self, analysis)
**Описание:** Set convergence analysis.

### set_quality_evolution(self, evolution)
**Описание:** Set quality evolution analysis.

### set_performance_trends(self, trends)
**Описание:** Set performance trend analysis.

### set_recommendations(self, recommendations)
**Описание:** Set recommendations.

### set_physics_validation(self, validation)
**Описание:** Set physics validation results.

### set_prediction_comparison(self, comparison)
**Описание:** Set theoretical prediction comparison.

### set_long_term_trends(self, trends)
**Описание:** Set long-term trend analysis.

### set_progress_assessment(self, assessment)
**Описание:** Set research progress assessment.

### set_future_recommendations(self, recommendations)
**Описание:** Set future recommendations.

### __init__(self, report_config, physics_interpreter)
**Описание:** Initialize automated reporting system.

Physical Meaning:
    Sets up reporting framework with physics interpretation
    capabilities for 7D theory validation results.

Args:
    report_config (Dict[...

### summarize_daily_physics(self, test_results)
**Описание:** Summarize daily physics validation results.

Physical Meaning:
    Creates daily summary of experimental validation progress,
    highlighting key physical principles tested and any
    deviations fro...

### analyze_weekly_trends(self, weekly_results)
**Описание:** Analyze weekly physics trends.

Physical Meaning:
    Analyzes weekly trends in physical validation,
    identifying patterns and progress toward
    theoretical predictions.

Args:
    weekly_results...

### comprehensive_validation(self, monthly_results)
**Описание:** Comprehensive physics validation for monthly report.

Physical Meaning:
    Provides comprehensive validation of 7D theory principles
    over monthly period, including detailed analysis of
    physic...

### _analyze_level_physics(self, level, level_results)
**Описание:** Analyze physics validation for specific level.

### _analyze_principle_trend(self, principle, trend_data)
**Описание:** Analyze trend for specific physical principle.

### _validate_principle_comprehensive(self, principle, data)
**Описание:** Comprehensive validation of physical principle.

### _calculate_theoretical_agreement(self, test_results) -> float
**Описание:** Calculate theoretical agreement score.

### _calculate_comprehensive_agreement(self, monthly_results)
**Описание:** Calculate comprehensive theoretical agreement.

### _generate_level_insights(self, level, level_results)
**Описание:** Generate insights for specific level.

### _calculate_level_physics_score(self, level_results) -> float
**Описание:** Calculate physics score for level.

### _generate_physics_insights(self, monthly_results)
**Описание:** Generate comprehensive physics insights.

### render_daily_report(self, report, role) -> str
**Описание:** Render daily report for specific role.

Physical Meaning:
    Generates role-appropriate daily report with physics
    interpretation and technical details as needed.

Args:
    report (DailyReport): ...

### render_weekly_report(self, report, role) -> str
**Описание:** Render weekly report for specific role.

### render_monthly_report(self, report, role) -> str
**Описание:** Render monthly report for specific role.

### _get_physics_context(self)
**Описание:** Get physics context for templates.

### aggregate_daily_data(self, test_results)
**Описание:** Aggregate daily test data.

Physical Meaning:
    Aggregates daily test results with physics validation
    metrics for comprehensive daily reporting.

Args:
    test_results (TestResults): Daily test...

### aggregate_weekly_data(self, daily_results)
**Описание:** Aggregate weekly data from daily results.

Physical Meaning:
    Aggregates weekly trends in physics validation
    and quality metrics for trend analysis.

Args:
    daily_results (List[TestResults])...

### aggregate_monthly_data(self, weekly_results)
**Описание:** Aggregate monthly data from weekly results.

Physical Meaning:
    Aggregates monthly trends for comprehensive
    physics validation assessment.

Args:
    weekly_results (List[Dict[str, Any]]): List...

### _aggregate_physics_metrics(self, test_results)
**Описание:** Aggregate physics validation metrics.

### _analyze_physics_trends(self, daily_results)
**Описание:** Analyze physics trends over time.

### _analyze_quality_evolution(self, weekly_results)
**Описание:** Analyze quality evolution.

### _analyze_comprehensive_validation(self, weekly_results)
**Описание:** Analyze comprehensive validation over monthly period.

### _analyze_long_term_trends(self, monthly_results)
**Описание:** Analyze long-term trends.

### send_report(self, email, report_content, role) -> bool
**Описание:** Send report to specific email address.

Physical Meaning:
    Distributes report with appropriate physics context
    to specified recipient.

Args:
    email (str): Recipient email address.
    repor...

### distribute_reports(self, reports, recipients)
**Описание:** Distribute reports with role-based customization.

Physical Meaning:
    Distributes reports to appropriate stakeholders with
    customized content based on their role in the research
    process (ph...

### _customize_report_for_role(self, report, role) -> str
**Описание:** Customize report content for specific role.

### generate_daily_report(self, test_results) -> DailyReport
**Описание:** Generate daily report with physics validation summary.

Physical Meaning:
    Creates daily summary of experimental validation progress,
    highlighting key physical principles tested and any
    dev...

### generate_weekly_report(self, weekly_results) -> WeeklyReport
**Описание:** Generate weekly report with trend analysis and physics insights.

Physical Meaning:
    Provides weekly analysis of experimental trends, identifying
    patterns in physical validation and progress to...

### generate_monthly_report(self, monthly_results) -> MonthlyReport
**Описание:** Generate monthly report with comprehensive physics validation.

Physical Meaning:
    Creates comprehensive monthly assessment of 7D theory
    validation progress, including detailed analysis of
    ...

### _analyze_level_results(self, level, level_results)
**Описание:** Analyze results for specific level.

### _generate_quality_summary(self, test_results)
**Описание:** Generate quality metrics summary.

### _generate_performance_summary(self, test_results)
**Описание:** Generate performance metrics summary.

### _assess_validation_status(self, test_results)
**Описание:** Assess overall validation status.

### _analyze_convergence_trends(self, weekly_results)
**Описание:** Analyze convergence trends.

### _analyze_performance_trends(self, weekly_results)
**Описание:** Analyze performance trends.

### _generate_recommendations(self, weekly_results)
**Описание:** Generate recommendations based on weekly analysis.

### _compare_with_theoretical_predictions(self, monthly_results)
**Описание:** Compare results with theoretical predictions.

### _assess_research_progress(self, monthly_results)
**Описание:** Assess research progress.

### _generate_future_recommendations(self, monthly_results)
**Описание:** Generate future recommendations.

## ./bhlff/testing/automated_testing.py
Methods: 32

### __post_init__(self)
**Описание:** Calculate execution time if test completed.

### add_test_result(self, result)
**Описание:** Add test result to level results.

### has_critical_physics_failures(self) -> bool
**Описание:** Check if level has critical physics validation failures.

### get_success_rate(self) -> float
**Описание:** Calculate success rate for level.

### add_level_results(self, level, results)
**Описание:** Add level results to overall results.

### calculate_overall_metrics(self)
**Описание:** Calculate overall metrics from level results.

### __init__(self)
**Описание:** Initialize results database.

### validate_result(self, test_result)
**Описание:** Validate test result against physics constraints.

Physical Meaning:
    Validates test results against fundamental physical principles
    of 7D phase field theory, ensuring conservation laws and
   ...

### _validate_energy_conservation(self, test_result)
**Описание:** Validate energy conservation in test result.

### _validate_virial_conditions(self, test_result)
**Описание:** Validate virial conditions in test result.

### _validate_topological_charge(self, test_result)
**Описание:** Validate topological charge conservation in test result.

### _validate_passivity_conditions(self, test_result)
**Описание:** Validate passivity conditions in test result.

### _calculate_compliance_score(self, validation_result) -> float
**Описание:** Calculate overall compliance score.

### add_test(self, test_id, level, priority, dependencies, physics_checks)
**Описание:** Add test to scheduler.

Physical Meaning:
    Adds test to scheduler with physics-aware prioritization
    and dependency management.

Args:
    test_id (str): Unique test identifier.
    level (str):...

### get_execution_order(self)
**Описание:** Get test execution order with physics prioritization.

Physical Meaning:
    Returns ordered list of test IDs with physics-first
    prioritization and dependency resolution.

Returns:
    List[str]: ...

### _parse_memory_limit(self, memory_limit) -> int
**Описание:** Parse memory limit string to bytes.

### get_execution_context(self)
**Описание:** Get execution context for resource management.

### __enter__(self)
**Описание:** Enter resource context.

### __exit__(self, exc_type, exc_val, exc_tb)
**Описание:** Exit resource context.

### _load_config(self, config_path)
**Описание:** Load testing configuration.

### _get_default_config(self)
**Описание:** Get default testing configuration.

### _setup_test_scheduling(self)
**Описание:** Setup test scheduling based on configuration.

### run_all_tests(self, levels, priority) -> TestResults
**Описание:** Run all tests with physics-first prioritization.

Physical Meaning:
    Executes comprehensive testing ensuring physical principles
    are validated before numerical accuracy tests.

Args:
    levels...

### run_level_tests(self, level) -> LevelTestResults
**Описание:** Run tests for specific experimental level.

Physical Meaning:
    Executes level-specific tests ensuring validation of
    corresponding physical phenomena and mathematical models.

Args:
    level (s...

### _prioritize_physics_tests(self, levels)
**Описание:** Prioritize tests based on physics importance.

### _build_test_suite(self, level, level_config)
**Описание:** Build test suite for specific level.

### _execute_test_suite(self, test_suite, context)
**Описание:** Execute test suite with parallel processing.

### _run_single_test(self, test_spec) -> TestResult
**Описание:** Run single test with physics validation.

### _validate_level_physics(self, level, level_results)
**Описание:** Validate physics for specific level.

### _handle_critical_failure(self, level, level_results)
**Описание:** Handle critical physics failure.

### store_result(self, result)
**Описание:** Store test result in database.

### get_results(self, level)
**Описание:** Get test results, optionally filtered by level.

## ./bhlff/testing/quality_monitor.py
Methods: 32

### add_physics_degradation(self, degradation)
**Описание:** Add physics degradation analysis.

### add_numerical_degradation(self, degradation)
**Описание:** Add numerical degradation analysis.

### add_spectral_degradation(self, degradation)
**Описание:** Add spectral degradation analysis.

### add_convergence_degradation(self, degradation)
**Описание:** Add convergence degradation analysis.

### set_overall_severity(self, severity)
**Описание:** Set overall degradation severity.

### __init__(self, baseline_metrics, physics_constraints)
**Описание:** Initialize quality monitor with physics-aware baselines.

Physical Meaning:
    Sets up monitoring with baseline values derived from
    theoretical predictions and validated experimental results.

Ar...

### validate_metrics(self, metrics) -> bool
**Описание:** Validate metrics against physics constraints.

Physical Meaning:
    Validates experimental metrics against fundamental
    physical principles of 7D phase field theory.

Args:
    metrics (Dict[str, ...

### add_metrics(self, metrics)
**Описание:** Add metrics to history.

Physical Meaning:
    Records quality metrics for historical analysis
    and trend detection.

Args:
    metrics (QualityMetrics): Quality metrics to record.

### get_recent_metrics(self, days)
**Описание:** Get recent metrics within specified time window.

Physical Meaning:
    Retrieves recent quality metrics for trend analysis
    and degradation detection.

Args:
    days (int): Number of days to look...

### get_trend_data(self, metric_name, days)
**Описание:** Get trend data for specific metric.

Physical Meaning:
    Extracts historical values for specific metric to
    analyze trends and detect degradation.

Args:
    metric_name (str): Name of metric to ...

### analyze_trends(self, historical_metrics)
**Описание:** Analyze trends in historical metrics.

Physical Meaning:
    Analyzes trends in physical and numerical metrics
    to detect degradation patterns that could indicate
    quality issues in 7D phase fie...

### _calculate_trend_score(self, values) -> float
**Описание:** Calculate trend score for metric values.

Physical Meaning:
    Calculates trend score indicating whether metric
    is improving (positive), degrading (negative), or
    stable (near zero).

Args:
  ...

### generate_alerts(self, degradation_report)
**Описание:** Generate alerts for quality degradation.

Physical Meaning:
    Creates alerts for quality degradation with specific
    physical interpretation and recommended actions.

Args:
    degradation_report ...

### _generate_physics_alerts(self, physics_degradation)
**Описание:** Generate physics-specific alerts.

### _generate_numerical_alerts(self, numerical_degradation)
**Описание:** Generate numerical accuracy alerts.

### _generate_spectral_alerts(self, spectral_degradation)
**Описание:** Generate spectral quality alerts.

### _generate_convergence_alerts(self, convergence_degradation)
**Описание:** Generate convergence quality alerts.

### check_quality_metrics(self, test_results) -> QualityMetrics
**Описание:** Check quality metrics against physics constraints.

Physical Meaning:
    Validates experimental results against physical principles
    of 7D theory, checking energy conservation, topological
    inv...

### detect_quality_degradation(self, current_metrics, historical_metrics) -> DegradationReport
**Описание:** Detect quality degradation with physics-aware analysis.

Physical Meaning:
    Identifies degradation in physical quantities that could
    indicate violations of conservation laws or theoretical prin...

### generate_quality_alerts(self, degraded_metrics)
**Описание:** Generate quality alerts with physics context.

Physical Meaning:
    Creates alerts for quality degradation with specific
    physical interpretation and recommended actions.

Args:
    degraded_metri...

### update_baseline_metrics(self, new_metrics)
**Описание:** Update baseline metrics with physics validation.

Physical Meaning:
    Updates baseline values only if they maintain physical
    validity and improve upon existing baselines.

Args:
    new_metrics ...

### _check_physics_metrics(self, test_results)
**Описание:** Check physics-based quality metrics.

### _check_numerical_metrics(self, test_results)
**Описание:** Check numerical quality metrics.

### _check_spectral_metrics(self, test_results)
**Описание:** Check spectral quality metrics.

### _compute_overall_quality_score(self, quality_metrics) -> float
**Описание:** Compute overall quality score.

### _determine_quality_status(self, overall_score) -> QualityStatus
**Описание:** Determine quality status from overall score.

### _detect_physics_degradation(self, current_metrics, historical_metrics)
**Описание:** Detect physics-specific degradation.

### _detect_numerical_degradation(self, current_metrics, historical_metrics)
**Описание:** Detect numerical accuracy degradation.

### _detect_spectral_degradation(self, current_metrics, historical_metrics)
**Описание:** Detect spectral quality degradation.

### _detect_convergence_degradation(self, current_metrics, historical_metrics)
**Описание:** Detect convergence quality degradation.

### _assess_degradation_severity(self, report) -> AlertSeverity
**Описание:** Assess overall degradation severity.

### _is_significant_improvement(self, new_metrics) -> bool
**Описание:** Check if new metrics represent significant improvement.

## ./bhlff/utils/cuda_utils.py
Methods: 15

### detect_cuda_availability() -> bool
**Описание:** Detect if CUDA is available and working.

Physical Meaning:
    Checks if CUDA is properly installed and functional
    for 7D phase field calculations.

Returns:
    bool: True if CUDA is available a...

### get_optimal_backend()
**Описание:** Get the optimal backend for computations.

Physical Meaning:
    Automatically selects the best available backend
    (CUDA GPU or CPU) for 7D phase field calculations
    based on availability and pe...

### get_backend_info() -> dict
**Описание:** Get information about the current backend.

Physical Meaning:
    Provides detailed information about the computational
    backend being used for 7D phase field calculations.

Returns:
    dict: Back...

### get_global_backend()
**Описание:** Get the global backend instance.

Physical Meaning:
    Returns the global backend instance for use throughout
    the BHLFF framework, ensuring consistent GPU/CPU usage.

Returns:
    Union[CUDABacke...

### reset_global_backend()
**Описание:** Reset the global backend instance.

Physical Meaning:
    Resets the global backend to allow re-detection
    of optimal backend configuration.

### __init__(self)
**Описание:** Initialize CPU backend.

### zeros(self, shape, dtype)
**Описание:** Create zero array on CPU.

### ones(self, shape, dtype)
**Описание:** Create ones array on CPU.

### array(self, array)
**Описание:** Return array as-is (already on CPU).

### to_numpy(self, array)
**Описание:** Return array as-is (already numpy).

### fft(self, array, axes)
**Описание:** Perform FFT on CPU.

### ifft(self, array, axes)
**Описание:** Perform inverse FFT on CPU.

### fftshift(self, array, axes)
**Описание:** Perform FFT shift on CPU.

### ifftshift(self, array, axes)
**Описание:** Perform inverse FFT shift on CPU.

### get_memory_info(self) -> dict
**Описание:** Get CPU memory information.

## ./code_mapper.py
Methods: 19

### main()
**Описание:** Main function.

### __init__(self, root_dir)
**Описание:** Initialize code mapper.

### scan_directory(self, directory)
**Описание:** Scan directory for Python files.

### analyze_file(self, file_path)
**Описание:** Analyze single Python file.

### extract_class_info(self, node, file_path, import_lines)
**Описание:** Extract class information from AST node.

### extract_function_info(self, node, file_path, import_lines)
**Описание:** Extract function information from AST node.

### _check_method_issues(self, node, file_path, class_name)
**Описание:** Check for common issues in methods and functions.

### _has_only_pass(self, node) -> bool
**Описание:** Check if function has only pass statement.

### _has_not_implemented(self, node) -> bool
**Описание:** Check if function contains NotImplemented.

### _is_abstract_method(self, node) -> bool
**Описание:** Check if method is abstract.

### extract_import_info(self, node)
**Описание:** Extract import information from AST node.

### generate_report(self) -> str
**Описание:** Generate code map report.

### generate_issues_report(self) -> str
**Описание:** Generate issues report.

### save_report(self)
**Описание:** Save code map report to file.

### save_issues_report(self)
**Описание:** Save issues report to file.

### generate_method_index(self) -> str
**Описание:** Generate method index report.

### save_method_index(self)
**Описание:** Save method index report to file.

### generate_yaml_method_index(self) -> str
**Описание:** Generate YAML method index.

### save_yaml_method_index(self)
**Описание:** Save YAML method index to file.

## ./docs/conf.py
Methods: 1

### setup(app)
**Описание:** Setup function for Sphinx.

## ./examples/bvp_7d_example.py
Methods: 5

### load_config(config_path) -> dict
**Описание:** Load configuration from JSON file.

### create_7d_domain(config) -> Domain7D
**Описание:** Create 7D space-time domain from configuration.

### create_3d_domain(config) -> Domain
**Описание:** Create 3D domain for compatibility.

### create_source_7d(domain_7d)
**Описание:** Create 7D source term for BVP equation.

Physical Meaning:
    Creates a source term s(x,φ,t) that represents external excitations
    or initial conditions in 7D space-time.

### run_bvp_7d_example()
**Описание:** Run complete 7D BVP example.

## ./examples/level_b_example.py
Methods: 2

### run_level_b_analysis_example()
**Описание:** Run comprehensive Level B analysis example.

Physical Meaning:
    Demonstrates the complete Level B analysis workflow,
    showing how to validate fundamental properties of the
    phase field in hom...

### run_parameter_variation_example()
**Описание:** Run parameter variation analysis example.

Physical Meaning:
    Demonstrates how to analyze the sensitivity of Level B
    properties to different parameters, validating the
    theoretical predictio...

## ./examples/level_c_example.py
Methods: 6

### example_abcd_model()
**Описание:** Example of ABCD model usage.

Physical Meaning:
    Demonstrates how to use the ABCD model for analyzing
    resonator chains and finding system resonance modes.

### example_memory_parameters()
**Описание:** Example of memory parameters usage.

Physical Meaning:
    Demonstrates how to create and use memory parameters
    for quench memory analysis.

### example_dual_mode_source()
**Описание:** Example of dual-mode source usage.

Physical Meaning:
    Demonstrates how to create dual-mode sources for
    mode beating analysis.

### example_beating_pattern()
**Описание:** Example of beating pattern usage.

Physical Meaning:
    Demonstrates how to create and analyze beating patterns
    for mode beating analysis.

### example_mathematical_operations()
**Описание:** Example of mathematical operations.

Physical Meaning:
    Demonstrates basic mathematical operations used in
    Level C analysis.

### main()
**Описание:** Main function to run all examples.

Physical Meaning:
    Runs all Level C analysis examples to demonstrate
    the capabilities of the framework.

## ./examples/level_d_example.py
Methods: 7

### create_test_field(domain)
**Описание:** Create test field for Level D analysis.

Physical Meaning:
    Creates a test field with multiple frequency components
    to demonstrate multimode superposition analysis.

Args:
    domain (Domain): ...

### run_mode_superposition_analysis(models, field)
**Описание:** Run mode superposition analysis (D1).

Physical Meaning:
    Tests the stability of the phase field frame when
    adding new modes, ensuring topological robustness.

Args:
    models (LevelDModels): ...

### run_field_projection_analysis(models, field)
**Описание:** Run field projection analysis (D2).

Physical Meaning:
    Separates the unified phase field into different
    interaction regimes based on frequency and amplitude
    characteristics.

Args:
    mod...

### run_streamline_analysis(models, field)
**Описание:** Run phase streamline analysis (D3).

Physical Meaning:
    Computes streamlines of the phase gradient field,
    revealing the topological structure of phase flow
    around defects and singularities....

### visualize_results(results, output_dir)
**Описание:** Visualize analysis results.

Physical Meaning:
    Creates visualizations of the analysis results to
    understand the field structure and dynamics.

Args:
    results (Dict): Analysis results
    ou...

### save_results(results, output_dir)
**Описание:** Save analysis results to files.

Physical Meaning:
    Saves the analysis results to JSON files for
    further analysis and documentation.

Args:
    results (Dict): Analysis results
    output_dir (...

### main()
**Описание:** Main function for Level D example.

## ./examples/level_f_example.py
Methods: 6

### create_example_system()
**Описание:** Create example multi-particle system.

Physical Meaning:
    Creates a system with two oppositely charged particles
    to study collective effects and interactions.

### study_collective_excitations(system)
**Описание:** Study collective excitations in the system.

Physical Meaning:
    Analyzes collective excitations by applying external
    fields and studying the system response.

### study_phase_transitions(system)
**Описание:** Study phase transitions in the system.

Physical Meaning:
    Analyzes phase transitions by varying system parameters
    and monitoring order parameters.

### study_nonlinear_effects(system)
**Описание:** Study nonlinear effects in the system.

Physical Meaning:
    Analyzes nonlinear effects by adding nonlinear
    interactions and studying soliton solutions.

### visualize_results(system, excitations, transitions, nonlinear)
**Описание:** Visualize results from Level F analysis.

Physical Meaning:
    Creates visualizations to show the results of
    collective effects analysis.

### main()
**Описание:** Main function to demonstrate Level F models.

Physical Meaning:
    Demonstrates the complete workflow for studying
    collective effects in multi-particle systems.

## ./examples/level_g_example.py
Methods: 8

### example_cosmological_evolution()
**Описание:** Example of cosmological evolution.

Physical Meaning:
    Demonstrates the evolution of phase field in expanding universe,
    including structure formation and cosmological parameters.

### example_astrophysical_objects()
**Описание:** Example of astrophysical objects.

Physical Meaning:
    Demonstrates the representation of astrophysical objects as
    phase field configurations with specific topological properties.

### example_gravitational_effects()
**Описание:** Example of gravitational effects.

Physical Meaning:
    Demonstrates the connection between phase field and gravity,
    including spacetime curvature and gravitational waves.

### example_large_scale_structure()
**Описание:** Example of large-scale structure formation.

Physical Meaning:
    Demonstrates the formation of large-scale structure in the
    universe through phase field evolution and gravitational effects.

### example_particle_inversion()
**Описание:** Example of particle parameter inversion.

Physical Meaning:
    Demonstrates the inversion of model parameters from
    observable particle properties.

### example_visualization()
**Описание:** Example of visualization.

Physical Meaning:
    Demonstrates visualization of cosmological and astrophysical
    results from the 7D phase field theory.

### main()
**Описание:** Main example function.

Physical Meaning:
    Demonstrates the complete workflow of level G models for
    cosmological and astrophysical applications.

### __init__(self)
**Описание:** Нет докстринга

## ./setup.py
Methods: 4

### read_readme()
**Описание:** Read the README file for long description.

### get_version()
**Описание:** Get version from bhlff/__init__.py.

### get_package_data()
**Описание:** Get package data files.

### get_entry_points()
**Описание:** Get console script entry points.

## ./tests/conftest.py
Methods: 18

### test_domain_7d() -> Domain
**Описание:** Create 7D domain for testing.

Physical Meaning:
    Provides a 7D computational domain M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ
    for testing the BVP theory implementation.

Mathematical Foundation:
    Domain with pe...
**Декораторы:** <ast.Call object at 0x753e6fcce010>

### test_domain_7d_high_res() -> Domain
**Описание:** Create high-resolution 7D domain for testing.

Physical Meaning:
    Provides a high-resolution 7D computational domain for
    testing spectral accuracy and convergence.
**Декораторы:** <ast.Call object at 0x753e6fccefd0>

### test_bvp_constants() -> BVPConstantsAdvanced
**Описание:** Create BVP constants for testing.

Physical Meaning:
    Provides physically meaningful BVP constants for testing
    the 7D BVP theory implementation.

Mathematical Foundation:
    Constants satisfyi...
**Декораторы:** <ast.Call object at 0x753e6fccf210>

### test_bvp_constants_extreme() -> BVPConstantsAdvanced
**Описание:** Create extreme BVP constants for testing.

Physical Meaning:
    Provides extreme but physically valid BVP constants for
    testing robustness and edge cases.
**Декораторы:** <ast.Call object at 0x753e6fccdb50>

### test_source_gaussian(domain_7d)
**Описание:** Create Gaussian source for testing.

Physical Meaning:
    Provides a Gaussian source with known analytical properties
    for testing BVP equation solution.
**Декораторы:** <ast.Attribute object at 0x753e6fb2d150>

### test_source_sinusoidal(domain_7d)
**Описание:** Create sinusoidal source for testing.

Physical Meaning:
    Provides a sinusoidal source with known analytical derivatives
    for testing spectral methods.
**Декораторы:** <ast.Attribute object at 0x753e6fbc0350>

### test_source_localized(domain_7d)
**Описание:** Create localized source for testing.

Physical Meaning:
    Provides a localized source for testing boundary conditions
    and field propagation.
**Декораторы:** <ast.Attribute object at 0x753e6fbbad10>

### test_envelope_analytical(domain_7d)
**Описание:** Create analytical envelope for testing.

Physical Meaning:
    Provides an envelope with known analytical properties
    for testing postulate validation.
**Декораторы:** <ast.Attribute object at 0x753e6fce2090>

### test_frequencies()
**Описание:** Create test frequencies for frequency-dependent testing.

Physical Meaning:
    Provides a range of frequencies for testing frequency-dependent
    material properties and spectral methods.
**Декораторы:** <ast.Attribute object at 0x753e6fce0750>

### test_amplitudes()
**Описание:** Create test amplitudes for nonlinear testing.

Physical Meaning:
    Provides a range of amplitudes for testing nonlinear
    material properties and effects.
**Декораторы:** <ast.Attribute object at 0x753e6fce0990>

### test_scales()
**Описание:** Create test scales for renormalization testing.

Physical Meaning:
    Provides a range of scales for testing renormalization
    group flow and scaling behavior.
**Декораторы:** <ast.Attribute object at 0x753e6fcebe50>

### setup_test_environment()
**Описание:** Setup test environment.

Physical Meaning:
    Configures the test environment for reproducible
    physical validation testing.
**Декораторы:** <ast.Call object at 0x753e6fceab50>

### cleanup_cuda_memory()
**Описание:** Cleanup CUDA memory after each test.

Physical Meaning:
    Ensures CUDA memory is properly freed after each test
    to prevent memory accumulation and out-of-memory errors.
**Декораторы:** <ast.Call object at 0x753e6fcfdd50>

### pytest_configure(config)
**Описание:** Configure pytest for physical validation testing.

Physical Meaning:
    Configures pytest with appropriate settings for
    physical validation testing of the 7D BVP theory.

### pytest_collection_modifyitems(config, items)
**Описание:** Modify test collection for physical validation.

Physical Meaning:
    Modifies test collection to prioritize physical
    validation tests and handle slow tests appropriately.

### test_results_cache()
**Описание:** Create test results cache.

Physical Meaning:
    Provides a cache for storing test results to avoid
    recomputation in long-running physical validation tests.
**Декораторы:** <ast.Call object at 0x753e6fcd90d0>

### pytest_runtest_setup(item)
**Описание:** Setup for each test run.

Physical Meaning:
    Sets up each test run with appropriate configuration
    for physical validation testing.

### pytest_addoption(parser)
**Описание:** Add command line options for physical validation testing.

Physical Meaning:
    Adds command line options for controlling physical
    validation test execution.

## ./tests/integration/test_automated_testing_integration.py
Methods: 13

### setup_method(self)
**Описание:** Setup integration test fixtures.

### teardown_method(self)
**Описание:** Cleanup integration test fixtures.

### _create_integration_config(self)
**Описание:** Create integration test configuration.

### _create_reporting_config(self)
**Описание:** Create reporting configuration.

### _load_reporting_config(self)
**Описание:** Load reporting configuration.

### test_complete_automated_testing_workflow(self)
**Описание:** Test complete automated testing workflow.

Physical Meaning:
    Verifies that the complete automated testing workflow
    correctly executes with physics validation, quality
    monitoring, and autom...

### test_physics_validation_integration(self)
**Описание:** Test physics validation integration.

Physical Meaning:
    Verifies that physics validation correctly
    identifies violations of 7D theory principles
    and maintains conservation laws.

### test_quality_monitoring_integration(self)
**Описание:** Test quality monitoring integration.

Physical Meaning:
    Verifies that quality monitoring correctly
    tracks physics validation metrics and detects
    degradation patterns.

### test_automated_reporting_integration(self)
**Описание:** Test automated reporting integration.

Physical Meaning:
    Verifies that automated reporting correctly
    generates physics-aware reports with appropriate
    interpretation and context.

### test_physics_interpreter_integration(self)
**Описание:** Test physics interpreter integration.

Physical Meaning:
    Verifies that physics interpreter correctly
    provides physical interpretation of experimental
    results in the context of 7D theory.

### test_error_handling_integration(self)
**Описание:** Test error handling integration.

Physical Meaning:
    Verifies that the system correctly handles
    errors and maintains physics validation
    integrity during failures.

### _create_invalid_config(self)
**Описание:** Create invalid configuration for error testing.

### test_performance_integration(self)
**Описание:** Test performance integration.

Physical Meaning:
    Verifies that the system maintains good
    performance while ensuring physics validation
    accuracy for 7D computations.

## ./tests/integration/test_bvp_core_pipeline_physics.py
Methods: 12

### domain_7d(self)
**Описание:** Create 7D domain for complete pipeline testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa17150>

### bvp_constants(self)
**Описание:** Create BVP constants for complete pipeline testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa15e90>

### bvp_core(self, domain_7d, bvp_constants)
**Описание:** Create BVP core for complete pipeline testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa16150>

### test_complete_bvp_pipeline_physics(self, domain_7d, bvp_core)
**Описание:** Test complete BVP pipeline physics.

Physical Meaning:
    Validates the complete BVP pipeline from source to solution,
    ensuring physical consistency and theoretical correctness
    throughout the...

### test_bvp_energy_conservation_pipeline(self, domain_7d, bvp_core)
**Описание:** Test energy conservation throughout BVP pipeline.

Physical Meaning:
    Validates that energy is conserved throughout the entire
    BVP pipeline, ensuring fundamental conservation laws.

Mathematica...

### _generate_physical_source(self, domain)
**Описание:** Generate a physical source for testing.

### _generate_time_evolving_source(self, domain)
**Описание:** Generate time-evolving source for energy conservation test.

### _compute_physical_quantities(self, envelope, domain)
**Описание:** Compute physical quantities from envelope.

### _validate_energy_conservation(self, envelope, source, domain)
**Описание:** Validate energy conservation.

### _compute_total_energy(self, envelope, domain) -> float
**Описание:** Compute total energy of the envelope.

### _compute_gradient_magnitude(self, envelope, domain)
**Описание:** Compute gradient magnitude.

### _compute_phase_coherence(self, envelope, domain) -> float
**Описание:** Compute phase coherence.

## ./tests/integration/test_bvp_impedance_calculation_physics.py
Methods: 5

### domain_7d(self)
**Описание:** Create 7D domain for complete pipeline testing.
**Декораторы:** <ast.Attribute object at 0x753e6fcfcc50>

### bvp_constants(self)
**Описание:** Create BVP constants for complete pipeline testing.
**Декораторы:** <ast.Attribute object at 0x753e6fcfca10>

### bvp_core(self, domain_7d, bvp_constants)
**Описание:** Create BVP core for complete pipeline testing.
**Декораторы:** <ast.Attribute object at 0x753e6fcfc110>

### test_bvp_impedance_calculation_physics(self, domain_7d, bvp_core)
**Описание:** Test BVP impedance calculation physics.

Physical Meaning:
    Validates that impedance calculation correctly computes
    the field impedance and maintains physical consistency.

Mathematical Foundat...

### _generate_physical_source(self, domain)
**Описание:** Generate a physical source for testing.

## ./tests/integration/test_bvp_interface_pipeline_physics.py
Methods: 5

### domain_7d(self)
**Описание:** Create 7D domain for complete pipeline testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb1b290>

### bvp_constants(self)
**Описание:** Create BVP constants for complete pipeline testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb19e50>

### bvp_interface(self, domain_7d, bvp_constants)
**Описание:** Create BVP interface for complete pipeline testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb18dd0>

### test_bvp_interface_physics(self, domain_7d, bvp_interface)
**Описание:** Test BVP interface physics.

Physical Meaning:
    Validates that the BVP interface correctly coordinates
    all BVP components and maintains physical consistency.

Mathematical Foundation:
    Tests...

### _generate_physical_source(self, domain)
**Описание:** Generate a physical source for testing.

## ./tests/integration/test_bvp_phase_vector_physics.py
Methods: 5

### domain_7d(self)
**Описание:** Create 7D domain for complete pipeline testing.
**Декораторы:** <ast.Attribute object at 0x753e6fbc72d0>

### bvp_constants(self)
**Описание:** Create BVP constants for complete pipeline testing.
**Декораторы:** <ast.Attribute object at 0x753e6fcfe8d0>

### bvp_core(self, domain_7d, bvp_constants)
**Описание:** Create BVP core for complete pipeline testing.
**Декораторы:** <ast.Attribute object at 0x753e6fcff390>

### test_bvp_phase_vector_physics(self, domain_7d, bvp_core)
**Описание:** Test BVP phase vector physics.

Physical Meaning:
    Validates that phase vector correctly implements
    U(1)³ phase structure and maintains physical consistency.

Mathematical Foundation:
    Tests...

### _generate_physical_source(self, domain)
**Описание:** Generate a physical source for testing.

## ./tests/integration/test_bvp_quench_dynamics_physics.py
Methods: 6

### domain_7d(self)
**Описание:** Create 7D domain for complete pipeline testing.
**Декораторы:** <ast.Attribute object at 0x753e6fbb8ad0>

### bvp_constants(self)
**Описание:** Create BVP constants for complete pipeline testing.
**Декораторы:** <ast.Attribute object at 0x753e6fbb90d0>

### bvp_core(self, domain_7d, bvp_constants)
**Описание:** Create BVP core for complete pipeline testing.
**Декораторы:** <ast.Attribute object at 0x753e6fbbbd90>

### test_bvp_quench_dynamics_physics(self, domain_7d, bvp_core)
**Описание:** Test BVP quench dynamics physics.

Physical Meaning:
    Validates that quench detection correctly identifies
    phase transition regions and maintains physical consistency.

Mathematical Foundation:...

### _generate_source_with_quenches(self, domain)
**Описание:** Generate source with known quench regions.

### _compute_gradient_magnitude(self, envelope, domain)
**Описание:** Compute gradient magnitude.

## ./tests/test_bvp_framework.py
Methods: 18

### domain_7d(self)
**Описание:** Create 7D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fbc7450>

### bvp_config(self)
**Описание:** Create BVP configuration for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fbc6250>

### bvp_core(self, domain_7d, bvp_config)
**Описание:** Create BVP core instance for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fbc5b50>

### test_domain_7d_structure(self, domain_7d)
**Описание:** Test 7D domain structure.

### test_bvp_core_initialization(self, bvp_core)
**Описание:** Test BVP core initialization.

### test_phase_vector_u1_structure(self, bvp_core)
**Описание:** Test U(1)³ phase structure.

### test_envelope_solver_7d(self, bvp_core)
**Описание:** Test 7D envelope solver.

### test_quench_detector(self, bvp_core)
**Описание:** Test quench detection.

### test_impedance_calculator(self, bvp_core)
**Описание:** Test impedance calculation.

### test_bvp_postulates(self, bvp_core)
**Описание:** Test all 9 BVP postulates.

### test_bvp_interface(self, bvp_core)
**Описание:** Test BVP interface.

### test_7d_gradient_computation(self, bvp_core)
**Описание:** Test 7D gradient computation.

### test_phase_vector_electroweak_currents(self, bvp_core)
**Описание:** Test electroweak current computation.

### test_bvp_constants(self, bvp_core)
**Описание:** Test BVP constants.

### test_bvp_core_solve_envelope(self, bvp_core)
**Описание:** Test BVP core envelope solving.

### test_bvp_core_detect_quenches(self, bvp_core)
**Описание:** Test BVP core quench detection.

### test_bvp_core_compute_impedance(self, bvp_core)
**Описание:** Test BVP core impedance computation.

### test_bvp_framework_validation(self, bvp_core)
**Описание:** Test BVP framework validation.

## ./tests/test_bvp_level_a_integration.py
Methods: 5

### domain(self)
**Описание:** Create test domain.
**Декораторы:** <ast.Attribute object at 0x753e6fb202d0>

### bvp_config(self)
**Описание:** Create BVP configuration.
**Декораторы:** <ast.Attribute object at 0x753e6fb23110>

### test_level_a_bvp_framework_validation(self, domain, bvp_config)
**Описание:** Test A0: BVP Framework Validation.

### test_level_a_bvp_enhanced_solvers(self, domain, bvp_config)
**Описание:** Test A1: BVP-Enhanced Solvers.

### test_level_a_bvp_scaling(self, domain, bvp_config)
**Описание:** Test A2: BVP Scaling and Nondimensionalization.

## ./tests/test_bvp_level_b_integration.py
Methods: 5

### domain(self)
**Описание:** Create test domain.
**Декораторы:** <ast.Attribute object at 0x753e6fce0110>

### bvp_config(self)
**Описание:** Create BVP configuration.
**Декораторы:** <ast.Attribute object at 0x753e6fce0950>

### test_level_b_bvp_power_law_tails(self, domain, bvp_config)
**Описание:** Test B1: BVP Power Law Tails.

### test_level_b_bvp_topological_charge(self, domain, bvp_config)
**Описание:** Test B2: BVP Topological Charge.

### test_level_b_bvp_zone_separation(self, domain, bvp_config)
**Описание:** Test B3: BVP Zone Separation.

## ./tests/test_bvp_level_c_integration.py
Methods: 5

### domain(self)
**Описание:** Create test domain.
**Декораторы:** <ast.Attribute object at 0x753e6fceafd0>

### bvp_config(self)
**Описание:** Create BVP configuration.
**Декораторы:** <ast.Attribute object at 0x753e6fcead10>

### test_level_c_bvp_boundary_effects(self, domain, bvp_config)
**Описание:** Test C1: BVP Boundary Effects.

### test_level_c_bvp_resonator_chains(self, domain, bvp_config)
**Описание:** Test C2: BVP Resonator Chains.

### test_level_c_bvp_quench_memory(self, domain, bvp_config)
**Описание:** Test C3: BVP Quench Memory.

## ./tests/test_bvp_level_d_integration.py
Methods: 5

### domain(self)
**Описание:** Create test domain.
**Декораторы:** <ast.Attribute object at 0x753e6fbc33d0>

### bvp_config(self)
**Описание:** Create BVP configuration.
**Декораторы:** <ast.Attribute object at 0x753e6fbc3c50>

### test_level_d_bvp_mode_superposition(self, domain, bvp_config)
**Описание:** Test D1: BVP Mode Superposition.

### test_level_d_bvp_field_projections(self, domain, bvp_config)
**Описание:** Test D2: BVP Field Projections.

### test_level_d_bvp_streamlines(self, domain, bvp_config)
**Описание:** Test D3: BVP Streamlines.

## ./tests/test_bvp_level_e_integration.py
Methods: 5

### domain(self)
**Описание:** Create test domain.
**Декораторы:** <ast.Attribute object at 0x753e6fb2e150>

### bvp_config(self)
**Описание:** Create BVP configuration.
**Декораторы:** <ast.Attribute object at 0x753e6fb2d790>

### test_level_e_bvp_solitons(self, domain, bvp_config)
**Описание:** Test E1: BVP Solitons.

### test_level_e_bvp_defect_dynamics(self, domain, bvp_config)
**Описание:** Test E2: BVP Defect Dynamics.

### test_level_e_bvp_theory_integration(self, domain, bvp_config)
**Описание:** Test E3: BVP Theory Integration.

## ./tests/test_bvp_level_f_integration.py
Methods: 5

### domain(self)
**Описание:** Create test domain.
**Декораторы:** <ast.Attribute object at 0x753e6fcd8a10>

### bvp_config(self)
**Описание:** Create BVP configuration.
**Декораторы:** <ast.Attribute object at 0x753e6fcdb750>

### test_level_f_bvp_multi_particle_systems(self, domain, bvp_config)
**Описание:** Test F1: BVP Multi-Particle Systems.

### test_level_f_bvp_collective_modes(self, domain, bvp_config)
**Описание:** Test F2: BVP Collective Modes.

### test_level_f_bvp_nonlinear_effects(self, domain, bvp_config)
**Описание:** Test F3: BVP Nonlinear Effects.

## ./tests/test_bvp_level_g_integration.py
Methods: 5

### domain(self)
**Описание:** Create test domain.
**Декораторы:** <ast.Attribute object at 0x753e6fbc7c90>

### bvp_config(self)
**Описание:** Create BVP configuration.
**Декораторы:** <ast.Attribute object at 0x753e6fbc7650>

### test_level_g_bvp_cosmological_evolution(self, domain, bvp_config)
**Описание:** Test G1: BVP Cosmological Evolution.

### test_level_g_bvp_astrophysical_objects(self, domain, bvp_config)
**Описание:** Test G2: BVP Astrophysical Objects.

### test_level_g_bvp_gravitational_effects(self, domain, bvp_config)
**Описание:** Test G3: BVP Gravitational Effects.

## ./tests/unit/test_7d_bvp_physics.py
Methods: 10

### domain_7d(self) -> Domain7DBVP
**Описание:** Create 7D BVP domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb37c10>

### parameters_7d(self) -> Parameters7DBVP
**Описание:** Create 7D BVP parameters for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb36810>

### solver_7d(self, domain_7d, parameters_7d) -> BVPEnvelopeSolver
**Описание:** Create 7D BVP solver for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa20550>

### test_7d_domain_structure(self, domain_7d)
**Описание:** Test 7D domain structure.

Physical Meaning:
    Verifies that the domain correctly represents the 7D space-time
    structure M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

### test_nonlinear_stiffness(self, parameters_7d)
**Описание:** Test nonlinear stiffness coefficient.

Physical Meaning:
    Verifies that κ(|a|) = κ₀ + κ₂|a|² correctly implements
    the nonlinear stiffness dependence on field amplitude.

### test_effective_susceptibility(self, parameters_7d)
**Описание:** Test effective susceptibility coefficient.

Physical Meaning:
    Verifies that χ(|a|) = χ' + iχ''(|a|) correctly implements
    the effective susceptibility with quench effects.

### test_linearized_solution_accuracy(self, solver_7d)
**Описание:** Test linearized solution accuracy.

Physical Meaning:
    Verifies that the linearized solution satisfies the linearized
    equation L_β a = μ(-Δ)^β a + λa = s with high accuracy.

### test_envelope_solution_accuracy(self, solver_7d)
**Описание:** Test envelope solution accuracy.

Physical Meaning:
    Verifies that the envelope solution satisfies the BVP equation
    ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s with high accuracy.

### test_nonlinear_coefficients(self, solver_7d)
**Описание:** Test nonlinear coefficients computation.

Physical Meaning:
    Verifies that nonlinear stiffness and susceptibility coefficients
    are correctly computed for given envelope amplitude.

### test_solution_validation(self, solver_7d)
**Описание:** Test solution validation methods.

Physical Meaning:
    Verifies that solution validation correctly checks whether
    a solution satisfies the BVP equation within specified tolerance.

## ./tests/unit/test_acceptance_criteria.py
Methods: 8

### domain_3d(self)
**Описание:** Create 3D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb07310>

### test_pass_1_memory_kernel_passivity(self, domain_3d)
**Описание:** Test PASS-1: ReY(ω)≥0 for memory kernels below resonances.

### test_gw_1_amplitude_law(self, domain_3d)
**Описание:** Test GW-1: |h|∝a^{-1} when Γ=K=0 through c_T=c_φ evolution.

### test_len_1_lensing_consistency(self, domain_3d)
**Описание:** Test LEN-1: lensing consistency using g_eff distance factors.

### test_frac_1_energy_monotonicity(self, domain_3d)
**Описание:** Test FRAC-1: G_β tail validation and energy monotonicity (ΔE≤0).

### test_mass_terms_forbidden_in_base_regime(self, domain_3d)
**Описание:** Test that mass terms are forbidden in base regime (tempered_lambda==0).

### test_stability_assertions(self, domain_3d)
**Описание:** Test stability assertions: c_φ^2>0, M_*^2>0.

### test_acceptance_criteria_integration(self, domain_3d)
**Описание:** Test integration of all acceptance criteria.

## ./tests/unit/test_automated_testing.py
Methods: 28

### setup_method(self)
**Описание:** Setup test fixtures.

### teardown_method(self)
**Описание:** Cleanup test fixtures.

### _create_test_config(self)
**Описание:** Create test configuration file.

### test_automated_testing_system_initialization(self)
**Описание:** Test automated testing system initialization.

Physical Meaning:
    Verifies that the testing system initializes correctly
    with physics validation and configuration.

### test_run_all_tests_physics_priority(self)
**Описание:** Test physics-first test execution.

Physical Meaning:
    Verifies that tests are executed with physics-first
    prioritization, ensuring fundamental physical
    principles are validated before nume...

### test_run_level_tests_physics_validation(self)
**Описание:** Test level-specific test execution with physics validation.

Physical Meaning:
    Verifies that level-specific tests include proper
    physics validation for 7D theory principles.

### test_physics_validator_energy_conservation(self)
**Описание:** Test physics validator for energy conservation.

Physical Meaning:
    Verifies that energy conservation validation
    correctly identifies violations of this fundamental
    physical principle.

### test_physics_validator_virial_conditions(self)
**Описание:** Test physics validator for virial conditions.

Physical Meaning:
    Verifies that virial condition validation
    correctly identifies violations of energy
    balance principles.

### test_physics_validator_topological_charge(self)
**Описание:** Test physics validator for topological charge conservation.

Physical Meaning:
    Verifies that topological charge validation
    correctly identifies violations of charge
    conservation principles...

### test_physics_validator_passivity(self)
**Описание:** Test physics validator for passivity conditions.

Physical Meaning:
    Verifies that passivity validation correctly
    identifies violations of physical realizability
    conditions.

### test_test_scheduler_physics_priority(self)
**Описание:** Test test scheduler with physics prioritization.

Physical Meaning:
    Verifies that test scheduler correctly prioritizes
    physics validation tests before numerical accuracy tests.

### test_resource_manager_initialization(self)
**Описание:** Test resource manager initialization.

Physical Meaning:
    Verifies that resource manager correctly sets up
    constraints for 7D phase field computations.

### test_level_test_results_aggregation(self)
**Описание:** Test level test results aggregation.

Physical Meaning:
    Verifies that level results correctly aggregate
    physics validation metrics and test outcomes.

### test_level_test_results_critical_failures(self)
**Описание:** Test detection of critical physics failures.

Physical Meaning:
    Verifies that critical physics violations are
    correctly identified and flagged.

### test_quality_monitor_initialization(self)
**Описание:** Test quality monitor initialization.

Physical Meaning:
    Verifies that quality monitor initializes with
    appropriate physics constraints and baseline metrics.

### test_quality_metrics_creation(self)
**Описание:** Test quality metrics creation.

Physical Meaning:
    Verifies that quality metrics correctly capture
    physics validation scores and overall quality status.

### test_physics_constraints_validation(self)
**Описание:** Test physics constraints validation.

Physical Meaning:
    Verifies that physics constraints correctly
    validate metrics against 7D theory principles.

### test_degradation_report_creation(self)
**Описание:** Test degradation report creation.

Physical Meaning:
    Verifies that degradation reports correctly
    identify and analyze quality degradation patterns.

### test_quality_alert_creation(self)
**Описание:** Test quality alert creation.

Physical Meaning:
    Verifies that quality alerts correctly identify
    physics violations with appropriate severity levels.

### test_reporting_system_initialization(self)
**Описание:** Test reporting system initialization.

Physical Meaning:
    Verifies that reporting system initializes with
    physics interpretation capabilities.

### test_daily_report_creation(self)
**Описание:** Test daily report creation.

Physical Meaning:
    Verifies that daily reports correctly summarize
    physics validation progress and key insights.

### test_weekly_report_creation(self)
**Описание:** Test weekly report creation.

Physical Meaning:
    Verifies that weekly reports correctly analyze
    trends in physics validation and quality metrics.

### test_monthly_report_creation(self)
**Описание:** Test monthly report creation.

Physical Meaning:
    Verifies that monthly reports provide comprehensive
    physics validation assessment and theoretical agreement.

### test_physics_interpreter_initialization(self)
**Описание:** Test physics interpreter initialization.

Physical Meaning:
    Verifies that physics interpreter correctly
    sets up physics context and interpretation rules.

### test_template_engine_initialization(self)
**Описание:** Test template engine initialization.

Physical Meaning:
    Verifies that template engine correctly sets up
    physics-aware templates for different audiences.

### test_data_aggregator_initialization(self)
**Описание:** Test data aggregator initialization.

Physical Meaning:
    Verifies that data aggregator correctly sets up
    for physics-aware data aggregation.

### test_end_to_end_automated_testing(self)
**Описание:** Test end-to-end automated testing workflow.

Physical Meaning:
    Verifies that the complete automated testing
    workflow correctly executes with physics validation.

### _create_integration_config(self)
**Описание:** Create integration test configuration.

## ./tests/unit/test_core/fft_solver_7d_validation/test_basic_validation.py
Methods: 9

### domain_7d(self)
**Описание:** Create 7D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f952e10>

### parameters_basic(self)
**Описание:** Basic parameters for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f952450>

### solver(self, domain_7d, parameters_basic)
**Описание:** Create FFT solver for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f951450>

### _create_plane_wave_source(self, domain, k_mode, amplitude)
**Описание:** Create a plane wave source for testing.

### test_A01_plane_wave_stationary(self, solver, domain_7d)
**Описание:** Test A0.1: Plane wave stationary solution.

Physical Meaning:
    Tests the fundamental spectral solution for a plane wave source,
    validating the formula â = ŝ / D for single frequency modes.

### test_A02_analytical_constant_source(self, solver, domain_7d)
**Описание:** Test A0.2: Analytical solution for constant source.

Physical Meaning:
    Tests the solution for a constant source, which should produce
    a constant solution scaled by the damping parameter λ.

### test_A03_linearity_property(self, solver, domain_7d)
**Описание:** Test A0.3: Linearity property of the solver.

Physical Meaning:
    Tests that the solver is linear: L(a·s₁ + b·s₂) = a·L(s₁) + b·L(s₂)
    where L is the linear operator and s₁, s₂ are sources.

### test_A04_energy_conservation(self, solver, domain_7d)
**Описание:** Test A0.4: Energy conservation properties.

Physical Meaning:
    Tests that the solver conserves energy appropriately,
    validating the physical correctness of the solution.

### test_A05_parameter_dependence(self, solver, domain_7d)
**Описание:** Test A0.5: Parameter dependence validation.

Physical Meaning:
    Tests that the solution depends correctly on the physical parameters
    μ, β, and λ, validating the mathematical formulation.

## ./tests/unit/test_core/fft_solver_7d_validation/test_boundary_cases.py
Methods: 12

### domain_7d(self)
**Описание:** Create 7D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fba4b10>

### parameters_basic(self)
**Описание:** Basic parameters for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa1c950>

### solver(self, domain_7d, parameters_basic)
**Описание:** Create FFT solver for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa1f550>

### test_C01_zero_source(self, solver, domain_7d)
**Описание:** Test C0.1: Zero source test.

Physical Meaning:
    Tests that the solver correctly handles zero source,
    which should produce zero solution.

### test_C02_very_small_source(self, solver, domain_7d)
**Описание:** Test C0.2: Very small source test.

Physical Meaning:
    Tests that the solver handles very small source values
    without numerical issues.

### test_C03_very_large_source(self, solver, domain_7d)
**Описание:** Test C0.3: Very large source test.

Physical Meaning:
    Tests that the solver handles very large source values
    without overflow or numerical instability.

### test_C04_extreme_parameters(self, domain_7d)
**Описание:** Test C0.4: Extreme parameter values test.

Physical Meaning:
    Tests that the solver handles extreme parameter values
    without breaking or producing invalid results.

### test_C05_singular_conditions(self, domain_7d)
**Описание:** Test C0.5: Singular conditions test.

Physical Meaning:
    Tests that the solver handles singular or near-singular
    conditions appropriately.

### test_C06_memory_usage(self, solver, domain_7d)
**Описание:** Test C0.6: Memory usage test.

Physical Meaning:
    Tests that the solver doesn't consume excessive memory
    for reasonable problem sizes.

### test_C07_error_handling(self, domain_7d)
**Описание:** Test C0.7: Error handling test.

Physical Meaning:
    Tests that the solver properly handles invalid inputs
    and produces appropriate error messages.

### test_C08_performance_benchmark(self, solver, domain_7d)
**Описание:** Test C0.8: Performance benchmark test.

Physical Meaning:
    Tests that the solver performs reasonably well
    for typical problem sizes.

### test_C09_consistency_across_runs(self, solver, domain_7d)
**Описание:** Test C0.9: Consistency across multiple runs.

Physical Meaning:
    Tests that the solver produces consistent results
    across multiple runs with the same input.

## ./tests/unit/test_core/fft_solver_7d_validation/test_numerical_validation.py
Methods: 9

### domain_7d(self)
**Описание:** Create 7D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fba6490>

### parameters_basic(self)
**Описание:** Basic parameters for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb20d50>

### solver(self, domain_7d, parameters_basic)
**Описание:** Create FFT solver for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb08290>

### _create_gaussian_source(self, domain, center, width)
**Описание:** Create a Gaussian source for testing.

### test_B01_convergence_test(self, domain_7d)
**Описание:** Test B0.1: Convergence test with increasing resolution.

Physical Meaning:
    Tests that the solution converges as the grid resolution increases,
    validating the numerical accuracy of the solver.

### test_B02_boundary_conditions(self, solver, domain_7d)
**Описание:** Test B0.2: Boundary condition handling.

Physical Meaning:
    Tests that the solver handles boundary conditions correctly,
    ensuring proper behavior at domain boundaries.

### test_B03_numerical_stability(self, solver, domain_7d)
**Описание:** Test B0.3: Numerical stability test.

Physical Meaning:
    Tests that the solver is numerically stable for various
    parameter combinations and source configurations.

### test_B04_precision_validation(self, domain_7d)
**Описание:** Test B0.4: Precision validation test.

Physical Meaning:
    Tests that the solver maintains appropriate precision
    for different numerical precision settings.

### test_B05_spectral_accuracy(self, solver, domain_7d)
**Описание:** Test B0.5: Spectral accuracy validation.

Physical Meaning:
    Tests that the spectral representation is accurate,
    validating the FFT operations and spectral coefficients.

## ./tests/unit/test_core/frequency_dependent_properties/test_advanced_properties.py
Methods: 12

### domain_7d(self)
**Описание:** Create 7D domain for constants testing.
**Декораторы:** <ast.Attribute object at 0x753e6f976610>

### bvp_constants(self)
**Описание:** Create BVP constants for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f975910>

### freq_props(self, domain_7d, bvp_constants)
**Описание:** Create frequency-dependent properties for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f975210>

### test_frequency_array_operations(self, freq_props)
**Описание:** Test operations on frequency arrays.

### test_susceptibility_frequency_dependence(self, freq_props)
**Описание:** Test frequency dependence of susceptibility.

### test_dispersion_relation_consistency(self, freq_props)
**Описание:** Test consistency of dispersion relation.

### test_velocity_relationships(self, freq_props)
**Описание:** Test relationships between different velocities.

### test_refractive_index_properties(self, freq_props)
**Описание:** Test properties of refractive index.

### test_absorption_properties(self, freq_props)
**Описание:** Test properties of absorption coefficient.

### test_high_frequency_behavior(self, freq_props)
**Описание:** Test behavior at high frequencies.

### test_low_frequency_behavior(self, freq_props)
**Описание:** Test behavior at low frequencies.

### test_parameter_sensitivity(self, domain_7d)
**Описание:** Test sensitivity to parameter changes.

## ./tests/unit/test_core/frequency_dependent_properties/test_basic_properties.py
Methods: 12

### domain_7d(self)
**Описание:** Create 7D domain for constants testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa8ad10>

### bvp_constants(self)
**Описание:** Create BVP constants for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa8be10>

### freq_props(self, domain_7d, bvp_constants)
**Описание:** Create frequency-dependent properties for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb1a1d0>

### test_frequency_domain_creation(self, freq_props, domain_7d)
**Описание:** Test that frequency domain is created correctly.

### test_susceptibility_calculation(self, freq_props)
**Описание:** Test susceptibility calculation.

### test_dispersion_relation(self, freq_props)
**Описание:** Test dispersion relation calculation.

### test_phase_velocity(self, freq_props)
**Описание:** Test phase velocity calculation.

### test_group_velocity(self, freq_props)
**Описание:** Test group velocity calculation.

### test_absorption_coefficient(self, freq_props)
**Описание:** Test absorption coefficient calculation.

### test_refractive_index(self, freq_props)
**Описание:** Test refractive index calculation.

### test_physical_constraints(self, freq_props)
**Описание:** Test that physical constraints are satisfied.

### test_energy_conservation(self, freq_props)
**Описание:** Test energy conservation properties.

## ./tests/unit/test_core/nonlinear_coefficients/test_advanced_coefficients.py
Methods: 12

### domain_7d(self)
**Описание:** Create 7D domain for constants testing.
**Декораторы:** <ast.Attribute object at 0x753e6f90b390>

### bvp_constants(self)
**Описание:** Create BVP constants for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f90b1d0>

### nonlinear_coeffs(self, bvp_constants)
**Описание:** Create nonlinear coefficients for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f90b910>

### test_nonlinear_effects(self, nonlinear_coeffs)
**Описание:** Test nonlinear effects of coefficients.

### test_susceptibility_properties(self, nonlinear_coeffs)
**Описание:** Test susceptibility coefficient properties.

### test_frequency_dependence(self, nonlinear_coeffs)
**Описание:** Test frequency dependence of coefficients.

### test_material_properties(self, nonlinear_coeffs, bvp_constants)
**Описание:** Test material property relationships.

### test_extreme_parameter_values(self, domain_7d)
**Описание:** Test behavior with extreme parameter values.

### test_zero_nonlinear_coefficient(self, domain_7d)
**Описание:** Test behavior with zero nonlinear coefficient.

### test_susceptibility_limits(self, domain_7d)
**Описание:** Test susceptibility coefficient limits.

### test_carrier_frequency_limits(self, domain_7d)
**Описание:** Test carrier frequency limits.

### test_coefficient_consistency(self, nonlinear_coeffs)
**Описание:** Test consistency between different coefficients.

## ./tests/unit/test_core/nonlinear_coefficients/test_basic_coefficients.py
Methods: 12

### domain_7d(self)
**Описание:** Create 7D domain for constants testing.
**Декораторы:** <ast.Attribute object at 0x753e6f910690>

### bvp_constants(self)
**Описание:** Create BVP constants for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f913350>

### nonlinear_coeffs(self, bvp_constants)
**Описание:** Create nonlinear coefficients for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f910a90>

### test_nonlinear_coefficients_creation(self, nonlinear_coeffs, domain_7d)
**Описание:** Test that nonlinear coefficients are created correctly.

### test_kappa_coefficients(self, nonlinear_coeffs)
**Описание:** Test kappa coefficients properties.

### test_chi_coefficients(self, nonlinear_coeffs)
**Описание:** Test chi coefficients properties.

### test_carrier_frequency(self, nonlinear_coeffs)
**Описание:** Test carrier frequency properties.

### test_k0_squared(self, nonlinear_coeffs)
**Описание:** Test k0_squared properties.

### test_physical_constraints(self, nonlinear_coeffs)
**Описание:** Test that physical constraints are satisfied.

### test_parameter_relationships(self, nonlinear_coeffs)
**Описание:** Test relationships between parameters.

### test_dimensional_consistency(self, nonlinear_coeffs)
**Описание:** Test dimensional consistency of coefficients.

### test_coefficient_scaling(self, domain_7d)
**Описание:** Test coefficient scaling with different parameters.

## ./tests/unit/test_core/test_7d_physics.py
Methods: 20

### domain_7d(self)
**Описание:** Create 7D domain for physical testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa874d0>

### bvp_constants(self)
**Описание:** Create BVP constants for physical testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa86c50>

### envelope_solver(self, domain_7d, bvp_constants)
**Описание:** Create envelope solver for physical testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa86550>

### quench_detector(self, domain_7d, bvp_constants)
**Описание:** Create quench detector for physical testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa85e50>

### test_7d_envelope_equation_physics(self, envelope_solver, domain_7d)
**Описание:** Test physical correctness of 7D envelope equation.

Physical Meaning:
    Validates that the 7D envelope equation correctly implements
    the physics of the Base High-Frequency Field in 7D space-time...

### test_u1_phase_structure_physics(self, domain_7d, bvp_constants)
**Описание:** Test U(1)³ phase structure physics.

Physical Meaning:
    Validates that the phase field correctly implements the U(1)³
    phase structure a = |a|e^(iφ₁)e^(iφ₂)e^(iφ₃) with proper
    phase coherenc...

### test_energy_conservation_physics(self, envelope_solver, domain_7d)
**Описание:** Test energy conservation in 7D BVP system.

Physical Meaning:
    Validates that energy is conserved in the 7D BVP system,
    ensuring the fundamental conservation law ∂E/∂t + ∇·S = 0
    is satisfie...

### test_quench_dynamics_physics(self, quench_detector, domain_7d)
**Описание:** Test quench dynamics physics.

Physical Meaning:
    Validates that quench detection correctly identifies
    regions where the field gradient exceeds the threshold,
    indicating phase transitions i...

### test_spectral_properties_physics(self, envelope_solver, domain_7d)
**Описание:** Test spectral properties of 7D BVP field.

Physical Meaning:
    Validates that the spectral properties of the BVP field
    are consistent with the theoretical predictions for
    the 7D space-time s...

### _create_physical_source(self, domain)
**Описание:** Create a source with known physical properties.

### _create_time_evolving_source(self, domain) -> list
**Описание:** Create time-evolving source for energy conservation test.

### _create_envelope_with_quenches(self, domain)
**Описание:** Create envelope with known quench regions.

### _create_spectral_source(self, domain)
**Описание:** Create source with known spectral properties.

### _validate_boundary_conditions(self, envelope, domain)
**Описание:** Validate boundary conditions.

### _validate_energy_bounds(self, envelope, domain)
**Описание:** Validate energy bounds.

### _compute_total_energy(self, envelope, domain) -> float
**Описание:** Compute total energy of the envelope.

### _compute_gradient_magnitude(self, envelope, domain)
**Описание:** Compute gradient magnitude.

### _compute_spatial_spectrum(self, envelope, domain)
**Описание:** Compute spatial spectrum.

### _compute_phase_spectrum(self, envelope, domain)
**Описание:** Compute phase spectrum.

### _fit_power_law(self, spectrum) -> float
**Описание:** Fit power law to spectrum.

## ./tests/unit/test_core/test_base_time_integrator.py
Methods: 6

### domain_7d(self)
**Описание:** Create 7D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa52590>

### parameters_basic(self)
**Описание:** Basic parameters for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa50f90>

### test_parameter_validation(self, domain_7d, parameters_basic)
**Описание:** Test parameter validation.

Physical Meaning:
    Validates that the integrator correctly validates physical
    parameters and raises appropriate errors for invalid values.

### test_abstract_methods(self, domain_7d, parameters_basic)
**Описание:** Test that abstract methods raise NotImplementedError.

Physical Meaning:
    Validates that the abstract base class properly enforces
    implementation of required methods in subclasses.

### test_domain_validation(self, parameters_basic)
**Описание:** Test domain validation.

Physical Meaning:
    Validates that the integrator correctly validates the
    computational domain and raises appropriate errors.

### test_initialization_state(self, domain_7d, parameters_basic)
**Описание:** Test initialization state tracking.

Physical Meaning:
    Validates that the integrator correctly tracks its
    initialization state and prevents operations before
    proper initialization.

## ./tests/unit/test_core/test_basic_physics_validation.py
Methods: 15

### domain_7d(self)
**Описание:** Create 7D domain for basic testing.
**Декораторы:** <ast.Attribute object at 0x753e6f915410>

### test_domain_7d_structure_physics(self, domain_7d)
**Описание:** Test 7D domain structure physics.

Physical Meaning:
    Validates that the 7D domain correctly represents
    the 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematical Foundation:
    Tests domain proper...

### test_field_energy_conservation_physics(self, domain_7d)
**Описание:** Test field energy conservation physics.

Physical Meaning:
    Validates that energy calculations are physically
    meaningful and conserve energy.

Mathematical Foundation:
    Tests energy conserva...

### test_gradient_physics(self, domain_7d)
**Описание:** Test gradient computation physics.

Physical Meaning:
    Validates that gradient computations are physically
    meaningful and follow mathematical principles.

Mathematical Foundation:
    Tests gra...

### test_laplacian_physics(self, domain_7d)
**Описание:** Test Laplacian computation physics.

Physical Meaning:
    Validates that Laplacian computations are physically
    meaningful and follow mathematical principles.

Mathematical Foundation:
    Tests L...

### test_fft_energy_conservation_physics(self, domain_7d)
**Описание:** Test FFT energy conservation physics.

Physical Meaning:
    Validates that FFT operations conserve energy
    (Parseval's theorem).

Mathematical Foundation:
    Tests Parseval's theorem: ∫|a(x)|²dx ...

### test_boundary_conditions_physics(self, domain_7d)
**Описание:** Test boundary conditions physics.

Physical Meaning:
    Validates that boundary conditions are physically
    meaningful and maintain field properties.

Mathematical Foundation:
    Tests periodic bo...

### test_phase_structure_physics(self, domain_7d)
**Описание:** Test phase structure physics.

Physical Meaning:
    Validates that phase structure is physically
    meaningful and follows U(1)³ symmetry.

Mathematical Foundation:
    Tests phase decomposition: a ...

### _create_test_field(self, domain)
**Описание:** Create test field for physics validation.

### _create_gradient_test_field(self, domain)
**Описание:** Create field with known gradient for testing.

### _create_laplacian_test_field(self, domain)
**Описание:** Create field with known Laplacian for testing.

### _create_phase_test_field(self, domain)
**Описание:** Create field with phase structure for testing.

### _compute_field_energy(self, field, domain) -> float
**Описание:** Compute field energy.

### _compute_gradient(self, field, domain)
**Описание:** Compute field gradient.

### _compute_laplacian(self, field, domain)
**Описание:** Compute field Laplacian.

## ./tests/unit/test_core/test_bvp_constants_base_coverage.py
Methods: 12

### domain_7d(self)
**Описание:** Create 7D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa87810>

### config(self)
**Описание:** Create test configuration.
**Декораторы:** <ast.Attribute object at 0x753e6fa87710>

### test_bvp_constants_base_creation(self, domain_7d, config)
**Описание:** Test BVP constants base creation.

### test_bvp_constants_base_methods(self, domain_7d, config)
**Описание:** Test BVP constants base methods.

### test_bvp_constants_base_validation(self, domain_7d, config)
**Описание:** Test BVP constants base validation.

### test_bvp_constants_base_physical_properties(self, domain_7d, config)
**Описание:** Test BVP constants base physical properties.

### test_bvp_constants_base_numerical_properties(self, domain_7d, config)
**Описание:** Test BVP constants base numerical properties.

### test_bvp_constants_base_derived_properties(self, domain_7d, config)
**Описание:** Test BVP constants base derived properties.

### test_bvp_constants_base_parameter_access(self, domain_7d, config)
**Описание:** Test BVP constants base parameter access.

### test_bvp_constants_base_serialization(self, domain_7d, config)
**Описание:** Test BVP constants base serialization.

### test_bvp_constants_base_comparison(self, domain_7d, config)
**Описание:** Test BVP constants base comparison.

### test_bvp_constants_base_string_representation(self, domain_7d, config)
**Описание:** Test BVP constants base string representation.

## ./tests/unit/test_core/test_bvp_constants_comprehensive.py
Methods: 41

### config(self)
**Описание:** Create test configuration.
**Декораторы:** <ast.Attribute object at 0x753e6face690>

### constants_base(self, config)
**Описание:** Create BVPConstantsBase instance.
**Декораторы:** <ast.Attribute object at 0x753e6f919050>

### test_constants_base_initialization(self, constants_base)
**Описание:** Test constants base initialization.

### test_constants_base_default_initialization(self)
**Описание:** Test constants base with default values.

### test_constants_base_material_properties(self, constants_base)
**Описание:** Test material properties setup.

### test_constants_base_physical_constants(self, constants_base)
**Описание:** Test physical constants setup.

### test_constants_base_quench_parameters(self, constants_base)
**Описание:** Test quench parameters setup.

### test_get_envelope_parameter(self, constants_base)
**Описание:** Test getting envelope parameters.

### test_get_envelope_parameter_invalid(self, constants_base)
**Описание:** Test getting invalid envelope parameter.

### test_get_basic_material_property(self, constants_base)
**Описание:** Test getting basic material properties.

### test_get_basic_material_property_invalid(self, constants_base)
**Описание:** Test getting invalid material property.

### test_get_physical_constant(self, constants_base)
**Описание:** Test getting physical constants.

### test_get_physical_constant_invalid(self, constants_base)
**Описание:** Test getting invalid physical constant.

### test_get_physical_parameter(self, constants_base)
**Описание:** Test getting physical parameters.

### test_get_physical_parameter_invalid(self, constants_base)
**Описание:** Test getting invalid physical parameter.

### test_get_quench_parameter(self, constants_base)
**Описание:** Test getting quench parameters.

### test_get_quench_parameter_invalid(self, constants_base)
**Описание:** Test getting invalid quench parameter.

### constants_advanced(self, config)
**Описание:** Create BVPConstantsAdvanced instance.
**Декораторы:** <ast.Attribute object at 0x753e6facc650>

### test_constants_advanced_initialization(self, constants_advanced)
**Описание:** Test constants advanced initialization.

### test_constants_advanced_renormalized_coeffs(self, constants_advanced)
**Описание:** Test renormalized coefficients.

### test_constants_advanced_boundary_coeffs(self, constants_advanced)
**Описание:** Test boundary coefficients.

### test_get_advanced_material_property(self, constants_advanced)
**Описание:** Test getting advanced material properties.

### test_get_advanced_material_property_invalid(self, constants_advanced)
**Описание:** Test getting invalid advanced material property.

### test_constants_advanced_components(self, constants_advanced)
**Описание:** Test that components are initialized.

### mock_constants(self)
**Описание:** Create mock constants.
**Декораторы:** <ast.Attribute object at 0x753e6f9418d0>

### frequency_properties(self, mock_constants)
**Описание:** Create FrequencyDependentProperties instance.
**Декораторы:** <ast.Attribute object at 0x753e6f914390>

### test_frequency_properties_initialization(self, frequency_properties, mock_constants)
**Описание:** Test frequency properties initialization.

### test_compute_frequency_dependent_conductivity(self, frequency_properties)
**Описание:** Test frequency-dependent conductivity computation.

### test_compute_frequency_dependent_capacitance(self, frequency_properties)
**Описание:** Test frequency-dependent capacitance computation.

### test_compute_frequency_dependent_inductance(self, frequency_properties)
**Описание:** Test frequency-dependent inductance computation.

### test_compute_frequency_dependent_conductivity_scalar(self, frequency_properties)
**Описание:** Test frequency-dependent conductivity with scalar input.

### test_compute_frequency_dependent_capacitance_scalar(self, frequency_properties)
**Описание:** Test frequency-dependent capacitance with scalar input.

### test_compute_frequency_dependent_inductance_scalar(self, frequency_properties)
**Описание:** Test frequency-dependent inductance with scalar input.

### nonlinear_coeffs(self, mock_constants)
**Описание:** Create NonlinearCoefficients instance.
**Декораторы:** <ast.Attribute object at 0x753e6f94b690>

### test_nonlinear_coeffs_initialization(self, nonlinear_coeffs, mock_constants)
**Описание:** Test nonlinear coefficients initialization.

### test_compute_nonlinear_admittance_coefficients(self, nonlinear_coeffs)
**Описание:** Test nonlinear admittance coefficients computation.

### test_compute_nonlinear_admittance_coefficients_scalar(self, nonlinear_coeffs)
**Описание:** Test nonlinear admittance coefficients with scalar inputs.

### renormalized_coeffs(self, mock_constants)
**Описание:** Create RenormalizedCoefficients instance.
**Декораторы:** <ast.Attribute object at 0x753e6f941050>

### test_renormalized_coeffs_initialization(self, renormalized_coeffs, mock_constants)
**Описание:** Test renormalized coefficients initialization.

### test_compute_renormalized_coefficients(self, renormalized_coeffs)
**Описание:** Test renormalized coefficients computation.

### test_compute_renormalized_coefficients_scalar(self, renormalized_coeffs)
**Описание:** Test renormalized coefficients with scalar inputs.

## ./tests/unit/test_core/test_bvp_constants_coverage.py
Methods: 14

### test_bvp_constants_base_creation(self)
**Описание:** Test BVP constants base creation.

### test_bvp_constants_advanced_creation(self)
**Описание:** Test BVP constants advanced creation.

### test_frequency_dependent_properties_creation(self)
**Описание:** Test frequency dependent properties creation.

### test_nonlinear_coefficients_creation(self)
**Описание:** Test nonlinear coefficients creation.

### test_renormalized_coefficients_creation(self)
**Описание:** Test renormalized coefficients creation.

### test_bvp_constants_base_properties(self)
**Описание:** Test BVP constants base properties.

### test_bvp_constants_advanced_properties(self)
**Описание:** Test BVP constants advanced properties.

### test_bvp_constants_base_methods(self)
**Описание:** Test BVP constants base methods.

### test_bvp_constants_advanced_methods(self)
**Описание:** Test BVP constants advanced methods.

### test_frequency_dependent_properties_methods(self)
**Описание:** Test frequency dependent properties methods.

### test_nonlinear_coefficients_methods(self)
**Описание:** Test nonlinear coefficients methods.

### test_renormalized_coefficients_methods(self)
**Описание:** Test renormalized coefficients methods.

### test_bvp_constants_validation(self)
**Описание:** Test BVP constants validation.

### test_bvp_constants_repr(self)
**Описание:** Test BVP constants string representation.

## ./tests/unit/test_core/test_bvp_constants_numerical_coverage.py
Methods: 13

### domain_7d(self)
**Описание:** Create 7D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa2ab10>

### config(self)
**Описание:** Create test configuration.
**Декораторы:** <ast.Attribute object at 0x753e6fa4fb90>

### test_bvp_constants_numerical_creation(self, domain_7d, config)
**Описание:** Test BVP constants numerical creation.

### test_bvp_constants_numerical_precision(self, domain_7d, config)
**Описание:** Test BVP constants numerical precision methods.

### test_bvp_constants_numerical_tolerance(self, domain_7d, config)
**Описание:** Test BVP constants numerical tolerance methods.

### test_bvp_constants_numerical_iterations(self, domain_7d, config)
**Описание:** Test BVP constants numerical iteration methods.

### test_bvp_constants_numerical_convergence(self, domain_7d, config)
**Описание:** Test BVP constants numerical convergence methods.

### test_bvp_constants_numerical_validation(self, domain_7d, config)
**Описание:** Test BVP constants numerical validation.

### test_bvp_constants_numerical_properties(self, domain_7d, config)
**Описание:** Test BVP constants numerical properties.

### test_bvp_constants_numerical_parameter_access(self, domain_7d, config)
**Описание:** Test BVP constants numerical parameter access.

### test_bvp_constants_numerical_serialization(self, domain_7d, config)
**Описание:** Test BVP constants numerical serialization.

### test_bvp_constants_numerical_comparison(self, domain_7d, config)
**Описание:** Test BVP constants numerical comparison.

### test_bvp_constants_numerical_string_representation(self, domain_7d, config)
**Описание:** Test BVP constants numerical string representation.

## ./tests/unit/test_core/test_bvp_constants_physics.py
Methods: 12

### domain_7d(self)
**Описание:** Create 7D domain for constants testing.
**Декораторы:** <ast.Attribute object at 0x753e6f911b10>

### bvp_constants(self)
**Описание:** Create BVP constants for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f912110>

### test_bvp_constants_physical_constraints(self, bvp_constants)
**Описание:** Test physical constraints on BVP constants.

Physical Meaning:
    Validates that BVP constants satisfy fundamental physical
    constraints required for a physically meaningful theory.

Mathematical ...

### test_bvp_constants_energy_conservation(self, bvp_constants)
**Описание:** Test energy conservation with BVP constants.

Physical Meaning:
    Validates that BVP constants maintain energy conservation
    in the 7D BVP theory.

Mathematical Foundation:
    Tests energy conse...

### test_bvp_constants_causality_constraints(self, bvp_constants)
**Описание:** Test causality constraints on BVP constants.

Physical Meaning:
    Validates that BVP constants satisfy causality constraints
    required for physical consistency.

Mathematical Foundation:
    Test...

### test_bvp_constants_thermodynamic_constraints(self, bvp_constants)
**Описание:** Test thermodynamic constraints on BVP constants.

Physical Meaning:
    Validates that BVP constants satisfy thermodynamic constraints
    required for physical consistency.

Mathematical Foundation:
...

### test_bvp_constants_7d_structure(self, bvp_constants)
**Описание:** Test BVP constants 7D structure consistency.

Physical Meaning:
    Validates that BVP constants are consistent with 7D structure
    of space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematical Foundation:
    ...

### test_bvp_constants_numerical_stability(self, bvp_constants)
**Описание:** Test BVP constants numerical stability.

Physical Meaning:
    Validates that BVP constants are numerically stable
    for computational purposes.

Mathematical Foundation:
    Tests numerical stabili...

### test_bvp_constants_precision(self, bvp_constants)
**Описание:** Test BVP constants precision.

Physical Meaning:
    Validates that BVP constants maintain high precision
    for computational purposes.

Mathematical Foundation:
    Tests precision of BVP constants...

### test_bvp_constants_validation(self, bvp_constants)
**Описание:** Test BVP constants validation.

Physical Meaning:
    Validates that BVP constants pass validation checks
    for physical consistency.

Mathematical Foundation:
    Tests validation of BVP constants.

### test_bvp_constants_consistency(self, bvp_constants)
**Описание:** Test BVP constants consistency.

Physical Meaning:
    Validates that BVP constants are consistent with each other
    and with the 7D BVP theory.

Mathematical Foundation:
    Tests consistency of BV...

### test_bvp_constants_physical_meaning(self, bvp_constants)
**Описание:** Test BVP constants physical meaning.

Physical Meaning:
    Validates that BVP constants have correct physical meaning
    in the 7D BVP theory.

Mathematical Foundation:
    Tests physical meaning of...

## ./tests/unit/test_core/test_bvp_core_coverage.py
Methods: 15

### domain_7d(self)
**Описание:** Create 7D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fada1d0>

### config(self)
**Описание:** Create test configuration.
**Декораторы:** <ast.Attribute object at 0x753e6fad9d50>

### bvp_constants(self, config)
**Описание:** Create BVP constants.
**Декораторы:** <ast.Attribute object at 0x753e6fadb410>

### test_bvp_core_creation(self, domain_7d, bvp_constants)
**Описание:** Test BVP core creation.

### test_bvp_core_solve_envelope(self, domain_7d, bvp_constants)
**Описание:** Test BVP core solve envelope method.

### test_bvp_core_compute_residual(self, domain_7d, bvp_constants)
**Описание:** Test BVP core compute residual method.

### test_bvp_core_compute_jacobian(self, domain_7d, bvp_constants)
**Описание:** Test BVP core compute jacobian method.

### test_bvp_core_compute_energy(self, domain_7d, bvp_constants)
**Описание:** Test BVP core compute energy method.

### test_bvp_core_compute_gradient(self, domain_7d, bvp_constants)
**Описание:** Test BVP core compute gradient method.

### test_bvp_core_compute_laplacian(self, domain_7d, bvp_constants)
**Описание:** Test BVP core compute laplacian method.

### test_bvp_core_validate_solution(self, domain_7d, bvp_constants)
**Описание:** Test BVP core validate solution method.

### test_bvp_core_get_solution_info(self, domain_7d, bvp_constants)
**Описание:** Test BVP core get solution info method.

### test_bvp_core_serialization(self, domain_7d, bvp_constants)
**Описание:** Test BVP core serialization.

### test_bvp_core_comparison(self, domain_7d, bvp_constants)
**Описание:** Test BVP core comparison.

### test_bvp_core_string_representation(self, domain_7d, bvp_constants)
**Описание:** Test BVP core string representation.

## ./tests/unit/test_core/test_bvp_interface_coverage.py
Methods: 14

### domain_7d(self)
**Описание:** Create 7D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa8a010>

### config(self)
**Описание:** Create test configuration.
**Декораторы:** <ast.Attribute object at 0x753e6fa88850>

### bvp_constants(self, config)
**Описание:** Create BVP constants.
**Декораторы:** <ast.Attribute object at 0x753e6fa897d0>

### test_bvp_interface_creation(self, domain_7d, bvp_constants)
**Описание:** Test BVP interface creation.

### test_bvp_interface_process_source(self, domain_7d, bvp_constants)
**Описание:** Test BVP interface process source method.

### test_bvp_interface_solve_envelope(self, domain_7d, bvp_constants)
**Описание:** Test BVP interface solve envelope method.

### test_bvp_interface_validate_postulates(self, domain_7d, bvp_constants)
**Описание:** Test BVP interface validate postulates method.

### test_bvp_interface_detect_quenches(self, domain_7d, bvp_constants)
**Описание:** Test BVP interface detect quenches method.

### test_bvp_interface_compute_impedance(self, domain_7d, bvp_constants)
**Описание:** Test BVP interface compute impedance method.

### test_bvp_interface_get_interface_info(self, domain_7d, bvp_constants)
**Описание:** Test BVP interface get interface info method.

### test_bvp_interface_validate_interface(self, domain_7d, bvp_constants)
**Описание:** Test BVP interface validate interface method.

### test_bvp_interface_serialization(self, domain_7d, bvp_constants)
**Описание:** Test BVP interface serialization.

### test_bvp_interface_comparison(self, domain_7d, bvp_constants)
**Описание:** Test BVP interface comparison.

### test_bvp_interface_string_representation(self, domain_7d, bvp_constants)
**Описание:** Test BVP interface string representation.

## ./tests/unit/test_core/test_bvp_postulate_coverage.py
Methods: 14

### test_bvp_postulate_base_creation(self)
**Описание:** Test BVP postulate base creation.

### test_quench_detector_creation(self)
**Описание:** Test quench detector creation.

### test_quench_detector_properties(self)
**Описание:** Test quench detector properties.

### test_quench_detector_methods(self)
**Описание:** Test quench detector methods.

### test_quench_detector_validation(self)
**Описание:** Test quench detector validation.

### test_quench_detector_repr(self)
**Описание:** Test quench detector string representation.

### test_quench_detector_edge_cases(self)
**Описание:** Test quench detector edge cases.

### test_quench_detector_numerical_stability(self)
**Описание:** Test quench detector numerical stability.

### test_quench_detector_performance(self)
**Описание:** Test quench detector performance.

### test_quench_detector_memory_usage(self)
**Описание:** Test quench detector memory usage.

### test_quench_detector_config_handling(self)
**Описание:** Test quench detector configuration handling.

### test_quench_detector_error_handling(self)
**Описание:** Test quench detector error handling.

### test_quench_detector_statistics(self)
**Описание:** Test quench detector statistics.

### test_quench_detector_7d_structure(self)
**Описание:** Test quench detector 7D structure handling.

## ./tests/unit/test_core/test_bvp_postulates_coverage.py
Methods: 14

### domain_7d(self)
**Описание:** Create 7D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb4a810>

### config(self)
**Описание:** Create test configuration.
**Декораторы:** <ast.Attribute object at 0x753e6fb4a5d0>

### bvp_constants(self, config)
**Описание:** Create BVP constants.
**Декораторы:** <ast.Attribute object at 0x753e6fb49ed0>

### test_bvp_postulates_creation(self, domain_7d, bvp_constants)
**Описание:** Test BVP postulates creation.

### test_bvp_postulates_validate_all_postulates(self, domain_7d, bvp_constants)
**Описание:** Test BVP postulates validate all postulates method.

### test_bvp_postulates_validate_postulate(self, domain_7d, bvp_constants)
**Описание:** Test BVP postulates validate postulate method.

### test_bvp_postulates_get_postulate_list(self, domain_7d, bvp_constants)
**Описание:** Test BVP postulates get postulate list method.

### test_bvp_postulates_get_postulate_info(self, domain_7d, bvp_constants)
**Описание:** Test BVP postulates get postulate info method.

### test_bvp_postulates_compute_postulate_metrics(self, domain_7d, bvp_constants)
**Описание:** Test BVP postulates compute postulate metrics method.

### test_bvp_postulates_validate_postulate_consistency(self, domain_7d, bvp_constants)
**Описание:** Test BVP postulates validate postulate consistency method.

### test_bvp_postulates_get_postulate_summary(self, domain_7d, bvp_constants)
**Описание:** Test BVP postulates get postulate summary method.

### test_bvp_postulates_serialization(self, domain_7d, bvp_constants)
**Описание:** Test BVP postulates serialization.

### test_bvp_postulates_comparison(self, domain_7d, bvp_constants)
**Описание:** Test BVP postulates comparison.

### test_bvp_postulates_string_representation(self, domain_7d, bvp_constants)
**Описание:** Test BVP postulates string representation.

## ./tests/unit/test_core/test_bvp_postulates_integration_physics.py
Methods: 5

### domain_7d(self)
**Описание:** Create 7D domain for postulate testing.
**Декораторы:** <ast.Attribute object at 0x753e6fad2910>

### bvp_constants(self)
**Описание:** Create BVP constants for postulate testing.
**Декораторы:** <ast.Attribute object at 0x753e6fbb83d0>

### test_envelope(self, domain_7d)
**Описание:** Create test envelope for postulate validation.
**Декораторы:** <ast.Attribute object at 0x753e6fb04590>

### test_all_postulates_integration_physics(self, domain_7d, bvp_constants, test_envelope)
**Описание:** Test integration of all BVP postulates.

Physical Meaning:
    Validates that all 9 BVP postulates work together to ensure
    complete physical consistency of the BVP theory.

Mathematical Foundation...

### _validate_physical_consistency(self, results)
**Описание:** Validate physical consistency across all postulate results.

## ./tests/unit/test_core/test_bvp_rigidity_postulate_physics.py
Methods: 4

### domain_7d(self)
**Описание:** Create 7D domain for postulate testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb56750>

### bvp_constants(self)
**Описание:** Create BVP constants for postulate testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb54850>

### test_envelope(self, domain_7d)
**Описание:** Create test envelope for postulate validation.
**Декораторы:** <ast.Attribute object at 0x753e6fa64d50>

### test_bvp_rigidity_postulate_physics(self, domain_7d, bvp_constants, test_envelope)
**Описание:** Test BVP Rigidity Postulate physics.

Physical Meaning:
    Validates that the BVP field maintains its structure and
    coherence under perturbations, ensuring field stability.

Mathematical Foundati...

## ./tests/unit/test_core/test_carrier_primacy_postulate_physics.py
Methods: 4

### domain_7d(self)
**Описание:** Create 7D domain for postulate testing.
**Декораторы:** <ast.Attribute object at 0x753e6fba4290>

### bvp_constants(self)
**Описание:** Create BVP constants for postulate testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa8d2d0>

### test_envelope(self, domain_7d)
**Описание:** Create test envelope for postulate validation.
**Декораторы:** <ast.Attribute object at 0x753e6fa1ec10>

### test_carrier_primacy_postulate_physics(self, domain_7d, bvp_constants, test_envelope)
**Описание:** Test Carrier Primacy Postulate physics.

Physical Meaning:
    Validates that the high-frequency carrier dominates the field
    structure, ensuring the BVP is truly a high-frequency field
    with en...

## ./tests/unit/test_core/test_core_renormalization_postulate_physics.py
Methods: 4

### domain_7d(self)
**Описание:** Create 7D domain for postulate testing.
**Декораторы:** <ast.Attribute object at 0x753e6fba5a50>

### bvp_constants(self)
**Описание:** Create BVP constants for postulate testing.
**Декораторы:** <ast.Attribute object at 0x753e6fba66d0>

### test_envelope(self, domain_7d)
**Описание:** Create test envelope for postulate validation.
**Декораторы:** <ast.Attribute object at 0x753e6fa86550>

### test_core_renormalization_postulate_physics(self, domain_7d, bvp_constants, test_envelope)
**Описание:** Test Core Renormalization Postulate physics.

Physical Meaning:
    Validates that renormalization effects in the field core
    are properly accounted for, ensuring physical consistency.

Mathematica...

## ./tests/unit/test_core/test_crank_nicolson_integrator.py
Methods: 11

### domain_7d(self)
**Описание:** Create 7D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb48210>

### parameters_basic(self)
**Описание:** Basic parameters for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb49850>

### integrator(self, domain_7d, parameters_basic)
**Описание:** Create Crank-Nicolson integrator for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb48190>

### test_initialization(self, integrator, domain_7d, parameters_basic)
**Описание:** Test integrator initialization.

Physical Meaning:
    Validates that the Crank-Nicolson integrator initializes correctly
    with the computational domain and physics parameters.

### test_single_step(self, integrator, domain_7d)
**Описание:** Test single time step.

Physical Meaning:
    Validates that a single time step produces physically reasonable
    results for the Crank-Nicolson integrator.

### test_implicit_step(self, integrator, domain_7d)
**Описание:** Test implicit time step.

Physical Meaning:
    Validates the implicit Crank-Nicolson scheme for unconditional
    stability in stiff problems.

### test_second_order_accuracy(self, integrator, domain_7d)
**Описание:** Test second-order accuracy.

Physical Meaning:
    Validates that the Crank-Nicolson integrator achieves second-order
    accuracy in time, which is its key advantage over explicit methods.

### test_unconditional_stability(self, integrator, domain_7d)
**Описание:** Test unconditional stability.

Physical Meaning:
    Validates that the Crank-Nicolson integrator remains stable
    even with large time steps, which is crucial for stiff problems.

### test_spectral_coefficients(self, integrator, domain_7d, parameters_basic)
**Описание:** Test spectral coefficients computation.

Physical Meaning:
    Validates that the spectral coefficients are computed correctly
    for the Crank-Nicolson integrator.

### test_time_step_validation(self, integrator, domain_7d)
**Описание:** Test time step validation.

Physical Meaning:
    Validates that the integrator properly validates time step sizes.

### test_field_validation(self, integrator, domain_7d)
**Описание:** Test field validation.

Physical Meaning:
    Validates that the integrator properly validates input fields.

## ./tests/unit/test_core/test_cuda_fft_backend_parity.py
Methods: 2

### test_forward_inverse_fft_parity_small_grid()
**Описание:** Нет докстринга

### __init__(self, shape)
**Описание:** Нет докстринга

## ./tests/unit/test_core/test_domain_comprehensive.py
Methods: 32

### domain(self)
**Описание:** Create domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb08d10>

### test_domain_initialization(self, domain)
**Описание:** Test domain initialization.

### test_domain_properties(self, domain)
**Описание:** Test domain properties.

### test_domain_coordinates(self, domain)
**Описание:** Test coordinate generation.

### test_domain_phase_coordinates(self, domain)
**Описание:** Test phase coordinate generation.

### test_domain_time_coordinates(self, domain)
**Описание:** Test time coordinate generation.

### test_domain_meshgrid(self, domain)
**Описание:** Test meshgrid generation.

### test_domain_phase_meshgrid(self, domain)
**Описание:** Test phase meshgrid generation.

### test_domain_validation(self)
**Описание:** Test domain validation.

### test_domain_repr(self, domain)
**Описание:** Test domain string representation.

### domain_7d(self)
**Описание:** Create 7D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb48a10>

### test_domain_7d_initialization(self, domain_7d)
**Описание:** Test 7D domain initialization.

### test_domain_7d_properties(self, domain_7d)
**Описание:** Test 7D domain properties.

### test_domain_7d_coordinates(self, domain_7d)
**Описание:** Test 7D coordinate generation.

### test_domain_7d_meshgrid(self, domain_7d)
**Описание:** Test 7D meshgrid generation.

### test_domain_7d_validation(self)
**Описание:** Test 7D domain validation.

### test_domain_7d_repr(self, domain_7d)
**Описание:** Test 7D domain string representation.

### field(self, domain)
**Описание:** Create field for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb0ab50>

### test_field_initialization(self, field, domain)
**Описание:** Test field initialization.

### test_field_set_data(self, field)
**Описание:** Test setting field data.

### test_field_get_data(self, field)
**Описание:** Test getting field data.

### test_field_energy(self, field)
**Описание:** Test field energy calculation.

### test_field_norm(self, field)
**Описание:** Test field norm calculation.

### test_field_gradient(self, field)
**Описание:** Test field gradient calculation.

### test_field_laplacian(self, field)
**Описание:** Test field Laplacian calculation.

### test_field_validation(self, field)
**Описание:** Test field validation.

### test_field_repr(self, field)
**Описание:** Test field string representation.

### parameters(self)
**Описание:** Create parameters for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa642d0>

### test_parameters_initialization(self, parameters)
**Описание:** Test parameters initialization.

### test_parameters_validation(self)
**Описание:** Test parameters validation.

### test_parameters_get_spectral_coefficients(self, parameters)
**Описание:** Test spectral coefficients calculation.

### test_parameters_repr(self, parameters)
**Описание:** Test parameters string representation.

## ./tests/unit/test_core/test_domain_coverage.py
Methods: 23

### test_domain_creation(self)
**Описание:** Test domain creation.

### test_domain_7d_creation(self)
**Описание:** Test 7D domain creation.

### test_field_creation(self)
**Описание:** Test field creation.

### test_parameters_creation(self)
**Описание:** Test parameters creation.

### test_domain_properties(self)
**Описание:** Test domain properties.

### test_domain_shape(self)
**Описание:** Test domain shape.

### test_domain_coordinates(self)
**Описание:** Test domain coordinates.

### test_domain_phase_coordinates(self)
**Описание:** Test domain phase coordinates.

### test_domain_time_coordinates(self)
**Описание:** Test domain time coordinates.

### test_domain_meshgrid(self)
**Описание:** Test domain meshgrid.

### test_domain_phase_meshgrid(self)
**Описание:** Test domain phase meshgrid.

### test_domain_validation(self)
**Описание:** Test domain validation.

### test_domain_repr(self)
**Описание:** Test domain string representation.

### test_field_properties(self)
**Описание:** Test field properties.

### test_field_energy(self)
**Описание:** Test field energy computation.

### test_field_norm(self)
**Описание:** Test field norm computation.

### test_field_gradient(self)
**Описание:** Test field gradient computation.

### test_field_laplacian(self)
**Описание:** Test field Laplacian computation.

### test_field_validation(self)
**Описание:** Test field validation.

### test_field_repr(self)
**Описание:** Test field string representation.

### test_parameters_properties(self)
**Описание:** Test parameters properties.

### test_parameters_validation(self)
**Описание:** Test parameters validation.

### test_parameters_repr(self)
**Описание:** Test parameters string representation.

## ./tests/unit/test_core/test_fft_backend.py
Methods: 14

### domain(self)
**Описание:** Create domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6faf4290>

### fft_backend(self, domain)
**Описание:** Create FFT backend for testing.
**Декораторы:** <ast.Attribute object at 0x753e6faf4e10>

### test_fft_backend_initialization(self, fft_backend, domain)
**Описание:** Test FFT backend initialization.

### test_fft_backend_forward_transform(self, fft_backend)
**Описание:** Test forward FFT transform.

### test_fft_backend_inverse_transform(self, fft_backend)
**Описание:** Test inverse FFT transform.

### test_fft_backend_round_trip(self, fft_backend)
**Описание:** Test round-trip FFT transform.

### test_fft_backend_energy_conservation(self, fft_backend)
**Описание:** Test FFT energy conservation (Parseval's theorem).

### test_fft_backend_get_wave_vectors(self, fft_backend)
**Описание:** Test wave vector computation.

### test_fft_backend_spectral_operations(self, fft_backend)
**Описание:** Test spectral operations.

### test_fft_backend_error_handling(self, fft_backend)
**Описание:** Test error handling for invalid inputs.

### test_fft_backend_memory_efficiency(self, fft_backend)
**Описание:** Test memory efficiency of FFT operations.

### test_fft_backend_numerical_stability(self, fft_backend)
**Описание:** Test numerical stability of FFT operations.

### test_fft_backend_precision(self, fft_backend)
**Описание:** Test FFT precision with known functions.

### test_fft_backend_7d_structure(self, fft_backend)
**Описание:** Test 7D structure preservation in FFT operations.

## ./tests/unit/test_core/test_fft_butterfly_computer.py
Methods: 21

### domain(self)
**Описание:** Create domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa1e1d0>

### fft_backend(self, domain)
**Описание:** Create FFT backend for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa1fe90>

### butterfly_computer(self, fft_backend)
**Описание:** Create FFT butterfly computer for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa1ff90>

### test_butterfly_computer_initialization(self, butterfly_computer, fft_backend)
**Описание:** Test FFT butterfly computer initialization.

### test_butterfly_computer_compute_butterfly(self, butterfly_computer)
**Описание:** Test butterfly computation.

### test_butterfly_computer_compute_inverse_butterfly(self, butterfly_computer)
**Описание:** Test inverse butterfly computation.

### test_butterfly_computer_butterfly_round_trip(self, butterfly_computer)
**Описание:** Test butterfly round-trip computation.

### test_butterfly_computer_butterfly_energy_conservation(self, butterfly_computer)
**Описание:** Test energy conservation in butterfly operations.

### test_butterfly_computer_butterfly_validation(self, butterfly_computer)
**Описание:** Test input validation for butterfly operations.

### test_butterfly_computer_butterfly_7d_structure(self, butterfly_computer)
**Описание:** Test 7D structure preservation in butterfly operations.

### test_butterfly_computer_butterfly_numerical_stability(self, butterfly_computer)
**Описание:** Test numerical stability of butterfly operations.

### test_butterfly_computer_butterfly_precision(self, butterfly_computer)
**Описание:** Test precision of butterfly operations.

### test_butterfly_computer_butterfly_performance(self, butterfly_computer)
**Описание:** Test performance of butterfly operations.

### test_butterfly_computer_butterfly_memory(self, butterfly_computer)
**Описание:** Test memory usage of butterfly operations.

### test_butterfly_computer_butterfly_statistics(self, butterfly_computer)
**Описание:** Test butterfly operation statistics.

### test_butterfly_computer_butterfly_optimization(self, butterfly_computer)
**Описание:** Test butterfly operation optimization.

### test_butterfly_computer_butterfly_parallel(self, butterfly_computer)
**Описание:** Test parallel butterfly computation.

### test_butterfly_computer_butterfly_vectorized(self, butterfly_computer)
**Описание:** Test vectorized butterfly computation.

### test_butterfly_computer_butterfly_error_handling(self, butterfly_computer)
**Описание:** Test error handling in butterfly operations.

### test_butterfly_computer_butterfly_edge_cases(self, butterfly_computer)
**Описание:** Test edge cases in butterfly operations.

### test_butterfly_computer_butterfly_complex_data(self, butterfly_computer)
**Описание:** Test butterfly operations with complex data.

## ./tests/unit/test_core/test_fft_coverage.py
Methods: 20

### domain(self)
**Описание:** Create domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f941190>

### test_fft_backend_creation(self, domain)
**Описание:** Test FFT backend creation.

### test_spectral_operations_creation(self, domain)
**Описание:** Test spectral operations creation.

### test_spectral_derivatives_creation(self, domain)
**Описание:** Test spectral derivatives creation.

### test_spectral_filtering_creation(self, domain)
**Описание:** Test spectral filtering creation.

### test_fft_plan_manager_creation(self, domain)
**Описание:** Test FFT plan manager creation.

### test_fft_butterfly_computer_creation(self, domain)
**Описание:** Test FFT butterfly computer creation.

### test_fft_twiddle_computer_creation(self, domain)
**Описание:** Test FFT twiddle computer creation.

### test_fft_backend_methods(self, domain)
**Описание:** Test FFT backend methods.

### test_spectral_operations_methods(self, domain)
**Описание:** Test spectral operations methods.

### test_spectral_derivatives_methods(self, domain)
**Описание:** Test spectral derivatives methods.

### test_spectral_filtering_methods(self, domain)
**Описание:** Test spectral filtering methods.

### test_fft_plan_manager_methods(self, domain)
**Описание:** Test FFT plan manager methods.

### test_fft_butterfly_computer_methods(self, domain)
**Описание:** Test FFT butterfly computer methods.

### test_fft_twiddle_computer_methods(self, domain)
**Описание:** Test FFT twiddle computer methods.

### test_fft_energy_conservation(self, domain)
**Описание:** Test FFT energy conservation.

### test_fft_round_trip(self, domain)
**Описание:** Test FFT round-trip accuracy.

### test_fft_7d_structure(self, domain)
**Описание:** Test FFT 7D structure preservation.

### test_fft_numerical_stability(self, domain)
**Описание:** Test FFT numerical stability.

### test_fft_precision(self, domain)
**Описание:** Test FFT precision.

## ./tests/unit/test_core/test_fft_physics.py
Methods: 11

### domain_7d(self)
**Описание:** Create 7D domain for spectral testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb0a250>

### fft_backend(self, domain_7d)
**Описание:** Create FFT backend for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb227d0>

### spectral_ops(self, fft_backend)
**Описание:** Create spectral operations for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa1f550>

### test_fft_energy_conservation_physics(self, domain_7d, spectral_ops)
**Описание:** Test FFT energy conservation (Parseval's theorem).

Physical Meaning:
    Validates that FFT operations conserve energy, ensuring
    Parseval's theorem is satisfied: ∫|a(x)|²dx = ∫|â(k)|²dk

Mathemat...

### test_fft_round_trip_physics(self, domain_7d, spectral_ops)
**Описание:** Test FFT round-trip accuracy.

Physical Meaning:
    Validates that forward and inverse FFT operations
    preserve field information accurately.

Mathematical Foundation:
    Tests FFT round-trip: a(...

### test_fft_7d_structure_physics(self, domain_7d, spectral_ops)
**Описание:** Test FFT 7D structure preservation.

Physical Meaning:
    Validates that FFT operations preserve the 7D structure
    of space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematical Foundation:
    Tests 7D FFT st...

### test_fft_numerical_stability_physics(self, domain_7d, spectral_ops)
**Описание:** Test FFT numerical stability.

Physical Meaning:
    Validates that FFT operations are numerically stable
    for extreme field values.

Mathematical Foundation:
    Tests numerical stability of 7D FF...

### test_fft_precision_physics(self, domain_7d, spectral_ops)
**Описание:** Test FFT precision with known functions.

Physical Meaning:
    Validates that FFT operations maintain high precision
    for known analytical functions.

Mathematical Foundation:
    Tests FFT precis...

### test_fft_boundary_conditions_physics(self, domain_7d, spectral_ops)
**Описание:** Test FFT boundary condition handling.

Physical Meaning:
    Validates that FFT operations handle boundary conditions
    correctly in 7D space-time.

Mathematical Foundation:
    Tests boundary condi...

### test_fft_phase_structure_physics(self, domain_7d, spectral_ops)
**Описание:** Test FFT phase structure preservation.

Physical Meaning:
    Validates that FFT operations preserve phase structure
    in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematical Foundation:
    Tests phas...

### _create_test_field(self, domain)
**Описание:** Create test field for FFT testing.

## ./tests/unit/test_core/test_fft_plan_manager.py
Methods: 24

### domain(self)
**Описание:** Create domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fad3810>

### fft_backend(self, domain)
**Описание:** Create FFT backend for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fad31d0>

### plan_manager(self, fft_backend)
**Описание:** Create FFT plan manager for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fad3050>

### test_plan_manager_initialization(self, plan_manager, fft_backend)
**Описание:** Test FFT plan manager initialization.

### test_plan_manager_create_plan(self, plan_manager)
**Описание:** Test FFT plan creation.

### test_plan_manager_get_plan(self, plan_manager)
**Описание:** Test FFT plan retrieval.

### test_plan_manager_plan_caching(self, plan_manager)
**Описание:** Test FFT plan caching.

### test_plan_manager_plan_optimization(self, plan_manager)
**Описание:** Test FFT plan optimization.

### test_plan_manager_plan_validation(self, plan_manager)
**Описание:** Test FFT plan validation.

### test_plan_manager_plan_cleanup(self, plan_manager)
**Описание:** Test FFT plan cleanup.

### test_plan_manager_plan_reset(self, plan_manager)
**Описание:** Test FFT plan reset.

### test_plan_manager_plan_execution(self, plan_manager)
**Описание:** Test FFT plan execution.

### test_plan_manager_plan_performance(self, plan_manager)
**Описание:** Test FFT plan performance.

### test_plan_manager_plan_memory(self, plan_manager)
**Описание:** Test FFT plan memory usage.

### test_plan_manager_plan_statistics(self, plan_manager)
**Описание:** Test FFT plan statistics.

### test_plan_manager_plan_comparison(self, plan_manager)
**Описание:** Test FFT plan comparison.

### test_plan_manager_plan_serialization(self, plan_manager)
**Описание:** Test FFT plan serialization.

### test_plan_manager_plan_deserialization(self, plan_manager)
**Описание:** Test FFT plan deserialization.

### test_plan_manager_plan_validation_errors(self, plan_manager)
**Описание:** Test FFT plan validation error handling.

### test_plan_manager_plan_cleanup_errors(self, plan_manager)
**Описание:** Test FFT plan cleanup error handling.

### test_plan_manager_plan_execution_errors(self, plan_manager)
**Описание:** Test FFT plan execution error handling.

### test_plan_manager_plan_statistics_errors(self, plan_manager)
**Описание:** Test FFT plan statistics error handling.

### test_plan_manager_plan_comparison_errors(self, plan_manager)
**Описание:** Test FFT plan comparison error handling.

### test_plan_manager_plan_serialization_errors(self, plan_manager)
**Описание:** Test FFT plan serialization error handling.

## ./tests/unit/test_core/test_fft_twiddle_computer.py
Methods: 27

### domain(self)
**Описание:** Create domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f94f9d0>

### fft_backend(self, domain)
**Описание:** Create FFT backend for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f94d250>

### twiddle_computer(self, fft_backend)
**Описание:** Create FFT twiddle computer for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f94c3d0>

### test_twiddle_computer_initialization(self, twiddle_computer, fft_backend)
**Описание:** Test FFT twiddle computer initialization.

### test_twiddle_computer_compute_twiddle_factors(self, twiddle_computer)
**Описание:** Test twiddle factors computation.

### test_twiddle_computer_get_twiddle_factor(self, twiddle_computer)
**Описание:** Test individual twiddle factor retrieval.

### test_twiddle_computer_compute_inverse_twiddle_factors(self, twiddle_computer)
**Описание:** Test inverse twiddle factors computation.

### test_twiddle_computer_twiddle_factor_caching(self, twiddle_computer)
**Описание:** Test twiddle factor caching.

### test_twiddle_computer_twiddle_factor_validation(self, twiddle_computer)
**Описание:** Test twiddle factor validation.

### test_twiddle_computer_twiddle_factor_energy_conservation(self, twiddle_computer)
**Описание:** Test energy conservation in twiddle factor operations.

### test_twiddle_computer_twiddle_factor_7d_structure(self, twiddle_computer)
**Описание:** Test 7D structure preservation in twiddle factor operations.

### test_twiddle_computer_twiddle_factor_numerical_stability(self, twiddle_computer)
**Описание:** Test numerical stability of twiddle factor operations.

### test_twiddle_computer_twiddle_factor_precision(self, twiddle_computer)
**Описание:** Test precision of twiddle factor operations.

### test_twiddle_computer_twiddle_factor_performance(self, twiddle_computer)
**Описание:** Test performance of twiddle factor operations.

### test_twiddle_computer_twiddle_factor_memory(self, twiddle_computer)
**Описание:** Test memory usage of twiddle factor operations.

### test_twiddle_computer_twiddle_factor_statistics(self, twiddle_computer)
**Описание:** Test twiddle factor statistics.

### test_twiddle_computer_twiddle_factor_optimization(self, twiddle_computer)
**Описание:** Test twiddle factor optimization.

### test_twiddle_computer_twiddle_factor_parallel(self, twiddle_computer)
**Описание:** Test parallel twiddle factor computation.

### test_twiddle_computer_twiddle_factor_vectorized(self, twiddle_computer)
**Описание:** Test vectorized twiddle factor computation.

### test_twiddle_computer_twiddle_factor_error_handling(self, twiddle_computer)
**Описание:** Test error handling in twiddle factor operations.

### test_twiddle_computer_twiddle_factor_edge_cases(self, twiddle_computer)
**Описание:** Test edge cases in twiddle factor operations.

### test_twiddle_computer_twiddle_factor_complex_data(self, twiddle_computer)
**Описание:** Test twiddle factor operations with complex data.

### test_twiddle_computer_twiddle_factor_unit_circle(self, twiddle_computer)
**Описание:** Test that twiddle factors lie on unit circle.

### test_twiddle_computer_twiddle_factor_phase_relationships(self, twiddle_computer)
**Описание:** Test phase relationships in twiddle factors.

### test_twiddle_computer_twiddle_factor_symmetry(self, twiddle_computer)
**Описание:** Test symmetry properties of twiddle factors.

### test_twiddle_computer_twiddle_factor_cleanup(self, twiddle_computer)
**Описание:** Test twiddle factor cleanup.

### test_twiddle_computer_twiddle_factor_reset(self, twiddle_computer)
**Описание:** Test twiddle factor reset.

## ./tests/unit/test_core/test_frequency_dependence_models.py
Methods: 3

### test_drude_conductivity_monotonicity()
**Описание:** Нет докстринга

### test_debye_conductivity_decreases_with_frequency()
**Описание:** Нет докстринга

### test_admittance_scales_with_conductivity()
**Описание:** Нет докстринга

## ./tests/unit/test_core/test_legacy_method_absence.py
Methods: 1

### test_no_basic_method_name_remaining()
**Описание:** Нет докстринга

## ./tests/unit/test_core/test_memory_kernel.py
Methods: 12

### domain_7d(self)
**Описание:** Create 7D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f917e90>

### memory_kernel(self, domain_7d)
**Описание:** Create memory kernel for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f9174d0>

### test_initialization(self, memory_kernel, domain_7d)
**Описание:** Test memory kernel initialization.

Physical Meaning:
    Validates that the memory kernel initializes correctly with
    the specified number of memory variables.

### test_memory_application(self, memory_kernel, domain_7d)
**Описание:** Test memory kernel application.

Physical Meaning:
    Validates that the memory kernel correctly applies non-local
    temporal effects to the field.

### test_memory_evolution(self, memory_kernel, domain_7d)
**Описание:** Test memory variable evolution.

Physical Meaning:
    Validates that memory variables evolve correctly according
    to their evolution equation.

### test_memory_reset(self, memory_kernel, domain_7d)
**Описание:** Test memory kernel reset.

Physical Meaning:
    Validates that the memory kernel can be reset to clear
    all memory of past configurations.

### test_relaxation_times(self, memory_kernel)
**Описание:** Test relaxation times.

Physical Meaning:
    Validates that relaxation times are positive and properly
    configured for the memory kernel.

### test_coupling_strengths(self, memory_kernel)
**Описание:** Test coupling strengths.

Physical Meaning:
    Validates that coupling strengths are properly configured
    for the memory kernel.

### test_memory_kernel_consistency(self, memory_kernel, domain_7d)
**Описание:** Test memory kernel consistency.

Physical Meaning:
    Validates that the memory kernel maintains consistency
    between its internal state and external interface.

### test_memory_kernel_linearity(self, memory_kernel, domain_7d)
**Описание:** Test memory kernel linearity.

Physical Meaning:
    Validates that the memory kernel behaves linearly
    for small field amplitudes.

### test_memory_kernel_time_dependence(self, memory_kernel, domain_7d)
**Описание:** Test memory kernel time dependence.

Physical Meaning:
    Validates that the memory kernel correctly handles
    time-dependent effects.

### test_memory_kernel_validation(self, memory_kernel, domain_7d)
**Описание:** Test memory kernel validation.

Physical Meaning:
    Validates that the memory kernel properly validates
    input parameters.

## ./tests/unit/test_core/test_operators_coverage.py
Methods: 25

### domain(self)
**Описание:** Create domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6faf7550>

### test_fractional_laplacian_creation(self, domain)
**Описание:** Test fractional Laplacian creation.

### test_memory_kernel_creation(self, domain)
**Описание:** Test memory kernel creation.

### test_operator_riesz_creation(self, domain)
**Описание:** Test operator Riesz creation.

### test_fractional_laplacian_methods(self, domain)
**Описание:** Test fractional Laplacian methods.

### test_memory_kernel_methods(self, domain)
**Описание:** Test memory kernel methods.

### test_operator_riesz_methods(self, domain)
**Описание:** Test operator Riesz methods.

### test_fractional_laplacian_validation(self, domain)
**Описание:** Test fractional Laplacian validation.

### test_memory_kernel_validation(self, domain)
**Описание:** Test memory kernel validation.

### test_operator_riesz_validation(self, domain)
**Описание:** Test operator Riesz validation.

### test_fractional_laplacian_7d_structure(self, domain)
**Описание:** Test fractional Laplacian 7D structure preservation.

### test_memory_kernel_7d_structure(self, domain)
**Описание:** Test memory kernel 7D structure preservation.

### test_operator_riesz_7d_structure(self, domain)
**Описание:** Test operator Riesz 7D structure preservation.

### test_fractional_laplacian_energy_conservation(self, domain)
**Описание:** Test fractional Laplacian energy conservation.

### test_memory_kernel_energy_conservation(self, domain)
**Описание:** Test memory kernel energy conservation.

### test_operator_riesz_energy_conservation(self, domain)
**Описание:** Test operator Riesz energy conservation.

### test_fractional_laplacian_precision(self, domain)
**Описание:** Test fractional Laplacian precision.

### test_memory_kernel_precision(self, domain)
**Описание:** Test memory kernel precision.

### test_operator_riesz_precision(self, domain)
**Описание:** Test operator Riesz precision.

### test_fractional_laplacian_error_handling(self, domain)
**Описание:** Test fractional Laplacian error handling.

### test_memory_kernel_error_handling(self, domain)
**Описание:** Test memory kernel error handling.

### test_operator_riesz_error_handling(self, domain)
**Описание:** Test operator Riesz error handling.

### test_fractional_laplacian_repr(self, domain)
**Описание:** Test fractional Laplacian string representation.

### test_memory_kernel_repr(self, domain)
**Описание:** Test memory kernel string representation.

### test_operator_riesz_repr(self, domain)
**Описание:** Test operator Riesz string representation.

## ./tests/unit/test_core/test_power_balance_postulate_physics.py
Methods: 4

### domain_7d(self)
**Описание:** Create 7D domain for postulate testing.
**Декораторы:** <ast.Attribute object at 0x753e6fad3150>

### bvp_constants(self)
**Описание:** Create BVP constants for postulate testing.
**Декораторы:** <ast.Attribute object at 0x753e6fcff350>

### test_envelope(self, domain_7d)
**Описание:** Create test envelope for postulate validation.
**Декораторы:** <ast.Attribute object at 0x753e6f9174d0>

### test_power_balance_postulate_physics(self, domain_7d, bvp_constants, test_envelope)
**Описание:** Test Power Balance Postulate physics.

Physical Meaning:
    Validates that energy is conserved in the BVP system,
    ensuring the fundamental conservation law is satisfied.

Mathematical Foundation:...

## ./tests/unit/test_core/test_quench_detector.py
Methods: 12

### domain_7d(self)
**Описание:** Create 7D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fad36d0>

### quench_detector(self, domain_7d)
**Описание:** Create quench detector for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fad3710>

### test_initialization(self, quench_detector, domain_7d)
**Описание:** Test quench detector initialization.

Physical Meaning:
    Validates that the quench detector initializes correctly with
    the specified thresholds.

### test_quench_detection_energy(self, quench_detector, domain_7d)
**Описание:** Test quench detection based on energy threshold.

Physical Meaning:
    Validates that the quench detector correctly identifies
    energy dumping events based on energy change threshold.

### test_quench_detection_magnitude(self, quench_detector, domain_7d)
**Описание:** Test quench detection based on magnitude threshold.

Physical Meaning:
    Validates that the quench detector correctly identifies
    quench events based on field magnitude threshold.

### test_quench_history(self, quench_detector, domain_7d)
**Описание:** Test quench event history.

Physical Meaning:
    Validates that the quench detector correctly records
    and manages quench event history.

### test_quench_clear_history(self, quench_detector, domain_7d)
**Описание:** Test quench history clearing.

Physical Meaning:
    Validates that the quench detector can clear its history
    to start fresh monitoring.

### test_quench_detection_rate(self, quench_detector, domain_7d)
**Описание:** Test quench detection based on rate threshold.

Physical Meaning:
    Validates that the quench detector correctly identifies
    quench events based on energy change rate threshold.

### test_quench_threshold_validation(self, domain_7d)
**Описание:** Test quench threshold validation.

Physical Meaning:
    Validates that the quench detector properly validates
    threshold parameters.

### test_quench_field_validation(self, quench_detector, domain_7d)
**Описание:** Test quench field validation.

Physical Meaning:
    Validates that the quench detector properly validates
    input fields.

### test_quench_statistics(self, quench_detector, domain_7d)
**Описание:** Test quench statistics computation.

Physical Meaning:
    Validates that the quench detector correctly computes
    statistics about quench events.

### test_quench_event_details(self, quench_detector, domain_7d)
**Описание:** Test quench event details.

Physical Meaning:
    Validates that the quench detector correctly records
    detailed information about quench events.

## ./tests/unit/test_core/test_quenches_postulate_physics.py
Methods: 4

### domain_7d(self)
**Описание:** Create 7D domain for postulate testing.
**Декораторы:** <ast.Attribute object at 0x753e6fbc4e50>

### bvp_constants(self)
**Описание:** Create BVP constants for postulate testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa524d0>

### test_envelope(self, domain_7d)
**Описание:** Create test envelope for postulate validation.
**Декораторы:** <ast.Attribute object at 0x753e6fb55310>

### test_quenches_postulate_physics(self, domain_7d, bvp_constants, test_envelope)
**Описание:** Test Quenches Postulate physics.

Physical Meaning:
    Validates that quench detection correctly identifies phase
    transition regions where the field gradient exceeds threshold.

Mathematical Foun...

## ./tests/unit/test_core/test_renormalized_coefficients_physics.py
Methods: 16

### domain_7d(self)
**Описание:** Create 7D domain for constants testing.
**Декораторы:** <ast.Attribute object at 0x753e6f9186d0>

### bvp_constants(self)
**Описание:** Create BVP constants for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f91b990>

### renormalized_coeffs(self, bvp_constants)
**Описание:** Create renormalized coefficients for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f91bad0>

### test_renormalized_coefficients_physics(self, renormalized_coeffs)
**Описание:** Test renormalized coefficients physical consistency.

Physical Meaning:
    Validates that renormalized coefficients satisfy physical constraints
    and theoretical requirements.

Mathematical Founda...

### test_renormalized_coefficients_energy_conservation_physics(self, renormalized_coeffs)
**Описание:** Test renormalized coefficients energy conservation.

Physical Meaning:
    Validates that renormalized coefficients maintain energy conservation
    in the 7D BVP theory.

Mathematical Foundation:
   ...

### test_renormalized_coefficients_causality_constraints_physics(self, renormalized_coeffs)
**Описание:** Test renormalized coefficients causality constraints.

Physical Meaning:
    Validates that renormalized coefficients satisfy causality constraints
    required for physical consistency.

Mathematical...

### test_renormalized_coefficients_thermodynamic_constraints_physics(self, renormalized_coeffs)
**Описание:** Test renormalized coefficients thermodynamic constraints.

Physical Meaning:
    Validates that renormalized coefficients satisfy thermodynamic constraints
    required for physical consistency.

Math...

### test_renormalized_coefficients_7d_structure_physics(self, renormalized_coeffs)
**Описание:** Test renormalized coefficients 7D structure consistency.

Physical Meaning:
    Validates that renormalized coefficients are consistent with 7D structure
    of space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathe...

### test_renormalized_coefficients_numerical_stability_physics(self, renormalized_coeffs)
**Описание:** Test renormalized coefficients numerical stability.

Physical Meaning:
    Validates that renormalized coefficients are numerically stable
    for computational purposes.

Mathematical Foundation:
   ...

### test_renormalized_coefficients_precision_physics(self, renormalized_coeffs)
**Описание:** Test renormalized coefficients precision.

Physical Meaning:
    Validates that renormalized coefficients maintain high precision
    for computational purposes.

Mathematical Foundation:
    Tests pr...

### test_renormalized_coefficients_validation_physics(self, renormalized_coeffs)
**Описание:** Test renormalized coefficients validation.

Physical Meaning:
    Validates that renormalized coefficients pass validation checks
    for physical consistency.

Mathematical Foundation:
    Tests vali...

### test_renormalized_coefficients_consistency_physics(self, renormalized_coeffs)
**Описание:** Test renormalized coefficients consistency.

Physical Meaning:
    Validates that renormalized coefficients are consistent with each other
    and with the 7D BVP theory.

Mathematical Foundation:
   ...

### test_renormalized_coefficients_physical_meaning_physics(self, renormalized_coeffs)
**Описание:** Test renormalized coefficients physical meaning.

Physical Meaning:
    Validates that renormalized coefficients have correct physical meaning
    in the 7D BVP theory.

Mathematical Foundation:
    T...

### test_renormalized_coefficients_renormalization_group_physics(self, renormalized_coeffs)
**Описание:** Test renormalized coefficients renormalization group flow.

Physical Meaning:
    Validates that renormalized coefficients follow correct renormalization group flow
    in the 7D BVP theory.

Mathemat...

### test_renormalized_coefficients_scale_dependence_physics(self, renormalized_coeffs)
**Описание:** Test renormalized coefficients scale dependence.

Physical Meaning:
    Validates that renormalized coefficients show correct scale dependence
    in the 7D BVP theory.

Mathematical Foundation:
    T...

### test_renormalized_coefficients_flow_equations_physics(self, renormalized_coeffs)
**Описание:** Test renormalized coefficients flow equations.

Physical Meaning:
    Validates that renormalized coefficients satisfy flow equations
    in the 7D BVP theory.

Mathematical Foundation:
    Tests flow...

## ./tests/unit/test_core/test_scale_separation_postulate_physics.py
Methods: 4

### domain_7d(self)
**Описание:** Create 7D domain for postulate testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb5b3d0>

### bvp_constants(self)
**Описание:** Create BVP constants for postulate testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb61d10>

### test_envelope(self, domain_7d)
**Описание:** Create test envelope for postulate validation.
**Декораторы:** <ast.Attribute object at 0x753e6fa86d50>

### test_scale_separation_postulate_physics(self, domain_7d, bvp_constants, test_envelope)
**Описание:** Test Scale Separation Postulate physics.

Physical Meaning:
    Validates that there is clear separation between the carrier
    scale (high-frequency) and envelope scale (low-frequency),
    ensuring...

## ./tests/unit/test_core/test_solvers_coverage.py
Methods: 38

### domain(self)
**Описание:** Create domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fab6750>

### parameters(self)
**Описание:** Create parameters for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fab5210>

### test_abstract_solver_creation(self, domain, parameters)
**Описание:** Test abstract solver creation.

### test_time_integrator_creation(self, domain)
**Описание:** Test time integrator creation.

### test_abstract_solver_methods(self, domain, parameters)
**Описание:** Test abstract solver methods.

### test_time_integrator_methods(self, domain)
**Описание:** Test time integrator methods.

### test_abstract_solver_validation(self, domain, parameters)
**Описание:** Test abstract solver validation.

### test_time_integrator_validation(self, domain)
**Описание:** Test time integrator validation.

### test_abstract_solver_7d_structure(self, domain, parameters)
**Описание:** Test abstract solver 7D structure handling.

### test_time_integrator_7d_structure(self, domain)
**Описание:** Test time integrator 7D structure handling.

### test_abstract_solver_numerical_stability(self, domain, parameters)
**Описание:** Test abstract solver numerical stability.

### test_time_integrator_numerical_stability(self, domain)
**Описание:** Test time integrator numerical stability.

### test_abstract_solver_precision(self, domain, parameters)
**Описание:** Test abstract solver precision.

### test_time_integrator_precision(self, domain)
**Описание:** Test time integrator precision.

### test_abstract_solver_error_handling(self, domain, parameters)
**Описание:** Test abstract solver error handling.

### test_time_integrator_error_handling(self, domain)
**Описание:** Test time integrator error handling.

### test_abstract_solver_edge_cases(self, domain, parameters)
**Описание:** Test abstract solver edge cases.

### test_time_integrator_edge_cases(self, domain)
**Описание:** Test time integrator edge cases.

### test_abstract_solver_repr(self, domain, parameters)
**Описание:** Test abstract solver string representation.

### test_time_integrator_repr(self, domain)
**Описание:** Test time integrator string representation.

### test_abstract_solver_config_handling(self, domain, parameters)
**Описание:** Test abstract solver configuration handling.

### test_time_integrator_config_handling(self, domain)
**Описание:** Test time integrator configuration handling.

### test_abstract_solver_performance(self, domain, parameters)
**Описание:** Test abstract solver performance.

### test_time_integrator_performance(self, domain)
**Описание:** Test time integrator performance.

### test_abstract_solver_memory_usage(self, domain, parameters)
**Описание:** Test abstract solver memory usage.

### test_time_integrator_memory_usage(self, domain)
**Описание:** Test time integrator memory usage.

### test_abstract_solver_statistics(self, domain, parameters)
**Описание:** Test abstract solver statistics.

### test_time_integrator_statistics(self, domain)
**Описание:** Test time integrator statistics.

### test_abstract_solver_optimization(self, domain, parameters)
**Описание:** Test abstract solver optimization.

### test_time_integrator_optimization(self, domain)
**Описание:** Test time integrator optimization.

### test_abstract_solver_parallel(self, domain, parameters)
**Описание:** Test abstract solver parallel processing.

### test_time_integrator_parallel(self, domain)
**Описание:** Test time integrator parallel processing.

### test_abstract_solver_vectorized(self, domain, parameters)
**Описание:** Test abstract solver vectorization.

### test_time_integrator_vectorized(self, domain)
**Описание:** Test time integrator vectorization.

### test_abstract_solver_cleanup(self, domain, parameters)
**Описание:** Test abstract solver cleanup.

### test_time_integrator_cleanup(self, domain)
**Описание:** Test time integrator cleanup.

### test_abstract_solver_reset(self, domain, parameters)
**Описание:** Test abstract solver reset.

### test_time_integrator_reset(self, domain)
**Описание:** Test time integrator reset.

## ./tests/unit/test_core/test_sources_coverage.py
Methods: 23

### domain(self)
**Описание:** Create domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb06290>

### test_source_creation(self, domain)
**Описание:** Test source creation.

### test_bvp_source_creation(self, domain)
**Описание:** Test BVP source creation.

### test_source_methods(self, domain)
**Описание:** Test source methods.

### test_bvp_source_methods(self, domain)
**Описание:** Test BVP source methods.

### test_source_validation(self, domain)
**Описание:** Test source validation.

### test_bvp_source_validation(self, domain)
**Описание:** Test BVP source validation.

### test_source_7d_structure(self, domain)
**Описание:** Test source 7D structure preservation.

### test_bvp_source_7d_structure(self, domain)
**Описание:** Test BVP source 7D structure preservation.

### test_source_numerical_stability(self, domain)
**Описание:** Test source numerical stability.

### test_bvp_source_numerical_stability(self, domain)
**Описание:** Test BVP source numerical stability.

### test_source_precision(self, domain)
**Описание:** Test source precision.

### test_bvp_source_precision(self, domain)
**Описание:** Test BVP source precision.

### test_source_error_handling(self, domain)
**Описание:** Test source error handling.

### test_bvp_source_error_handling(self, domain)
**Описание:** Test BVP source error handling.

### test_source_edge_cases(self, domain)
**Описание:** Test source edge cases.

### test_bvp_source_edge_cases(self, domain)
**Описание:** Test BVP source edge cases.

### test_source_repr(self, domain)
**Описание:** Test source string representation.

### test_bvp_source_repr(self, domain)
**Описание:** Test BVP source string representation.

### test_source_config_handling(self, domain)
**Описание:** Test source configuration handling.

### test_bvp_source_config_handling(self, domain)
**Описание:** Test BVP source configuration handling.

### test_source_performance(self, domain)
**Описание:** Test source performance.

### test_bvp_source_performance(self, domain)
**Описание:** Test BVP source performance.

## ./tests/unit/test_core/test_spectral_boundary_conditions_physics.py
Methods: 18

### domain_7d(self)
**Описание:** Create 7D domain for spectral testing.
**Декораторы:** <ast.Attribute object at 0x753e6f90b7d0>

### fft_backend(self, domain_7d)
**Описание:** Create FFT backend for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f90b750>

### spectral_ops(self, fft_backend)
**Описание:** Create spectral operations for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f90be50>

### test_spectral_boundary_conditions_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral boundary conditions physical consistency.

Physical Meaning:
    Validates that spectral boundary conditions are handled
    correctly in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematica...

### test_spectral_boundary_conditions_energy_conservation_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral boundary conditions energy conservation.

Physical Meaning:
    Validates that spectral boundary conditions conserve energy
    in the spectral domain.

Mathematical Foundation:
    Test...

### test_spectral_boundary_conditions_7d_structure_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral boundary conditions 7D structure preservation.

Physical Meaning:
    Validates that spectral boundary conditions preserve the 7D structure
    of space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathe...

### test_spectral_boundary_conditions_numerical_stability_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral boundary conditions numerical stability.

Physical Meaning:
    Validates that spectral boundary conditions are numerically stable
    for extreme field values.

Mathematical Foundation:...

### test_spectral_boundary_conditions_precision_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral boundary conditions precision.

Physical Meaning:
    Validates that spectral boundary conditions maintain high precision
    for known analytical functions.

Mathematical Foundation:
  ...

### test_spectral_boundary_conditions_phase_structure_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral boundary conditions phase structure preservation.

Physical Meaning:
    Validates that spectral boundary conditions preserve phase structure
    in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

...

### test_spectral_boundary_conditions_periodic_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral periodic boundary conditions.

Physical Meaning:
    Validates that spectral periodic boundary conditions are handled
    correctly in 7D space-time.

Mathematical Foundation:
    Tests ...

### test_spectral_boundary_conditions_dirichlet_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral Dirichlet boundary conditions.

Physical Meaning:
    Validates that spectral Dirichlet boundary conditions are handled
    correctly in 7D space-time.

Mathematical Foundation:
    Test...

### test_spectral_boundary_conditions_neumann_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral Neumann boundary conditions.

Physical Meaning:
    Validates that spectral Neumann boundary conditions are handled
    correctly in 7D space-time.

Mathematical Foundation:
    Tests Ne...

### test_spectral_boundary_conditions_mixed_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral mixed boundary conditions.

Physical Meaning:
    Validates that spectral mixed boundary conditions are handled
    correctly in 7D space-time.

Mathematical Foundation:
    Tests mixed ...

### _create_boundary_test_field(self, domain)
**Описание:** Create test field with boundary conditions.

### _create_periodic_boundary_test_field(self, domain)
**Описание:** Create test field with periodic boundary conditions.

### _create_dirichlet_boundary_test_field(self, domain)
**Описание:** Create test field with Dirichlet boundary conditions.

### _create_neumann_boundary_test_field(self, domain)
**Описание:** Create test field with Neumann boundary conditions.

### _create_mixed_boundary_test_field(self, domain)
**Описание:** Create test field with mixed boundary conditions.

## ./tests/unit/test_core/test_spectral_convergence_physics.py
Methods: 17

### domain_7d(self)
**Описание:** Create 7D domain for spectral testing.
**Декораторы:** <ast.Attribute object at 0x753e6facf610>

### fft_backend(self, domain_7d)
**Описание:** Create FFT backend for testing.
**Декораторы:** <ast.Attribute object at 0x753e6facf290>

### spectral_ops(self, fft_backend)
**Описание:** Create spectral operations for testing.
**Декораторы:** <ast.Attribute object at 0x753e6facef90>

### test_spectral_convergence_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral convergence physical consistency.

Physical Meaning:
    Validates that spectral methods converge correctly
    in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematical Foundation:
    Tests...

### test_spectral_convergence_energy_conservation_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral convergence energy conservation.

Physical Meaning:
    Validates that spectral convergence conserves energy
    in the spectral domain.

Mathematical Foundation:
    Tests energy conser...

### test_spectral_convergence_7d_structure_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral convergence 7D structure preservation.

Physical Meaning:
    Validates that spectral convergence preserves the 7D structure
    of space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematical Foundat...

### test_spectral_convergence_numerical_stability_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral convergence numerical stability.

Physical Meaning:
    Validates that spectral convergence is numerically stable
    for extreme field values.

Mathematical Foundation:
    Tests numeri...

### test_spectral_convergence_precision_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral convergence precision.

Physical Meaning:
    Validates that spectral convergence maintains high precision
    for known analytical functions.

Mathematical Foundation:
    Tests precisi...

### test_spectral_convergence_phase_structure_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral convergence phase structure preservation.

Physical Meaning:
    Validates that spectral convergence preserves phase structure
    in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematical Fo...

### test_spectral_convergence_resolution_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral convergence with different resolutions.

Physical Meaning:
    Validates that spectral convergence improves with higher resolution
    in 7D space-time.

Mathematical Foundation:
    Tes...

### test_spectral_convergence_accuracy_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral convergence accuracy.

Physical Meaning:
    Validates that spectral convergence maintains high accuracy
    for known analytical functions.

Mathematical Foundation:
    Tests accuracy ...

### test_spectral_convergence_stability_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral convergence stability.

Physical Meaning:
    Validates that spectral convergence is stable
    for different field configurations.

Mathematical Foundation:
    Tests stability of spect...

### test_spectral_convergence_efficiency_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral convergence efficiency.

Physical Meaning:
    Validates that spectral convergence is efficient
    for different field sizes.

Mathematical Foundation:
    Tests efficiency of spectral ...

### _create_test_field(self, domain)
**Описание:** Create test field for spectral convergence testing.

### _create_sinusoidal_field(self, domain)
**Описание:** Create sinusoidal test field.

### _create_gaussian_field(self, domain)
**Описание:** Create Gaussian test field.

### _create_random_field(self, domain)
**Описание:** Create random test field.

## ./tests/unit/test_core/test_spectral_derivatives.py
Methods: 15

### domain(self)
**Описание:** Create domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fada950>

### fft_backend(self, domain)
**Описание:** Create FFT backend for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fada910>

### spectral_derivs(self, fft_backend)
**Описание:** Create spectral derivatives for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fadbf50>

### test_spectral_derivs_initialization(self, spectral_derivs, fft_backend)
**Описание:** Test spectral derivatives initialization.

### test_spectral_derivs_first_derivative(self, spectral_derivs)
**Описание:** Test first-order derivative computation.

### test_spectral_derivs_second_derivative(self, spectral_derivs)
**Описание:** Test second-order derivative computation.

### test_spectral_derivs_nth_derivative(self, spectral_derivs)
**Описание:** Test nth-order derivative computation.

### test_spectral_derivs_mixed_derivative(self, spectral_derivs)
**Описание:** Test mixed derivative computation.

### test_spectral_derivs_validation(self, spectral_derivs)
**Описание:** Test input validation.

### test_spectral_derivs_energy_conservation(self, spectral_derivs)
**Описание:** Test energy conservation in spectral derivatives.

### test_spectral_derivs_7d_structure(self, spectral_derivs)
**Описание:** Test 7D structure preservation in spectral derivatives.

### test_spectral_derivs_numerical_stability(self, spectral_derivs)
**Описание:** Test numerical stability of spectral derivatives.

### test_spectral_derivs_precision(self, spectral_derivs)
**Описание:** Test precision of spectral derivatives.

### test_spectral_derivs_axis_handling(self, spectral_derivs)
**Описание:** Test derivative computation along different axes.

### test_spectral_derivs_order_handling(self, spectral_derivs)
**Описание:** Test derivative computation with different orders.

## ./tests/unit/test_core/test_spectral_derivatives_physics.py
Methods: 13

### domain_7d(self)
**Описание:** Create 7D domain for spectral testing.
**Декораторы:** <ast.Attribute object at 0x753e6fad2790>

### fft_backend(self, domain_7d)
**Описание:** Create FFT backend for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fad3a90>

### spectral_derivs(self, fft_backend)
**Описание:** Create spectral derivatives for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fad27d0>

### test_spectral_derivatives_physics(self, domain_7d, spectral_derivs)
**Описание:** Test spectral derivatives physical consistency.

Physical Meaning:
    Validates that spectral derivatives correctly compute
    derivatives in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematical Founda...

### test_spectral_derivatives_energy_conservation_physics(self, domain_7d, spectral_derivs)
**Описание:** Test spectral derivatives energy conservation.

Physical Meaning:
    Validates that spectral derivatives conserve energy
    in the spectral domain.

Mathematical Foundation:
    Tests energy conserv...

### test_spectral_derivatives_7d_structure_physics(self, domain_7d, spectral_derivs)
**Описание:** Test spectral derivatives 7D structure preservation.

Physical Meaning:
    Validates that spectral derivatives preserve the 7D structure
    of space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematical Foundati...

### test_spectral_derivatives_numerical_stability_physics(self, domain_7d, spectral_derivs)
**Описание:** Test spectral derivatives numerical stability.

Physical Meaning:
    Validates that spectral derivatives are numerically stable
    for extreme field values.

Mathematical Foundation:
    Tests numer...

### test_spectral_derivatives_precision_physics(self, domain_7d, spectral_derivs)
**Описание:** Test spectral derivatives precision.

Physical Meaning:
    Validates that spectral derivatives maintain high precision
    for known analytical functions.

Mathematical Foundation:
    Tests precisio...

### test_spectral_derivatives_boundary_conditions_physics(self, domain_7d, spectral_derivs)
**Описание:** Test spectral derivatives boundary condition handling.

Physical Meaning:
    Validates that spectral derivatives handle boundary conditions
    correctly in 7D space-time.

Mathematical Foundation:
 ...

### test_spectral_derivatives_phase_structure_physics(self, domain_7d, spectral_derivs)
**Описание:** Test spectral derivatives phase structure preservation.

Physical Meaning:
    Validates that spectral derivatives preserve phase structure
    in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematical Fou...

### test_spectral_derivatives_mixed_derivatives_physics(self, domain_7d, spectral_derivs)
**Описание:** Test spectral mixed derivatives.

Physical Meaning:
    Validates that spectral mixed derivatives correctly compute
    mixed derivatives in 7D space-time.

Mathematical Foundation:
    Tests mixed sp...

### test_spectral_derivatives_higher_order_physics(self, domain_7d, spectral_derivs)
**Описание:** Test spectral higher-order derivatives.

Physical Meaning:
    Validates that spectral higher-order derivatives correctly compute
    higher-order derivatives in 7D space-time.

Mathematical Foundatio...

### _create_test_field(self, domain)
**Описание:** Create test field for spectral derivatives testing.

## ./tests/unit/test_core/test_spectral_energy_spectrum_physics.py
Methods: 16

### domain_7d(self)
**Описание:** Create 7D domain for spectral testing.
**Декораторы:** <ast.Attribute object at 0x753e6f94e890>

### fft_backend(self, domain_7d)
**Описание:** Create FFT backend for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f94d310>

### spectral_ops(self, fft_backend)
**Описание:** Create spectral operations for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f94c1d0>

### test_spectral_energy_spectrum_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral energy spectrum physical consistency.

Physical Meaning:
    Validates that spectral energy spectrum is computed correctly
    in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematical Founda...

### test_spectral_energy_spectrum_energy_conservation_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral energy spectrum energy conservation.

Physical Meaning:
    Validates that spectral energy spectrum conserves energy
    in the spectral domain.

Mathematical Foundation:
    Tests energ...

### test_spectral_energy_spectrum_7d_structure_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral energy spectrum 7D structure preservation.

Physical Meaning:
    Validates that spectral energy spectrum preserves the 7D structure
    of space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematical...

### test_spectral_energy_spectrum_numerical_stability_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral energy spectrum numerical stability.

Physical Meaning:
    Validates that spectral energy spectrum is numerically stable
    for extreme field values.

Mathematical Foundation:
    Test...

### test_spectral_energy_spectrum_precision_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral energy spectrum precision.

Physical Meaning:
    Validates that spectral energy spectrum maintains high precision
    for known analytical functions.

Mathematical Foundation:
    Tests...

### test_spectral_energy_spectrum_phase_structure_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral energy spectrum phase structure preservation.

Physical Meaning:
    Validates that spectral energy spectrum preserves phase structure
    in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathema...

### test_spectral_energy_spectrum_frequency_dependence_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral energy spectrum frequency dependence.

Physical Meaning:
    Validates that spectral energy spectrum shows correct frequency dependence
    in 7D space-time.

Mathematical Foundation:
  ...

### test_spectral_energy_spectrum_resolution_dependence_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral energy spectrum resolution dependence.

Physical Meaning:
    Validates that spectral energy spectrum shows correct resolution dependence
    in 7D space-time.

Mathematical Foundation:
...

### test_spectral_energy_spectrum_energy_distribution_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral energy spectrum energy distribution.

Physical Meaning:
    Validates that spectral energy spectrum shows correct energy distribution
    in 7D space-time.

Mathematical Foundation:
    ...

### test_spectral_energy_spectrum_spectral_properties_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral energy spectrum spectral properties.

Physical Meaning:
    Validates that spectral energy spectrum shows correct spectral properties
    in 7D space-time.

Mathematical Foundation:
    ...

### _create_test_field(self, domain)
**Описание:** Create test field for spectral energy spectrum testing.

### _create_frequency_test_field(self, domain)
**Описание:** Create test field with specific frequency content.

### _compute_energy_spectrum(self, field, spectral_ops)
**Описание:** Compute spectral energy spectrum.

## ./tests/unit/test_core/test_spectral_filtering.py
Methods: 16

### domain(self)
**Описание:** Create domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fccfbd0>

### fft_backend(self, domain)
**Описание:** Create FFT backend for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fccf550>

### spectral_filtering(self, fft_backend)
**Описание:** Create spectral filtering for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa20450>

### test_spectral_filtering_initialization(self, spectral_filtering, fft_backend)
**Описание:** Test spectral filtering initialization.

### test_spectral_filtering_low_pass(self, spectral_filtering)
**Описание:** Test low-pass filtering.

### test_spectral_filtering_high_pass(self, spectral_filtering)
**Описание:** Test high-pass filtering.

### test_spectral_filtering_band_pass(self, spectral_filtering)
**Описание:** Test band-pass filtering.

### test_spectral_filtering_gaussian(self, spectral_filtering)
**Описание:** Test Gaussian filtering.

### test_spectral_filtering_validation(self, spectral_filtering)
**Описание:** Test input validation.

### test_spectral_filtering_energy_conservation(self, spectral_filtering)
**Описание:** Test energy conservation in spectral filtering.

### test_spectral_filtering_7d_structure(self, spectral_filtering)
**Описание:** Test 7D structure preservation in spectral filtering.

### test_spectral_filtering_numerical_stability(self, spectral_filtering)
**Описание:** Test numerical stability of spectral filtering.

### test_spectral_filtering_precision(self, spectral_filtering)
**Описание:** Test precision of spectral filtering.

### test_spectral_filtering_cutoff_effects(self, spectral_filtering)
**Описание:** Test effects of different cutoff frequencies.

### test_spectral_filtering_band_pass_validation(self, spectral_filtering)
**Описание:** Test band-pass filter validation.

### test_spectral_filtering_gaussian_validation(self, spectral_filtering)
**Описание:** Test Gaussian filter validation.

## ./tests/unit/test_core/test_spectral_laplacian_physics.py
Methods: 17

### domain_7d(self)
**Описание:** Create 7D domain for spectral testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa8cfd0>

### fft_backend(self, domain_7d)
**Описание:** Create FFT backend for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa8e850>

### spectral_ops(self, fft_backend)
**Описание:** Create spectral operations for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa8ee90>

### fractional_laplacian(self, domain_7d)
**Описание:** Create fractional Laplacian for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa8d1d0>

### test_spectral_laplacian_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral Laplacian physical consistency.

Physical Meaning:
    Validates that spectral Laplacian correctly computes
    Laplacian in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematical Foundation:...

### test_spectral_laplacian_energy_conservation_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral Laplacian energy conservation.

Physical Meaning:
    Validates that spectral Laplacian conserves energy
    in the spectral domain.

Mathematical Foundation:
    Tests energy conservati...

### test_spectral_laplacian_7d_structure_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral Laplacian 7D structure preservation.

Physical Meaning:
    Validates that spectral Laplacian preserves the 7D structure
    of space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematical Foundation:...

### test_spectral_laplacian_numerical_stability_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral Laplacian numerical stability.

Physical Meaning:
    Validates that spectral Laplacian is numerically stable
    for extreme field values.

Mathematical Foundation:
    Tests numerical ...

### test_spectral_laplacian_precision_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral Laplacian precision.

Physical Meaning:
    Validates that spectral Laplacian maintains high precision
    for known analytical functions.

Mathematical Foundation:
    Tests precision o...

### test_spectral_laplacian_boundary_conditions_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral Laplacian boundary condition handling.

Physical Meaning:
    Validates that spectral Laplacian handles boundary conditions
    correctly in 7D space-time.

Mathematical Foundation:
    ...

### test_spectral_laplacian_phase_structure_physics(self, domain_7d, spectral_ops)
**Описание:** Test spectral Laplacian phase structure preservation.

Physical Meaning:
    Validates that spectral Laplacian preserves phase structure
    in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematical Founda...

### test_fractional_laplacian_physics(self, domain_7d, fractional_laplacian)
**Описание:** Test fractional Laplacian physical consistency.

Physical Meaning:
    Validates that fractional Laplacian correctly computes
    fractional Laplacian in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathemati...

### test_fractional_laplacian_energy_conservation_physics(self, domain_7d, fractional_laplacian)
**Описание:** Test fractional Laplacian energy conservation.

Physical Meaning:
    Validates that fractional Laplacian conserves energy
    in the spectral domain.

Mathematical Foundation:
    Tests energy conser...

### test_fractional_laplacian_7d_structure_physics(self, domain_7d, fractional_laplacian)
**Описание:** Test fractional Laplacian 7D structure preservation.

Physical Meaning:
    Validates that fractional Laplacian preserves the 7D structure
    of space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematical Foundat...

### test_fractional_laplacian_numerical_stability_physics(self, domain_7d, fractional_laplacian)
**Описание:** Test fractional Laplacian numerical stability.

Physical Meaning:
    Validates that fractional Laplacian is numerically stable
    for extreme field values.

Mathematical Foundation:
    Tests numeri...

### test_fractional_laplacian_precision_physics(self, domain_7d, fractional_laplacian)
**Описание:** Test fractional Laplacian precision.

Physical Meaning:
    Validates that fractional Laplacian maintains high precision
    for known analytical functions.

Mathematical Foundation:
    Tests precisi...

### _create_test_field(self, domain)
**Описание:** Create test field for spectral Laplacian testing.

## ./tests/unit/test_core/test_spectral_operations.py
Methods: 14

### domain(self)
**Описание:** Create domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa8dad0>

### fft_backend(self, domain)
**Описание:** Create FFT backend for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa8d110>

### spectral_ops(self, fft_backend)
**Описание:** Create spectral operations for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa8d050>

### test_spectral_ops_initialization(self, spectral_ops, fft_backend)
**Описание:** Test spectral operations initialization.

### test_spectral_ops_compute_derivative(self, spectral_ops)
**Описание:** Test spectral derivative computation.

### test_spectral_ops_compute_laplacian(self, spectral_ops)
**Описание:** Test spectral Laplacian computation.

### test_spectral_ops_compute_gradient(self, spectral_ops)
**Описание:** Test spectral gradient computation.

### test_spectral_ops_compute_divergence(self, spectral_ops)
**Описание:** Test spectral divergence computation.

### test_spectral_ops_compute_curl(self, spectral_ops)
**Описание:** Test spectral curl computation.

### test_spectral_ops_validation(self, spectral_ops)
**Описание:** Test input validation.

### test_spectral_ops_energy_conservation(self, spectral_ops)
**Описание:** Test energy conservation in spectral operations.

### test_spectral_ops_7d_structure(self, spectral_ops)
**Описание:** Test 7D structure preservation in spectral operations.

### test_spectral_ops_numerical_stability(self, spectral_ops)
**Описание:** Test numerical stability of spectral operations.

### test_spectral_ops_precision(self, spectral_ops)
**Описание:** Test precision of spectral operations.

## ./tests/unit/test_core/test_step_resonator_boundary.py
Methods: 3

### test_step_resonator_basic_1d()
**Описание:** Нет докстринга

### test_step_resonator_7d_axes_subset()
**Описание:** Нет докстринга

### test_step_resonator_frequency_dependent_rt()
**Описание:** Нет докстринга

## ./tests/unit/test_core/test_tail_resonatorness_postulate_physics.py
Methods: 4

### domain_7d(self)
**Описание:** Create 7D domain for postulate testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa8e390>

### bvp_constants(self)
**Описание:** Create BVP constants for postulate testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa8d650>

### test_envelope(self, domain_7d)
**Описание:** Create test envelope for postulate validation.
**Декораторы:** <ast.Attribute object at 0x753e6fb48ad0>

### test_tail_resonatorness_postulate_physics(self, domain_7d, bvp_constants, test_envelope)
**Описание:** Test Tail Resonatorness Postulate physics.

Physical Meaning:
    Validates that the field tail exhibits resonator properties
    with proper resonance frequencies and quality factors.

Mathematical F...

## ./tests/unit/test_core/test_transition_zone_postulate_physics.py
Methods: 4

### domain_7d(self)
**Описание:** Create 7D domain for postulate testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa4e790>

### bvp_constants(self)
**Описание:** Create BVP constants for postulate testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb74a10>

### test_envelope(self, domain_7d)
**Описание:** Create test envelope for postulate validation.
**Декораторы:** <ast.Attribute object at 0x753e6fa86f90>

### test_transition_zone_postulate_physics(self, domain_7d, bvp_constants, test_envelope)
**Описание:** Test Transition Zone Postulate physics.

Physical Meaning:
    Validates that the transition zone between different field
    regions exhibits proper nonlinear interface behavior.

Mathematical Founda...

## ./tests/unit/test_core/test_u1_phase_structure_postulate_physics.py
Methods: 4

### domain_7d(self)
**Описание:** Create 7D domain for postulate testing.
**Декораторы:** <ast.Attribute object at 0x753e6fba7950>

### bvp_constants(self)
**Описание:** Create BVP constants for postulate testing.
**Декораторы:** <ast.Attribute object at 0x753e6f9132d0>

### test_envelope(self, domain_7d)
**Описание:** Create test envelope for postulate validation.
**Декораторы:** <ast.Attribute object at 0x753e6faf7810>

### test_u1_phase_structure_postulate_physics(self, domain_7d, bvp_constants, test_envelope)
**Описание:** Test U(1)³ Phase Structure Postulate physics.

Physical Meaning:
    Validates that the field has proper U(1)³ phase structure
    with correct phase coherence and topological properties.

Mathematica...

## ./tests/unit/test_core/time_integrators/test_advanced_integrators.py
Methods: 12

### domain_7d(self)
**Описание:** Create 7D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f96f550>

### parameters_basic(self)
**Описание:** Basic parameters for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f96efd0>

### test_integrator_stability(self, domain_7d, parameters_basic)
**Описание:** Test integrator stability with different parameter values.

### test_integrator_accuracy(self, domain_7d, parameters_basic)
**Описание:** Test integrator accuracy with known solutions.

### test_memory_kernel_advanced(self, domain_7d, parameters_basic)
**Описание:** Test advanced memory kernel functionality.

### test_quench_detector_advanced(self, domain_7d, parameters_basic)
**Описание:** Test advanced quench detector functionality.

### test_integrator_convergence(self, domain_7d, parameters_basic)
**Описание:** Test integrator convergence with increasing resolution.

### test_integrator_boundary_conditions(self, domain_7d, parameters_basic)
**Описание:** Test integrator behavior with boundary conditions.

### test_integrator_extreme_values(self, domain_7d, parameters_basic)
**Описание:** Test integrator behavior with extreme values.

### test_integrator_consistency_across_runs(self, domain_7d, parameters_basic)
**Описание:** Test that integrators produce consistent results across runs.

### test_integrator_memory_usage(self, domain_7d, parameters_basic)
**Описание:** Test integrator memory usage.

### test_integrator_error_recovery(self, domain_7d, parameters_basic)
**Описание:** Test integrator error recovery.

## ./tests/unit/test_core/time_integrators/test_basic_integrators.py
Methods: 14

### domain_7d(self)
**Описание:** Create 7D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f918a90>

### parameters_basic(self)
**Описание:** Basic parameters for testing.
**Декораторы:** <ast.Attribute object at 0x753e6f91b8d0>

### test_envelope_integrator_creation(self, domain_7d, parameters_basic)
**Описание:** Test envelope integrator creation.

### test_crank_nicolson_integrator_creation(self, domain_7d, parameters_basic)
**Описание:** Test Crank-Nicolson integrator creation.

### test_memory_kernel_creation(self, domain_7d, parameters_basic)
**Описание:** Test memory kernel creation.

### test_quench_detector_creation(self, domain_7d, parameters_basic)
**Описание:** Test quench detector creation.

### test_integrator_parameter_validation(self, domain_7d)
**Описание:** Test parameter validation in integrators.

### test_integrator_domain_validation(self, parameters_basic)
**Описание:** Test domain validation in integrators.

### test_integrator_basic_functionality(self, domain_7d, parameters_basic)
**Описание:** Test basic functionality of integrators.

### test_memory_kernel_functionality(self, domain_7d, parameters_basic)
**Описание:** Test memory kernel functionality.

### test_quench_detector_functionality(self, domain_7d, parameters_basic)
**Описание:** Test quench detector functionality.

### test_integrator_consistency(self, domain_7d, parameters_basic)
**Описание:** Test consistency between different integrators.

### test_integrator_error_handling(self, domain_7d, parameters_basic)
**Описание:** Test error handling in integrators.

### test_integrator_performance(self, domain_7d, parameters_basic)
**Описание:** Test basic performance of integrators.

## ./tests/unit/test_level_a/test_A01_minimal.py
Methods: 9

### setup_method(self)
**Описание:** Setup test parameters.

### test_domain_creation(self)
**Описание:** Test that domain is created correctly.

### test_parameters_creation(self)
**Описание:** Test that parameters are created correctly.

### test_fractional_laplacian_creation(self)
**Описание:** Test that fractional Laplacian is created correctly.

### test_spectral_coefficients(self)
**Описание:** Test that spectral coefficients can be computed.

### test_simple_field_application(self)
**Описание:** Test that fractional Laplacian can be applied to a simple field.

### test_plane_wave_creation(self)
**Описание:** Test that plane wave can be created.

### test_fft_operations(self)
**Описание:** Test basic FFT operations.

### test_spectral_solution(self)
**Описание:** Test basic spectral solution approach.

## ./tests/unit/test_level_a/test_A01_plane_wave.py
Methods: 13

### setup_method(self)
**Описание:** Setup test parameters.

### create_plane_wave_source(self, k_mode)
**Описание:** Create plane wave source s(x) = exp(i k·x) for 7D domain.

Physical Meaning:
    Creates a monochromatic source with wave vector k_mode
    for testing the spectral solution in 7D space-time M₇ = ℝ³ₓ ...

### compute_analytical_solution(self, k_mode)
**Описание:** Compute analytical solution a(x) = s(x) / D(k) for 7D domain.

Physical Meaning:
    Computes the analytical solution for the plane wave
    using the spectral formula in 7D space-time.

Args:
    k_m...

### create_plane_wave_source_for_domain(self, k_mode, domain)
**Описание:** Create plane wave source for a specific domain.

Args:
    k_mode: Wave vector [kx, ky, kz]
    domain: Domain object
    
Returns:
    Source field for the given domain

### compute_analytical_solution_for_domain(self, k_mode, domain)
**Описание:** Compute analytical solution for a specific domain.

Args:
    k_mode: Wave vector [kx, ky, kz]
    domain: Domain object
    
Returns:
    Analytical solution for the given domain

### test_plane_wave_single_mode(self)
**Описание:** Test plane wave solution for single mode.

Physical Meaning:
    Tests the basic functionality of the FFT solver
    for a single plane wave mode.

### test_plane_wave_multiple_modes(self)
**Описание:** Test plane wave solution for multiple modes.

Physical Meaning:
    Tests the solver for different wave vectors to ensure
    correct spectral handling.

### test_anisotropy_check(self)
**Описание:** Test anisotropy for modes with same |k|.

Physical Meaning:
    Tests that modes with the same wave vector magnitude
    produce solutions with the same amplitude, ensuring
    isotropy of the operato...

### test_grid_convergence(self)
**Описание:** Test convergence with grid refinement.

Physical Meaning:
    Tests that the solution converges as the grid is refined,
    ensuring numerical accuracy.

### test_fractional_laplacian_operator(self)
**Описание:** Test fractional Laplacian operator directly.

Physical Meaning:
    Tests the fractional Laplacian operator implementation
    to ensure correct spectral coefficients.

### test_spectral_coefficients(self)
**Описание:** Test spectral coefficients computation.

Physical Meaning:
    Tests that the spectral coefficients are computed correctly
    for the fractional Laplacian operator.

### test_solver_validation(self)
**Описание:** Test solver validation functionality.

Physical Meaning:
    Tests the built-in validation methods to ensure
    solution quality.

### test_solver_info(self)
**Описание:** Test solver information retrieval.

Physical Meaning:
    Tests that solver information is correctly provided
    for debugging and monitoring.

## ./tests/unit/test_level_a/test_A01_simple_plane_wave.py
Methods: 8

### setup_method(self)
**Описание:** Setup test parameters.

### create_plane_wave_source(self, k_mode)
**Описание:** Create plane wave source s(x) = exp(i k·x).

Physical Meaning:
    Creates a monochromatic source with wave vector k_mode
    for testing the spectral solution.

### compute_analytical_solution(self, k_mode)
**Описание:** Compute analytical solution a(x) = s(x) / D(k).

Physical Meaning:
    Computes the analytical solution for the plane wave
    using the spectral formula.

### test_plane_wave_basic(self)
**Описание:** Test basic plane wave solution.

Physical Meaning:
    Tests the basic functionality of the fractional Laplacian
    for a simple plane wave mode.

### test_spectral_coefficients(self)
**Описание:** Test spectral coefficients computation.

Physical Meaning:
    Tests that the spectral coefficients are computed correctly
    for the fractional Laplacian operator.

### test_simple_fft_solution(self)
**Описание:** Test simple FFT-based solution.

Physical Meaning:
    Tests a simple FFT-based solution using basic NumPy operations
    to validate the spectral approach.

### test_multiple_modes(self)
**Описание:** Test multiple wave modes.

Physical Meaning:
    Tests that the fractional Laplacian works correctly
    for different wave vectors.

### test_operator_properties(self)
**Описание:** Test basic operator properties.

Physical Meaning:
    Tests that the fractional Laplacian operator has the
    expected mathematical properties.

## ./tests/unit/test_level_a/test_A02_multi_plane.py
Methods: 11

### setup_method(self)
**Описание:** Setup test parameters.

### generate_random_modes(self, n_modes)
**Описание:** Generate random wave vectors within Nyquist limit.

Physical Meaning:
    Generates random wave vectors for testing multi-frequency
    sources while avoiding aliasing effects.

Args:
    n_modes: Num...

### create_multi_frequency_source(self, modes, amplitudes)
**Описание:** Create multi-frequency source s(x) = Σ c_j e^(i k_j·x).

Physical Meaning:
    Creates a multi-frequency source by superposition of
    plane waves with different wave vectors and amplitudes.

Args:
 ...

### compute_analytical_solution(self, modes, amplitudes)
**Описание:** Compute analytical solution for multi-frequency source.

Physical Meaning:
    Computes the analytical solution using superposition
    principle: a(x) = Σ c_j e^(i k_j·x) / D(k_j).

Args:
    modes: ...

### test_superposition_principle(self)
**Описание:** Test superposition principle for multi-frequency source.

Physical Meaning:
    Tests that the solution for a multi-frequency source
    equals the sum of solutions for individual modes.

### test_individual_mode_solutions(self)
**Описание:** Test individual mode solutions and their superposition.

Physical Meaning:
    Tests that the solution for a multi-frequency source
    equals the sum of individual mode solutions.

### test_aliasing_detection(self)
**Описание:** Test for aliasing effects in multi-frequency sources.

Physical Meaning:
    Tests that high-frequency modes do not create aliasing
    effects that would contaminate the solution.

### test_spectral_analysis(self)
**Описание:** Test spectral analysis of multi-frequency solution.

Physical Meaning:
    Tests that the spectral content of the solution
    matches the expected spectral coefficients.

### test_frequency_response(self)
**Описание:** Test frequency response of the solver.

Physical Meaning:
    Tests that the solver correctly applies the spectral
    operator D(k) = μ|k|^(2β) + λ to different frequencies.

### test_phase_preservation(self)
**Описание:** Test phase preservation in multi-frequency solution.

Physical Meaning:
    Tests that the phase relationships between different
    frequency components are preserved.

### test_energy_conservation(self)
**Описание:** Test energy conservation in multi-frequency solution.

Physical Meaning:
    Tests that the total energy in the solution is
    consistent with the spectral operator.

## ./tests/unit/test_level_a/test_A02_simple_multi_frequency.py
Methods: 8

### setup_method(self)
**Описание:** Setup test parameters.

### create_single_mode_source(self, k_mode)
**Описание:** Create single mode source.

### create_multi_frequency_source(self, modes, amplitudes)
**Описание:** Create multi-frequency source.

### test_superposition_principle(self)
**Описание:** Test superposition principle for multi-frequency source.

### test_frequency_response(self)
**Описание:** Test frequency response of the operator.

### test_spectral_analysis(self)
**Описание:** Test spectral analysis of multi-frequency solution.

### test_energy_conservation(self)
**Описание:** Test energy conservation in multi-frequency solution.

### test_phase_preservation(self)
**Описание:** Test phase preservation in multi-frequency solution.

## ./tests/unit/test_level_a/test_A03_simple_zero_mode.py
Methods: 7

### setup_method(self)
**Описание:** Setup test parameters.

### test_zero_mode_detection(self)
**Описание:** Test detection of zero mode condition.

### test_zero_mode_handling(self)
**Описание:** Test handling of zero mode when λ=0.

### test_zero_dc_source(self)
**Описание:** Test solution with zero DC component source.

### test_plane_wave_solution(self)
**Описание:** Test plane wave solution when λ=0.

### test_spectral_coefficients_zero_mode(self)
**Описание:** Test spectral coefficients at k=0 when λ=0.

### test_operator_singularity(self)
**Описание:** Test operator singularity at k=0.

## ./tests/unit/test_level_a/test_A03_zero_mode.py
Methods: 12

### setup_method(self)
**Описание:** Setup test parameters.

### create_constant_source(self)
**Описание:** Create constant source s(x) = 1.

Physical Meaning:
    Creates a constant source that has non-zero DC component,
    which should cause problems when λ=0.

Returns:
    Constant source field

### create_zero_dc_source(self)
**Описание:** Create source with zero DC component.

Physical Meaning:
    Creates a source that satisfies the condition ŝ(0) = 0,
    which is required for a solution to exist when λ=0.

Returns:
    Source field...

### create_plane_wave_source(self, k_mode)
**Описание:** Create plane wave source s(x) = exp(i k·x).

Physical Meaning:
    Creates a plane wave source for testing non-zero modes
    when λ=0.

Args:
    k_mode: Wave vector [kx, ky, kz]

Returns:
    Plane ...

### test_zero_mode_detection(self)
**Описание:** Test detection of zero mode condition.

Physical Meaning:
    Tests that the solver correctly detects when the
    zero mode condition is violated.

### test_zero_mode_handling(self)
**Описание:** Test handling of zero mode when λ=0.

Physical Meaning:
    Tests that the solver correctly handles the zero mode
    when λ=0 by either raising an exception or handling
    it gracefully.

### test_zero_dc_source(self)
**Описание:** Test solution with zero DC component source.

Physical Meaning:
    Tests that the solver works correctly when the source
    satisfies the condition ŝ(0) = 0.

### test_plane_wave_solution(self)
**Описание:** Test plane wave solution when λ=0.

Physical Meaning:
    Tests that plane wave solutions work correctly
    when λ=0, as they have zero DC component.

### test_spectral_coefficients_zero_mode(self)
**Описание:** Test spectral coefficients at k=0 when λ=0.

Physical Meaning:
    Tests that the spectral coefficients are handled
    correctly at k=0 when λ=0.

### test_operator_singularity(self)
**Описание:** Test operator singularity at k=0.

Physical Meaning:
    Tests that the operator correctly handles the singularity
    at k=0 when λ=0.

### test_mixed_frequency_source(self)
**Описание:** Test mixed frequency source with zero DC component.

Physical Meaning:
    Tests that the solver works correctly with sources
    that have multiple frequency components but zero DC component.

### test_error_messages(self)
**Описание:** Test error messages for zero mode violations.

Physical Meaning:
    Tests that appropriate error messages are provided
    when the zero mode condition is violated.

## ./tests/unit/test_level_a/test_A05_residual_energy.py
Methods: 13

### setup_method(self)
**Описание:** Setup test parameters.

### create_plane_wave_source(self, k_mode)
**Описание:** Create plane wave source s(x) = exp(i k·x).

Physical Meaning:
    Creates a plane wave source for testing residual
    computation and energy balance.

Args:
    k_mode: Wave vector [kx, ky, kz]

Ret...

### create_multi_frequency_source(self, modes, amplitudes)
**Описание:** Create multi-frequency source s(x) = Σ c_j e^(i k_j·x).

Physical Meaning:
    Creates a multi-frequency source for testing residual
    computation with complex source terms.

Args:
    modes: List o...

### create_plane_wave_source_for_domain(self, k_mode, domain)
**Описание:** Create plane wave source for a specific domain.

Args:
    k_mode: Wave vector [kx, ky, kz]
    domain: Domain object
    
Returns:
    Source field for the given domain

### compute_residual(self, solution, source, laplacian)
**Описание:** Compute residual R = L_β a - s.

Physical Meaning:
    Computes the residual of the fractional Laplacian equation,
    which measures how well the solution satisfies the original
    equation.

Args:
...

### test_residual_computation(self)
**Описание:** Test residual computation for plane wave solution.

Physical Meaning:
    Tests that the residual R = L_β a - s is computed
    correctly and is small for accurate solutions.

### test_residual_orthogonality(self)
**Описание:** Test orthogonality of residual to solution.

Physical Meaning:
    Tests that the residual is orthogonal to the solution,
    which is a property of the variational formulation.

### test_energy_balance(self)
**Описание:** Test energy balance in the solution.

Physical Meaning:
    Tests that the energy balance is satisfied,
    which is a fundamental property of the equation.

### test_multi_frequency_residual(self)
**Описание:** Test residual computation for multi-frequency source.

Physical Meaning:
    Tests residual computation for complex
    multi-frequency sources to ensure robustness.

### test_residual_spectral_analysis(self)
**Описание:** Test spectral analysis of residual.

Physical Meaning:
    Tests that the residual has the expected spectral
    properties, particularly that it's small in all
    frequency components.

### test_residual_convergence(self)
**Описание:** Test residual convergence with grid refinement.

Physical Meaning:
    Tests that the residual decreases as the grid
    is refined, ensuring numerical convergence.

### test_solver_validation(self)
**Описание:** Test built-in solver validation.

Physical Meaning:
    Tests the built-in validation methods to ensure
    they correctly assess solution quality.

### test_energy_conservation(self)
**Описание:** Test energy conservation properties.

Physical Meaning:
    Tests that energy is conserved in the solution,
    which is a fundamental property of the equation.

## ./tests/unit/test_level_a/test_final_summary.py
Methods: 11

### setup_method(self)
**Описание:** Setup test parameters.

### test_framework_components(self)
**Описание:** Test that all framework components can be created.

### test_spectral_operations(self)
**Описание:** Test spectral operations.

### test_fft_operations(self)
**Описание:** Test FFT operations.

### test_fractional_laplacian_application(self)
**Описание:** Test fractional Laplacian application.

### test_spectral_solution_approach(self)
**Описание:** Test spectral solution approach.

### test_plane_wave_handling(self)
**Описание:** Test plane wave handling.

### test_memory_management(self)
**Описание:** Test memory management.

### test_error_handling(self)
**Описание:** Test error handling.

### test_validation_metrics(self)
**Описание:** Test validation metrics.

### test_framework_summary(self)
**Описание:** Test framework summary.

## ./tests/unit/test_level_a/test_reporting.py
Methods: 10

### __init__(self, output_dir)
**Описание:** Initialize test reporter.

Args:
    output_dir: Directory for output files

### record_test_result(self, test_id, test_name, status, metrics, parameters, execution_time, memory_usage)
**Описание:** Record test result.

Physical Meaning:
    Records the results of a validation test, including
    performance metrics and physical parameters.

Args:
    test_id: Test identifier
    test_name: Test ...

### generate_json_report(self) -> str
**Описание:** Generate JSON report of all test results.

Physical Meaning:
    Generates a comprehensive JSON report containing
    all test results, metrics, and parameters.

Returns:
    Path to generated JSON re...

### generate_csv_log(self) -> str
**Описание:** Generate CSV log of test results.

Physical Meaning:
    Generates a CSV log for trend analysis and
    automated processing of test results.

Returns:
    Path to generated CSV log

### generate_html_report(self) -> str
**Описание:** Generate HTML report with visualizations.

Physical Meaning:
    Generates a comprehensive HTML report with
    visualizations and analysis of test results.

Returns:
    Path to generated HTML report

### create_visualizations(self)
**Описание:** Create visualizations for test results.

Physical Meaning:
    Creates visualizations to analyze test results,
    including error trends, performance metrics,
    and validation quality.

Returns:
  ...

### _create_error_trend_plot(self) -> str
**Описание:** Create error trend plot.

### _create_performance_plot(self) -> str
**Описание:** Create performance plot.

### _create_metrics_summary_plot(self) -> str
**Описание:** Create metrics summary plot.

### generate_summary_report(self) -> str
**Описание:** Generate summary report.

Physical Meaning:
    Generates a summary report with key findings
    and recommendations for Level A validation.

Returns:
    Path to generated summary report

## ./tests/unit/test_level_a/test_simple_basic.py
Methods: 8

### test_domain_creation(self)
**Описание:** Test that Domain can be created.

### test_parameters_creation(self)
**Описание:** Test that Parameters can be created.

### test_simple_fft_operation(self)
**Описание:** Test basic FFT operations.

### test_fractional_laplacian_import(self)
**Описание:** Test that FractionalLaplacian can be imported.

### test_spectral_operations_import(self)
**Описание:** Test that SpectralOperations can be imported.

### test_memory_manager_import(self)
**Описание:** Test that MemoryManager7D can be imported.

### test_fft_plan_import(self)
**Описание:** Test that FFTPlan7D can be imported.

### test_spectral_cache_import(self)
**Описание:** Test that SpectralCoefficientCache can be imported.

## ./tests/unit/test_level_b/test_fundamental_properties.py
Methods: 10

### test_power_law_tail(self)
**Описание:** Test B1: Power law tail in homogeneous medium.

Physical Meaning:
    Validates that the phase field exhibits power law decay
    A(r) ∝ r^(2β-3) in homogeneous medium, confirming the
    fundamental ...

### test_no_spherical_nodes(self)
**Описание:** Test B2: Absence of spherical standing nodes.

Physical Meaning:
    Confirms that spherical standing nodes do not form in
    homogeneous medium, validating the spectral properties
    of the Riesz o...

### test_topological_charge(self)
**Описание:** Test B3: Topological charge of defect.

Physical Meaning:
    Validates the topological stability of the particle core
    through computation of the topological charge.

### test_zone_separation(self)
**Описание:** Test B4: Zone separation (core/transition/tail).

Physical Meaning:
    Quantitatively separates the phase field into three
    characteristic zones and validates their properties.

### __init__(self, config_path)
**Описание:** Initialize Level B test suite.

Args:
    config_path (str): Path to test configuration file.

### _load_config(self, config_path)
**Описание:** Load test configuration from JSON file.

### _get_default_config(self)
**Описание:** Get default test configuration.

### _setup_analyzers(self)
**Описание:** Setup analysis tools.

### run_all_tests(self)
**Описание:** Run all Level B tests and return comprehensive results.

Returns:
    Dict[str, Any]: Complete test results with analysis.

### _create_test_solution(self, domain, center, parameters)
**Описание:** Create a test solution for Level B analysis.

Physical Meaning:
    Creates an analytical test solution that exhibits the expected
    power law behavior A(r) ∝ r^(2β-3) for validation of Level B
    ...

## ./tests/unit/test_level_b/test_simple_level_b.py
Methods: 5

### setUp(self)
**Описание:** Set up test fixtures.

### _create_test_solution(self, beta)
**Описание:** Create analytical test solution with power law behavior.

Physical Meaning:
    Creates a test solution that exhibits the expected
    power law behavior A(r) ∝ r^(2β-3) for validation.

### test_power_law_analysis_basic(self)
**Описание:** Test basic power law analysis functionality.

### test_power_law_different_beta(self)
**Описание:** Test power law analysis for different beta values.

### test_radial_profile_computation(self)
**Описание:** Test radial profile computation.

## ./tests/unit/test_level_b/test_zone_analyzer_7d.py
Methods: 1

### test_zone_indicators_7d_axes()
**Описание:** Нет докстринга

## ./tests/unit/test_level_c/test_level_c_simple.py
Methods: 20

### test_abcd_model_initialization(self)
**Описание:** Test ABCD model initialization.

Physical Meaning:
    Tests that the ABCD model initializes correctly
    with resonator layers.

### test_compute_transmission_matrix(self)
**Описание:** Test transmission matrix computation.

Physical Meaning:
    Tests the computation of 2x2 transmission matrices
    for the resonator chain at given frequencies.

### test_compute_system_admittance(self)
**Описание:** Test system admittance computation.

Physical Meaning:
    Tests the computation of complex admittance Y(ω)
    for the resonator chain.

### test_find_resonance_conditions(self)
**Описание:** Test resonance condition finding.

Physical Meaning:
    Tests the identification of frequencies where
    det(T_total - I) = 0, corresponding to system resonances.

### test_find_system_modes(self)
**Описание:** Test system mode finding.

Physical Meaning:
    Tests the identification of system resonance modes
    with their frequencies and quality factors.

### test_resonator_layer_creation(self)
**Описание:** Test ResonatorLayer creation.

Physical Meaning:
    Tests that ResonatorLayer objects are created correctly
    with proper parameter assignment.

### test_resonator_layer_defaults(self)
**Описание:** Test ResonatorLayer with default values.

Physical Meaning:
    Tests that ResonatorLayer works correctly with
    default parameter values.

### test_system_mode_creation(self)
**Описание:** Test SystemMode creation.

Physical Meaning:
    Tests that SystemMode objects are created correctly
    with proper parameter assignment.

### test_system_mode_defaults(self)
**Описание:** Test SystemMode with default values.

Physical Meaning:
    Tests that SystemMode works correctly with
    default parameter values.

### test_memory_parameters_creation(self)
**Описание:** Test MemoryParameters creation.

Physical Meaning:
    Tests that MemoryParameters objects are created correctly
    with proper parameter assignment.

### test_memory_parameters_defaults(self)
**Описание:** Test MemoryParameters with default values.

Physical Meaning:
    Tests that MemoryParameters works correctly with
    default parameter values.

### test_quench_event_creation(self)
**Описание:** Test QuenchEvent creation.

Physical Meaning:
    Tests that QuenchEvent objects are created correctly
    with proper parameter assignment.

### test_quench_event_types(self)
**Описание:** Test QuenchEvent with different threshold types.

Physical Meaning:
    Tests that QuenchEvent works correctly with
    different threshold types.

### test_dual_mode_source_creation(self)
**Описание:** Test DualModeSource creation.

Physical Meaning:
    Tests that DualModeSource objects are created correctly
    with proper parameter assignment.

### test_dual_mode_source_defaults(self)
**Описание:** Test DualModeSource with default values.

Physical Meaning:
    Tests that DualModeSource works correctly with
    default parameter values.

### test_beating_pattern_creation(self)
**Описание:** Test BeatingPattern creation.

Physical Meaning:
    Tests that BeatingPattern objects are created correctly
    with proper parameter assignment.

### test_beating_pattern_defaults(self)
**Описание:** Test BeatingPattern with default values.

Physical Meaning:
    Tests that BeatingPattern works correctly with
    default parameter values.

### test_matrix_operations(self)
**Описание:** Test matrix operations for ABCD model.

Physical Meaning:
    Tests basic matrix operations used in transmission
    matrix calculations.

### test_complex_operations(self)
**Описание:** Test complex number operations.

Physical Meaning:
    Tests complex number operations used in admittance
    and field calculations.

### test_numerical_stability(self)
**Описание:** Test numerical stability of operations.

Physical Meaning:
    Tests that mathematical operations are numerically
    stable and don't produce NaN or infinite values.

## ./tests/unit/test_level_d/test_level_d_models.py
Methods: 21

### domain(self)
**Описание:** Create test domain.
**Декораторы:** <ast.Attribute object at 0x753e6fa1ed90>

### parameters(self)
**Описание:** Create test parameters.
**Декораторы:** <ast.Attribute object at 0x753e6fa1f790>

### test_field(self, domain)
**Описание:** Create test field.
**Декораторы:** <ast.Attribute object at 0x753e6fa52550>

### test_level_d_models_initialization(self, domain, parameters)
**Описание:** Test Level D models initialization.

### test_create_multi_mode_field(self, domain, parameters, test_field)
**Описание:** Test multi-mode field creation.

### test_analyze_mode_superposition(self, domain, parameters, test_field)
**Описание:** Test mode superposition analysis.

### test_project_field_windows(self, test_field, projection_params)
**Описание:** Test field projection onto windows.

### test_trace_phase_streamlines(self, domain, parameters, test_field)
**Описание:** Test phase streamline tracing.

### test_analyze_multimode_field(self, domain, parameters, test_field)
**Описание:** Test comprehensive multimode field analysis.

### test_validate_field(self, domain, parameters)
**Описание:** Test field validation.

### test_multi_mode_model_initialization(self, domain, parameters)
**Описание:** Test multi-mode model initialization.

### test_analyze_frame_stability(self, domain, parameters, test_field)
**Описание:** Test frame stability analysis.

### test_compute_jaccard_index(self, domain, parameters)
**Описание:** Test Jaccard index computation.

### projection_params(self)
**Описание:** Create projection parameters.
**Декораторы:** <ast.Attribute object at 0x753e6fb04b10>

### test_field_projection_initialization(self, test_field, projection_params)
**Описание:** Test field projection initialization.

### test_project_em_field(self, test_field, projection_params)
**Описание:** Test EM field projection.

### test_project_strong_field(self, test_field, projection_params)
**Описание:** Test strong field projection.

### test_project_weak_field(self, test_field, projection_params)
**Описание:** Test weak field projection.

### test_streamline_analyzer_initialization(self, domain, parameters)
**Описание:** Test streamline analyzer initialization.

### test_analyze_streamlines(self, domain, parameters, test_field)
**Описание:** Test streamline analysis.

### test_level_d_integration(self, domain, parameters)
**Описание:** Test Level D integration.

## ./tests/unit/test_level_e/test_defect_fractional_physics.py
Methods: 11

### domain_3d(self)
**Описание:** Create 3D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb0b7d0>

### physics_params_beta_05(self)
**Описание:** Physics parameters with β=0.5 (fractional case).
**Декораторы:** <ast.Attribute object at 0x753e6fb0bb50>

### physics_params_beta_1(self)
**Описание:** Physics parameters with β=1.0 (classical case).
**Декораторы:** <ast.Attribute object at 0x753e6fb0a790>

### test_fractional_green_normalization_beta_05(self, domain_3d, physics_params_beta_05)
**Описание:** Test fractional Green function normalization for β=0.5.

### test_fractional_green_normalization_beta_1(self, domain_3d, physics_params_beta_1)
**Описание:** Test fractional Green function normalization for β=1.0 (classical case).

### test_energy_monotonicity_under_approach(self, domain_3d, physics_params_beta_05)
**Описание:** Test FRAC-1: energy monotonicity (ΔE≤0) under defect approach.

### test_fractional_green_tail_behavior(self, domain_3d, physics_params_beta_05)
**Описание:** Test that fractional Green function has proper power-law tail.

### test_no_mass_term_in_base_regime(self, domain_3d, physics_params_beta_05)
**Описание:** Test that no mass term is present in base regime (λ=0).

### test_defect_analyzer_fractional_interactions(self, domain_3d, physics_params_beta_05)
**Описание:** Test that topological defect analyzer uses fractional interactions.

### test_fractional_green_normalization_consistency(self, domain_3d)
**Описание:** Test that fractional Green function normalization is consistent.

### test_annihilation_energy_fractional(self, domain_3d, physics_params_beta_05)
**Описание:** Test that annihilation energy uses fractional Green function.

## ./tests/unit/test_level_e/test_defect_physics.py
Methods: 12

### domain_7d(self)
**Описание:** Create 7D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb18510>

### physics_params(self)
**Описание:** Create realistic physics parameters.
**Декораторы:** <ast.Attribute object at 0x753e6fb19f50>

### test_vortex_defect_creation(self, domain_7d, physics_params)
**Описание:** Test vortex defect creation with physical properties.

Physical Meaning:
    Verifies that vortex defects are created with correct
    topological charge and phase winding properties.

### test_topological_charge_calculation(self, domain_7d, physics_params)
**Описание:** Test topological charge calculation.

Physical Meaning:
    Verifies that topological charge is correctly calculated
    for different defect configurations.

### test_defect_interaction_forces(self, domain_7d, physics_params)
**Описание:** Test interaction forces between defects.

Physical Meaning:
    Verifies that interaction forces follow expected
    physical behavior for different charge configurations.

### test_defect_annihilation_physics(self, domain_7d, physics_params)
**Описание:** Test defect annihilation process.

Physical Meaning:
    Verifies that defect-antidefect annihilation
    follows correct physical behavior.

### test_multi_defect_system_physics(self, domain_7d, physics_params)
**Описание:** Test multi-defect system behavior.

Physical Meaning:
    Verifies that multi-defect systems exhibit correct
    collective behavior and energy conservation.

### test_defect_dynamics_physics(self, domain_7d, physics_params)
**Описание:** Test defect dynamics and motion.

Physical Meaning:
    Verifies that defect motion follows the Thiele equation
    with correct force calculations and dynamics.

### test_interaction_potential_setup(self, domain_7d, physics_params)
**Описание:** Test interaction potential setup.

Physical Meaning:
    Verifies that interaction potentials are correctly
    initialized with proper physical parameters.

### test_green_function_physics(self, domain_7d, physics_params)
**Описание:** Test Green function calculations.

Physical Meaning:
    Verifies that Green functions follow correct
    physical behavior for defect interactions.

### test_defect_charge_conservation(self, domain_7d, physics_params)
**Описание:** Test topological charge conservation.

Physical Meaning:
    Verifies that topological charge is conserved
    during defect interactions and dynamics.

### test_defect_energy_scaling(self, domain_7d, physics_params)
**Описание:** Test energy scaling with defect parameters.

Physical Meaning:
    Verifies that defect energies scale correctly
    with physical parameters.

## ./tests/unit/test_level_e/test_level_e_simple.py
Methods: 28

### test_initialization(self)
**Описание:** Test LevelEExperiments initialization.

### test_lhs_sampling(self)
**Описание:** Test Latin Hypercube sampling.

### test_sobol_indices(self)
**Описание:** Test Sobol index computation.

### test_parameter_sensitivity_analysis(self)
**Описание:** Test complete parameter sensitivity analysis.

### test_noise_robustness(self)
**Описание:** Test noise robustness testing.

### test_parameter_uncertainty(self)
**Описание:** Test parameter uncertainty testing.

### test_geometry_perturbations(self)
**Описание:** Test geometry perturbation testing.

### test_grid_convergence(self)
**Описание:** Test grid convergence analysis.

### test_domain_size_effects(self)
**Описание:** Test domain size effects analysis.

### test_time_step_stability(self)
**Описание:** Test time step stability analysis.

### test_failure_detection(self)
**Описание:** Test failure detection.

### test_failure_boundaries(self)
**Описание:** Test failure boundary analysis.

### test_phase_mapping(self)
**Описание:** Test phase mapping.

### test_resonance_classification(self)
**Описание:** Test resonance classification.

### test_performance_analysis(self)
**Описание:** Test performance analysis.

### test_scaling_analysis(self)
**Описание:** Test scaling analysis.

### test_soliton_model_initialization(self)
**Описание:** Test SolitonModel initialization.

### test_baryon_soliton_initialization(self)
**Описание:** Test BaryonSoliton initialization.

### test_skyrmion_soliton_initialization(self)
**Описание:** Test SkyrmionSoliton initialization.

### test_defect_model_initialization(self)
**Описание:** Test DefectModel initialization.

### test_vortex_defect_initialization(self)
**Описание:** Test VortexDefect initialization.

### test_multi_defect_system_initialization(self)
**Описание:** Test MultiDefectSystem initialization.

### test_full_analysis(self)
**Описание:** Test full analysis execution.

### test_soliton_experiments(self)
**Описание:** Test soliton experiments.

### test_defect_experiments(self)
**Описание:** Test defect experiments.

### test_save_and_load_results(self)
**Описание:** Test saving and loading results.

### test_configuration_loading(self)
**Описание:** Test loading configuration from file.

### __init__(self)
**Описание:** Нет докстринга

## ./tests/unit/test_level_e/test_soliton_energy_physics.py
Methods: 12

### domain_3d(self)
**Описание:** Create 7D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb6cd10>

### physics_params(self)
**Описание:** Create realistic physics parameters.
**Декораторы:** <ast.Attribute object at 0x753e6fb6d7d0>

### simple_u1_phase_field(self, domain_3d)
**Описание:** Create simple U(1)^3 phase field for testing.

Physical Meaning:
    Creates a simplified U(1)^3 phase field configuration that
    is easier to analyze and has known theoretical properties.
**Декораторы:** <ast.Attribute object at 0x753e6fb62d90>

### test_kinetic_energy_physical_properties(self, domain_3d, physics_params, simple_u1_phase_field)
**Описание:** Test physical properties of kinetic energy calculation.

Physical Meaning:
    Verifies that kinetic energy has the correct physical properties:
    - Non-negative for all field configurations
    - Z...

### test_skyrme_energy_physical_properties(self, domain_3d, physics_params, simple_u1_phase_field)
**Описание:** Test physical properties of Skyrme energy calculation.

Physical Meaning:
    Verifies that Skyrme energy has the correct physical properties:
    - Non-negative for all field configurations
    - Pro...

### test_wzw_energy_physical_properties(self, domain_3d, physics_params, simple_u1_phase_field)
**Описание:** Test physical properties of WZW energy calculation.

Physical Meaning:
    Verifies that WZW energy has the correct physical properties:
    - Finite for all field configurations
    - Contributes to ...

### test_energy_scaling_with_domain_size(self, physics_params)
**Описание:** Test energy scaling with domain size.

Physical Meaning:
    Verifies that energy calculations scale correctly with
    domain size and resolution.

### test_energy_conservation_under_rotation(self, domain_3d, physics_params, simple_u1_phase_field)
**Описание:** Test energy conservation under field rotations.

Physical Meaning:
    Verifies that energy is conserved under global rotations
    of the field configuration, as required by rotational symmetry.

### test_energy_minimum_properties(self, domain_3d, physics_params)
**Описание:** Test energy minimum properties.

Physical Meaning:
    Verifies that the energy functional has the correct
    minimum properties for stable soliton configurations.

### test_energy_gradient_properties(self, domain_3d, physics_params, simple_u1_phase_field)
**Описание:** Test energy gradient properties.

Physical Meaning:
    Verifies that the energy gradient has the correct
    properties for optimization algorithms.

### test_energy_hessian_properties(self, domain_3d, physics_params, simple_u1_phase_field)
**Описание:** Test energy Hessian properties.

Physical Meaning:
    Verifies that the energy Hessian has the correct
    properties for Newton-Raphson optimization.

### create_u1_phase(domain)
**Описание:** Нет докстринга

## ./tests/unit/test_level_e/test_soliton_physics.py
Methods: 21

### domain_3d(self)
**Описание:** Create 7D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa2b510>

### physics_params(self)
**Описание:** Create realistic physics parameters.
**Декораторы:** <ast.Attribute object at 0x753e6fa2ad50>

### u1_phase_field(self, domain_3d)
**Описание:** Create U(1)^3 phase field configuration.

Physical Meaning:
    Creates a 7D phase field configuration Θ(x,φ) ∈ T^3_φ with
    controlled winding over φ-coordinates. This represents a B=1
    soliton ...
**Декораторы:** <ast.Attribute object at 0x753e6fa50950>

### test_wzw_term_setup(self, domain_3d, physics_params)
**Описание:** Test WZW term setup with physical parameters.

Physical Meaning:
    Verifies that the WZW term is correctly initialized with
    the proper coefficient (N_c/240π²) and integration parameters.

### test_topological_charge_setup(self, domain_3d, physics_params)
**Описание:** Test topological charge setup with physical parameters.

Physical Meaning:
    Verifies that the topological charge calculation is correctly
    initialized with proper integration parameters.

### test_fr_constraints_setup(self, domain_3d, physics_params)
**Описание:** Test FR constraints setup with physical parameters.

Physical Meaning:
    Verifies that Finkelstein-Rubinstein constraints are correctly
    initialized to ensure fermionic statistics.

### test_charge_specific_terms_baryon(self, domain_3d, physics_params)
**Описание:** Test charge-specific terms for B=1 baryon.

Physical Meaning:
    Verifies that B=1 soliton is correctly configured with
    U(1)^3 phase boundary condition and single baryon constraints.

### test_charge_specific_terms_skyrmion(self, domain_3d, physics_params)
**Описание:** Test charge-specific terms for B>1 skyrmion.

Physical Meaning:
    Verifies that B>1 soliton is correctly configured with
    multi-U(1)^3 phase boundary condition and multi-baryon constraints.

### test_charge_specific_terms_antibaryon(self, domain_3d, physics_params)
**Описание:** Test charge-specific terms for B<0 antibaryon.

Physical Meaning:
    Verifies that B<0 soliton is correctly configured with
    anti-U(1)^3 phase boundary condition and antibaryon constraints.

### test_kinetic_energy_calculation(self, domain_3d, physics_params, u1_phase_field)
**Описание:** Test kinetic energy calculation with U(1)^3 phase field.

Physical Meaning:
    Verifies that kinetic energy is correctly computed for
    a U(1)^3 phase field configuration. The kinetic energy should...

### test_skyrme_energy_calculation(self, domain_3d, physics_params, u1_phase_field)
**Описание:** Test Skyrme energy calculation with U(1)^3 phase field.

Physical Meaning:
    Verifies that Skyrme energy is correctly computed for
    a U(1)^3 phase field configuration. The Skyrme energy should
  ...

### test_wzw_energy_calculation(self, domain_3d, physics_params, u1_phase_field)
**Описание:** Test WZW energy calculation with U(1)^3 phase field.

Physical Meaning:
    Verifies that WZW energy is correctly computed for
    a U(1)^3 phase field configuration. The WZW energy should
    be fini...

### test_total_energy_calculation(self, domain_3d, physics_params, u1_phase_field)
**Описание:** Test total energy calculation with U(1)^3 phase field.

Physical Meaning:
    Verifies that total energy is correctly computed as the sum
    of kinetic, Skyrme, and WZW contributions.

### test_topological_charge_calculation(self, domain_3d, physics_params, u1_phase_field)
**Описание:** Test topological charge calculation with U(1)^3 phase field.

Physical Meaning:
    Verifies that topological charge is correctly computed for
    a U(1)^3 phase field configuration. For a U(1)^3 phas...

### test_fr_constraints_application(self, domain_3d, physics_params, u1_phase_field)
**Описание:** Test FR constraints application to field.

Physical Meaning:
    Verifies that FR constraints are correctly applied to ensure
    fermionic statistics by checking the sign change under 2π rotation.

### test_soliton_solution_finding(self, domain_3d, physics_params, u1_phase_field)
**Описание:** Test soliton solution finding with U(1)^3 phase initial guess.

Physical Meaning:
    Verifies that the soliton finding algorithm converges to a
    stable solution with proper energy and topological ...

### test_stability_analysis(self, domain_3d, physics_params, u1_phase_field)
**Описание:** Test stability analysis of soliton solution.

Physical Meaning:
    Verifies that stability analysis correctly identifies
    stable and unstable modes of the soliton solution.

### test_energy_conservation(self, domain_3d, physics_params, u1_phase_field)
**Описание:** Test energy conservation properties.

Physical Meaning:
    Verifies that energy calculations are consistent and
    that the energy functional is properly implemented.

### test_charge_conservation(self, domain_3d, physics_params, u1_phase_field)
**Описание:** Test topological charge conservation properties.

Physical Meaning:
    Verifies that topological charge calculations are consistent
    and that charge is properly conserved.

### test_parameter_dependence(self, domain_3d)
**Описание:** Test dependence on physical parameters.

Physical Meaning:
    Verifies that soliton properties depend correctly on
    physical parameters like coupling constants.

### test_domain_independence(self, physics_params)
**Описание:** Test independence on domain parameters.

Physical Meaning:
    Verifies that soliton properties are correctly scaled
    with domain size and resolution.

## ./tests/unit/test_level_e/test_soliton_topology_physics.py
Methods: 14

### domain_3d(self)
**Описание:** Create 7D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fab6e50>

### physics_params(self)
**Описание:** Create realistic physics parameters.
**Декораторы:** <ast.Attribute object at 0x753e6fab5f50>

### u1_phase_field_b1(self, domain_3d)
**Описание:** Create B=1 U(1)^3 phase field configuration.

Physical Meaning:
    Creates a U(1)^3 phase field configuration with topological
    charge B=1, representing a single baryon.
**Декораторы:** <ast.Attribute object at 0x753e6fa968d0>

### trivial_field(self, domain_3d)
**Описание:** Create trivial field configuration.

Physical Meaning:
    Creates a trivial field configuration U(x) = I everywhere,
    which should have topological charge B=0.
**Декораторы:** <ast.Attribute object at 0x753e6fa8e610>

### test_topological_charge_b1_u1_phase(self, domain_3d, physics_params, u1_phase_field_b1)
**Описание:** Test topological charge calculation for B=1 U(1)^3 phase field.

Physical Meaning:
    Verifies that a U(1)^3 phase field configuration has
    topological charge B≈1, as expected theoretically.

### test_topological_charge_trivial_field(self, domain_3d, physics_params, trivial_field)
**Описание:** Test topological charge calculation for trivial field.

Physical Meaning:
    Verifies that a trivial field configuration has
    topological charge B=0, as expected theoretically.

### test_topological_charge_conservation(self, domain_3d, physics_params, u1_phase_field_b1)
**Описание:** Test topological charge conservation.

Physical Meaning:
    Verifies that topological charge is conserved under
    continuous deformations of the field configuration.

### test_topological_charge_scaling(self, physics_params)
**Описание:** Test topological charge scaling with domain size.

Physical Meaning:
    Verifies that topological charge calculations scale
    correctly with domain size and resolution.

### test_topological_charge_under_rotation(self, domain_3d, physics_params, u1_phase_field_b1)
**Описание:** Test topological charge under field rotations.

Physical Meaning:
    Verifies that topological charge is invariant under
    global rotations of the field configuration.

### test_topological_charge_precision(self, domain_3d, physics_params, u1_phase_field_b1)
**Описание:** Test topological charge calculation precision.

Physical Meaning:
    Verifies that topological charge calculations have
    sufficient precision for physical applications.

### test_topological_charge_boundary_effects(self, physics_params)
**Описание:** Test topological charge boundary effects.

Physical Meaning:
    Verifies that topological charge calculations are
    not significantly affected by boundary conditions.

### test_topological_charge_integration_radius(self, domain_3d, physics_params, u1_phase_field_b1)
**Описание:** Test topological charge integration radius dependence.

Physical Meaning:
    Verifies that topological charge calculations are
    not significantly affected by integration radius.

### test_topological_charge_charge_specific_terms(self, domain_3d, physics_params)
**Описание:** Test topological charge for different charge-specific terms.

Physical Meaning:
    Verifies that different soliton types (baryon, skyrmion, antibaryon)
    have the correct topological charge propert...

### create_u1_phase(domain)
**Описание:** Нет докстринга

## ./tests/unit/test_level_f/test_collective.py
Methods: 26

### domain(self)
**Описание:** Create test domain.
**Декораторы:** <ast.Attribute object at 0x753e6fa8c390>

### particles(self)
**Описание:** Create test particles.
**Декораторы:** <ast.Attribute object at 0x753e6fa8cd90>

### system(self, domain, particles)
**Описание:** Create test system.
**Декораторы:** <ast.Attribute object at 0x753e6fa8d590>

### excitation_params(self)
**Описание:** Create test excitation parameters.
**Декораторы:** <ast.Attribute object at 0x753e6fa8df10>

### excitations(self, system, excitation_params)
**Описание:** Create test excitations.
**Декораторы:** <ast.Attribute object at 0x753e6fa8e610>

### test_initialization(self, system, excitation_params)
**Описание:** Test excitations initialization.

Physical Meaning:
    Verifies that the excitations model is correctly
    initialized with the specified parameters.

### test_harmonic_excitation(self, excitations)
**Описание:** Test harmonic excitation.

Physical Meaning:
    Verifies that harmonic excitation is correctly
    applied to the system.

### test_impulse_excitation(self, excitations)
**Описание:** Test impulse excitation.

Physical Meaning:
    Verifies that impulse excitation is correctly
    applied to the system.

### test_frequency_sweep_excitation(self, excitations)
**Описание:** Test frequency sweep excitation.

Physical Meaning:
    Verifies that frequency sweep excitation is
    correctly applied to the system.

### test_system_excitation(self, excitations)
**Описание:** Test system excitation with different types.

Physical Meaning:
    Verifies that the system is correctly excited
    with different types of external fields.

### test_response_analysis(self, excitations)
**Описание:** Test response analysis.

Physical Meaning:
    Verifies that system response is correctly
    analyzed to extract collective mode properties.

### test_dispersion_relations(self, excitations)
**Описание:** Test dispersion relations computation.

Physical Meaning:
    Verifies that dispersion relations are correctly
    computed for collective modes.

### test_susceptibility_computation(self, excitations)
**Описание:** Test susceptibility computation.

Physical Meaning:
    Verifies that the susceptibility function is
    correctly computed for collective excitations.

### test_spectral_peak_detection(self, excitations)
**Описание:** Test spectral peak detection.

Physical Meaning:
    Verifies that spectral peaks are correctly
    identified in the response spectrum.

### test_step_resonator_transmission_analysis(self, excitations)
**Описание:** Test step resonator transmission analysis.

Physical Meaning:
    Verifies that transmission/reflection coefficients
    are correctly computed through step resonator boundaries.

### test_participation_ratios(self, excitations)
**Описание:** Test participation ratios computation.

Physical Meaning:
    Verifies that participation ratios are correctly
    computed for collective modes.

### test_quality_factors(self, excitations)
**Описание:** Test quality factors computation.

Physical Meaning:
    Verifies that quality factors are correctly
    computed for collective modes.

### test_dispersion_equation_solution(self, excitations)
**Описание:** Test dispersion equation solution.

Physical Meaning:
    Verifies that the dispersion equation is
    correctly solved for given wave vectors.

### test_group_velocity_computation(self, excitations)
**Описание:** Test group velocity computation.

Physical Meaning:
    Verifies that group velocities are correctly
    computed for collective modes.

### test_dispersion_relation_fitting(self, excitations)
**Описание:** Test dispersion relation fitting.

Physical Meaning:
    Verifies that dispersion relations are correctly
    fitted to computed data.

### test_external_force_computation(self, excitations)
**Описание:** Test external force computation.

Physical Meaning:
    Verifies that external forces are correctly
    computed for particles.

### test_parameter_dependence(self, system)
**Описание:** Test dependence on excitation parameters.

Physical Meaning:
    Verifies that the system response changes
    correctly with excitation parameters.

### test_different_excitation_types(self, system)
**Описание:** Test different excitation types.

Physical Meaning:
    Verifies that different excitation types
    work correctly.

### test_error_handling(self, system)
**Описание:** Test error handling for invalid parameters.

Physical Meaning:
    Verifies that the system handles invalid
    parameters gracefully.

### test_step_resonator_transmission(self, excitations)
**Описание:** Test step resonator transmission analysis.

Physical Meaning:
    Verifies that step resonator transmission/reflection
    coefficients are correctly computed.

### test_boundary_energy_flux(self, excitations)
**Описание:** Test boundary energy flux computation.

Physical Meaning:
    Verifies that energy flux through step resonator
    boundaries is correctly computed.

## ./tests/unit/test_level_f/test_multi_particle.py
Methods: 26

### domain(self)
**Описание:** Create test domain.
**Декораторы:** <ast.Attribute object at 0x753e6fa8b610>

### particles(self)
**Описание:** Create test particles.
**Декораторы:** <ast.Attribute object at 0x753e6fa8ac50>

### system(self, domain, particles)
**Описание:** Create test system.
**Декораторы:** <ast.Attribute object at 0x753e6fa8a450>

### test_initialization(self, domain, particles)
**Описание:** Test system initialization.

Physical Meaning:
    Verifies that the system is correctly initialized
    with the specified domain and particles.

### test_effective_potential_computation(self, system)
**Описание:** Test effective potential computation.

Physical Meaning:
    Verifies that the effective potential is correctly
    computed including single-particle and pair-wise
    interactions.

### test_collective_modes_analysis(self, system)
**Описание:** Test collective modes analysis.

Physical Meaning:
    Verifies that collective modes are correctly
    identified and analyzed.

### test_correlation_analysis(self, system)
**Описание:** Test correlation analysis.

Physical Meaning:
    Verifies that correlation functions are correctly
    computed and analyzed.

### test_stability_check(self, system)
**Описание:** Test stability check.

Physical Meaning:
    Verifies that the system stability is correctly
    analyzed.

### test_single_particle_potential(self, system)
**Описание:** Test single particle potential computation.

Physical Meaning:
    Verifies that single-particle potentials are
    correctly computed.

### test_pair_interaction(self, system)
**Описание:** Test pair interaction computation.

Physical Meaning:
    Verifies that pair-wise interactions are
    correctly computed.

### test_three_body_interaction(self, system)
**Описание:** Test three-body interaction computation.

Physical Meaning:
    Verifies that three-body interactions are
    correctly computed.

### test_dynamics_matrix(self, system)
**Описание:** Test dynamics matrix computation.

Physical Meaning:
    Verifies that the dynamics matrix is correctly
    computed for collective mode analysis.

### test_participation_ratios(self, system)
**Описание:** Test participation ratios computation.

Physical Meaning:
    Verifies that participation ratios are correctly
    computed for collective modes.

### test_spatial_correlations(self, system)
**Описание:** Test spatial correlations computation.

Physical Meaning:
    Verifies that spatial correlations are correctly
    computed between particle positions.

### test_phase_correlations(self, system)
**Описание:** Test phase correlations computation.

Physical Meaning:
    Verifies that phase correlations are correctly
    computed between particle phases.

### test_interaction_strength_dependence(self, domain, particles)
**Описание:** Test dependence on interaction strength.

Physical Meaning:
    Verifies that the system behavior changes
    correctly with interaction strength.

### test_interaction_range_dependence(self, domain, particles)
**Описание:** Test dependence on interaction range.

Physical Meaning:
    Verifies that the system behavior changes
    correctly with interaction range.

### test_energy_conservation(self, system)
**Описание:** Test energy conservation.

Physical Meaning:
    Verifies that energy is conserved in the
    multi-particle system.

### test_topological_charge_conservation(self, system)
**Описание:** Test topological charge conservation.

Physical Meaning:
    Verifies that total topological charge is
    conserved in the system.

### test_system_with_single_particle(self, domain)
**Описание:** Test system with single particle.

Physical Meaning:
    Verifies that the system works correctly
    with a single particle.

### test_system_with_multiple_particles(self, domain)
**Описание:** Test system with multiple particles.

Physical Meaning:
    Verifies that the system works correctly
    with multiple particles.

### test_7d_phase_field_energy(self, system)
**Описание:** Test 7D phase field energy computation.

Physical Meaning:
    Verifies that 7D phase field energy is correctly
    computed for particles using 7D BVP theory.

### test_7d_bvp_energy(self, system)
**Описание:** Test 7D BVP energy computation.

Physical Meaning:
    Verifies that 7D BVP energy is correctly computed
    using the fractional Laplacian operator.

### test_7d_phase_coherence(self, system)
**Описание:** Test 7D phase coherence computation.

Physical Meaning:
    Verifies that 7D phase coherence is correctly
    computed between particles.

### test_get_phase_field_around_particle(self, system)
**Описание:** Test phase field extraction around particle.

Physical Meaning:
    Verifies that phase field is correctly extracted
    in the vicinity of a particle.

### test_extract_spherical_field(self, system)
**Описание:** Test spherical field extraction.

Physical Meaning:
    Verifies that phase field is correctly extracted
    in a spherical region around a center point.

## ./tests/unit/test_level_f/test_nonlinear.py
Methods: 29

### domain(self)
**Описание:** Create test domain.
**Декораторы:** <ast.Attribute object at 0x753e6fa67590>

### particles(self)
**Описание:** Create test particles.
**Декораторы:** <ast.Attribute object at 0x753e6fa66b90>

### system(self, domain, particles)
**Описание:** Create test system.
**Декораторы:** <ast.Attribute object at 0x753e6fa66390>

### nonlinear_params(self)
**Описание:** Create test nonlinear parameters.
**Декораторы:** <ast.Attribute object at 0x753e6fa64050>

### nonlinear(self, system, nonlinear_params)
**Описание:** Create test nonlinear effects.
**Декораторы:** <ast.Attribute object at 0x753e6fa64750>

### test_initialization(self, system, nonlinear_params)
**Описание:** Test nonlinear effects initialization.

Physical Meaning:
    Verifies that the nonlinear effects model is
    correctly initialized with the specified parameters.

### test_cubic_nonlinearity_setup(self, system)
**Описание:** Test cubic nonlinearity setup.

Physical Meaning:
    Verifies that cubic nonlinearity is correctly
    set up with the appropriate potential and derivative.

### test_quartic_nonlinearity_setup(self, system)
**Описание:** Test quartic nonlinearity setup.

Physical Meaning:
    Verifies that quartic nonlinearity is correctly
    set up with the appropriate potential and derivative.

### test_sine_gordon_nonlinearity_setup(self, system)
**Описание:** Test sine-Gordon nonlinearity setup.

Physical Meaning:
    Verifies that sine-Gordon nonlinearity is correctly
    set up with the appropriate potential and derivative.

### test_add_nonlinear_interactions(self, nonlinear)
**Описание:** Test adding nonlinear interactions.

Physical Meaning:
    Verifies that nonlinear interactions are correctly
    added to the system.

### test_find_nonlinear_modes(self, nonlinear)
**Описание:** Test finding nonlinear modes.

Physical Meaning:
    Verifies that nonlinear modes are correctly
    identified in the system.

### test_find_soliton_solutions(self, nonlinear)
**Описание:** Test finding soliton solutions.

Physical Meaning:
    Verifies that soliton solutions are correctly
    identified in the system.

### test_sine_gordon_solitons(self, system)
**Описание:** Test sine-Gordon soliton solutions.

Physical Meaning:
    Verifies that sine-Gordon solitons are correctly
    identified.

### test_cubic_solitons(self, system)
**Описание:** Test cubic soliton solutions.

Physical Meaning:
    Verifies that cubic solitons are correctly
    identified.

### test_quartic_solitons(self, system)
**Описание:** Test quartic soliton solutions.

Physical Meaning:
    Verifies that quartic solitons are correctly
    identified.

### test_soliton_profile_computation(self, nonlinear)
**Описание:** Test soliton profile computation.

Physical Meaning:
    Verifies that soliton profiles are correctly
    computed.

### test_nonlinear_corrections(self, nonlinear)
**Описание:** Test nonlinear corrections computation.

Physical Meaning:
    Verifies that nonlinear corrections are correctly
    computed for linear modes.

### test_bifurcation_points(self, nonlinear)
**Описание:** Test bifurcation points identification.

Physical Meaning:
    Verifies that bifurcation points are correctly
    identified in the system.

### test_nonlinear_stability_analysis(self, nonlinear)
**Описание:** Test nonlinear stability analysis.

Physical Meaning:
    Verifies that nonlinear stability is correctly
    analyzed.

### test_stability_check(self, nonlinear)
**Описание:** Test stability check.

Physical Meaning:
    Verifies that the stability of nonlinear solutions
    is correctly checked.

### test_growth_rates_computation(self, nonlinear)
**Описание:** Test growth rates computation.

Physical Meaning:
    Verifies that growth rates are correctly
    computed for instability analysis.

### test_stability_region_identification(self, nonlinear)
**Описание:** Test stability region identification.

Physical Meaning:
    Verifies that stability regions are correctly
    identified in parameter space.

### test_different_nonlinear_types(self, system)
**Описание:** Test different nonlinear types.

Physical Meaning:
    Verifies that different nonlinear types
    work correctly.

### test_parameter_dependence(self, system)
**Описание:** Test dependence on nonlinear parameters.

Physical Meaning:
    Verifies that the system behavior changes
    correctly with nonlinear parameters.

### test_soliton_analysis(self, nonlinear)
**Описание:** Test soliton analysis.

Physical Meaning:
    Verifies that soliton properties are correctly
    analyzed.

### test_error_handling(self, system)
**Описание:** Test error handling for invalid parameters.

Physical Meaning:
    Verifies that the system handles invalid
    parameters gracefully.

### test_linear_stability_analysis(self, nonlinear)
**Описание:** Test linear stability analysis.

Physical Meaning:
    Verifies that linear stability is correctly
    analyzed.

### test_boundary_energy_exchange(self, nonlinear)
**Описание:** Test boundary energy exchange computation.

Physical Meaning:
    Verifies that energy exchange through step resonator
    boundaries is correctly computed.

### test_equations_of_motion_without_damping(self, nonlinear)
**Описание:** Test equations of motion without damping.

Physical Meaning:
    Verifies that equations of motion are correctly
    formulated without classical damping terms.

## ./tests/unit/test_level_f/test_transitions.py
Methods: 26

### domain(self)
**Описание:** Create test domain.
**Декораторы:** <ast.Attribute object at 0x753e6fba5290>

### particles(self)
**Описание:** Create test particles.
**Декораторы:** <ast.Attribute object at 0x753e6fba44d0>

### system(self, domain, particles)
**Описание:** Create test system.
**Декораторы:** <ast.Attribute object at 0x753e6fa95850>

### transitions(self, system)
**Описание:** Create test transitions.
**Декораторы:** <ast.Attribute object at 0x753e6fa95410>

### test_initialization(self, system)
**Описание:** Test transitions initialization.

Physical Meaning:
    Verifies that the phase transitions model is
    correctly initialized with the system.

### test_parameter_sweep(self, transitions)
**Описание:** Test parameter sweep.

Physical Meaning:
    Verifies that parameter sweeps are correctly
    performed to study phase transitions.

### test_order_parameters_computation(self, transitions)
**Описание:** Test order parameters computation.

Physical Meaning:
    Verifies that order parameters are correctly
    computed for the system.

### test_critical_points_identification(self, transitions)
**Описание:** Test critical points identification.

Physical Meaning:
    Verifies that critical points are correctly
    identified in the phase diagram.

### test_topological_order_computation(self, transitions)
**Описание:** Test topological order parameter computation.

Physical Meaning:
    Verifies that the topological order parameter
    is correctly computed.

### test_phase_coherence_computation(self, transitions)
**Описание:** Test phase coherence computation.

Physical Meaning:
    Verifies that the phase coherence order
    parameter is correctly computed.

### test_spatial_order_computation(self, transitions)
**Описание:** Test spatial order computation.

Physical Meaning:
    Verifies that the spatial order parameter
    is correctly computed.

### test_energy_density_computation(self, transitions)
**Описание:** Test energy density computation.

Physical Meaning:
    Verifies that the energy density order
    parameter is correctly computed.

### test_system_state_analysis(self, transitions)
**Описание:** Test system state analysis.

Physical Meaning:
    Verifies that the system state is correctly
    analyzed.

### test_discontinuity_detection(self, transitions)
**Описание:** Test discontinuity detection.

Physical Meaning:
    Verifies that discontinuities are correctly
    detected in order parameters.

### test_critical_point_detection(self, transitions)
**Описание:** Test critical point detection.

Physical Meaning:
    Verifies that critical points are correctly
    detected in order parameters.

### test_critical_exponents_computation(self, transitions)
**Описание:** Test critical exponents computation.

Physical Meaning:
    Verifies that critical exponents are correctly
    computed for phase transitions.

### test_phase_stability_analysis(self, transitions)
**Описание:** Test phase stability analysis.

Physical Meaning:
    Verifies that phase stability is correctly
    analyzed.

### test_parameter_update(self, transitions)
**Описание:** Test parameter update.

Physical Meaning:
    Verifies that system parameters are correctly
    updated.

### test_invalid_parameter_update(self, transitions)
**Описание:** Test invalid parameter update.

Physical Meaning:
    Verifies that invalid parameters are handled
    correctly.

### test_equilibration(self, transitions)
**Описание:** Test system equilibration.

Physical Meaning:
    Verifies that the system is correctly
    equilibrated.

### test_phase_boundary_analysis(self, transitions)
**Описание:** Test phase boundary analysis.

Physical Meaning:
    Verifies that phase boundaries are correctly
    analyzed.

### test_stability_region_identification(self, transitions)
**Описание:** Test stability region identification.

Physical Meaning:
    Verifies that stability regions are correctly
    identified.

### test_phase_stability_check(self, transitions)
**Описание:** Test phase stability check.

Physical Meaning:
    Verifies that phase stability is correctly
    checked.

### test_different_parameter_types(self, transitions)
**Описание:** Test different parameter types.

Physical Meaning:
    Verifies that different parameter types
    work correctly.

### test_order_parameter_consistency(self, transitions)
**Описание:** Test order parameter consistency.

Physical Meaning:
    Verifies that order parameters are consistent
    across multiple computations.

### test_critical_point_properties(self, transitions)
**Описание:** Test critical point properties.

Physical Meaning:
    Verifies that critical points have the
    expected properties.

## ./tests/unit/test_level_g/test_envelope_effective_metric.py
Methods: 2

### test_effective_metric_basic_isotropic()
**Описание:** Нет докстринга

### test_effective_metric_fallback_params()
**Описание:** Нет докстринга

## ./tests/unit/test_level_g/test_gravity_physics.py
Methods: 14

### domain_7d(self)
**Описание:** Create 7D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fce9b90>

### gravity_params(self)
**Описание:** Create realistic gravity parameters.
**Декораторы:** <ast.Attribute object at 0x753e6fce8850>

### mock_system(self, domain_7d)
**Описание:** Create mock system for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fcfc250>

### test_effective_metric_basic_properties(self, gravity_params)
**Описание:** Нет докстринга

### test_metric_sensitivity_to_invariants(self)
**Описание:** Нет докстринга

### test_metric_is_real_and_finite(self)
**Описание:** Нет докстринга

### test_time_component_negative(self)
**Описание:** Нет докстринга

### test_phase_block_is_identity(self)
**Описание:** Нет докстринга

### test_metric_stability_under_small_param_change(self)
**Описание:** Нет докстринга

### test_diagonal_dominance(self)
**Описание:** Нет докстринга

### test_metric_output_consistency(self)
**Описание:** Нет докстринга

### test_metric_scaling_changes_with_params(self)
**Описание:** Нет докстринга

### test_metric_diagonal_structure(self)
**Описание:** Нет докстринга

### __init__(self, domain)
**Описание:** Нет докстринга

## ./tests/unit/test_level_g/test_vbp_envelope_physics.py
Methods: 22

### domain_7d(self)
**Описание:** Create 7D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb63190>

### envelope_params(self)
**Описание:** Create VBP envelope parameters.
**Декораторы:** <ast.Attribute object at 0x753e6fb51a90>

### phase_field_7d(self, domain_7d)
**Описание:** Create 7D phase field for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa20990>

### test_envelope_curvature_scalar_positivity(self, domain_7d, envelope_params, phase_field_7d)
**Описание:** Test that envelope curvature scalar is positive.

### test_anisotropy_index_boundedness(self, domain_7d, envelope_params, phase_field_7d)
**Описание:** Test that anisotropy index is bounded.

### test_effective_metric_7d_structure(self, domain_7d, envelope_params, phase_field_7d)
**Описание:** Test that effective metric has correct 7D structure.

### test_effective_metric_time_component(self, domain_7d, envelope_params, phase_field_7d)
**Описание:** Test that time component follows g00=-1/c_φ^2 with correction factor.

### test_focusing_rate_energy_argument(self, domain_7d, envelope_params, phase_field_7d)
**Описание:** Test that focusing rate is consistent with energy argument.

### test_envelope_balance_convergence(self, domain_7d, envelope_params, phase_field_7d)
**Описание:** Test that envelope balance equation converges.

### test_effective_metric_from_solution(self, domain_7d, envelope_params, phase_field_7d)
**Описание:** Test that effective metric is computed from solution.

### test_balance_operator_components(self, domain_7d, envelope_params, phase_field_7d)
**Описание:** Test that balance operator has correct components.

### wave_params(self)
**Описание:** Create gravitational wave parameters.
**Декораторы:** <ast.Attribute object at 0x753e6fb49590>

### envelope_solution_7d(self, domain_7d)
**Описание:** Create 7D envelope solution for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb18f50>

### test_gravitational_waves_c_T_equals_c_phi(self, domain_7d, wave_params, envelope_solution_7d)
**Описание:** Test that gravitational waves use c_T=c_φ.

### test_gw1_amplitude_law(self, domain_7d, wave_params, envelope_solution_7d)
**Описание:** Test GW-1 amplitude law: |h|∝a^{-1} when Γ=K=0.

### test_7d_polarization_modes(self, domain_7d, wave_params, envelope_solution_7d)
**Описание:** Test that gravitational waves have 7D polarization modes.

### test_strain_tensor_7d_structure(self, domain_7d, wave_params, envelope_solution_7d)
**Описание:** Test that strain tensor has 7D structure.

### gravity_params(self)
**Описание:** Create gravitational parameters.
**Декораторы:** <ast.Attribute object at 0x753e6fb634d0>

### mock_system(self, domain_7d)
**Описание:** Create mock system with phase field.
**Декораторы:** <ast.Attribute object at 0x753e6fb54c90>

### test_vbp_gravitational_effects_integration(self, mock_system, gravity_params)
**Описание:** Test integration of all VBP gravitational effects.

### test_physical_consistency(self, mock_system, gravity_params)
**Описание:** Test physical consistency of VBP gravitational effects.

### __init__(self, domain)
**Описание:** Нет докстринга

## ./tests/unit/test_level_g_astrophysics.py
Methods: 20

### test_star_model_initialization(self)
**Описание:** Test star model initialization.

### test_galaxy_model_initialization(self)
**Описание:** Test galaxy model initialization.

### test_black_hole_model_initialization(self)
**Описание:** Test black hole model initialization.

### test_star_phase_profile_creation(self)
**Описание:** Test star phase profile creation.

### test_galaxy_phase_profile_creation(self)
**Описание:** Test galaxy phase profile creation.

### test_black_hole_phase_profile_creation(self)
**Описание:** Test black hole phase profile creation.

### test_phase_properties_analysis(self)
**Описание:** Test phase properties analysis.

### test_observable_properties_computation(self)
**Описание:** Test observable properties computation.

### test_star_model_creation(self)
**Описание:** Test star model creation.

### test_galaxy_model_creation(self)
**Описание:** Test galaxy model creation.

### test_black_hole_model_creation(self)
**Описание:** Test black hole model creation.

### test_phase_correlation_length_computation(self)
**Описание:** Test phase correlation length computation.

### test_effective_radius_computation(self)
**Описание:** Test effective radius computation.

### test_phase_energy_computation(self)
**Описание:** Test phase energy computation.

### test_defect_density_computation(self)
**Описание:** Test defect density computation.

### test_star_phase_profile_physical_properties(self)
**Описание:** Test star phase profile physical properties.

### test_galaxy_spiral_structure(self)
**Описание:** Test galaxy spiral structure.

### test_black_hole_singularity_behavior(self)
**Описание:** Test black hole singularity behavior.

### test_phase_field_energy_conservation(self)
**Описание:** Test phase field energy conservation.

### test_topological_charge_conservation(self)
**Описание:** Test topological charge conservation.

## ./tests/unit/test_level_g_cosmology.py
Methods: 16

### test_metric_initialization(self)
**Описание:** Test metric initialization.

### test_scale_factors_computation(self)
**Описание:** Test scale factors computation.

### test_metric_tensor_computation(self)
**Описание:** Test metric tensor computation.

### test_model_initialization(self)
**Описание:** Test model initialization.

### test_universe_evolution(self)
**Описание:** Test universe evolution.

### test_structure_formation_analysis(self)
**Описание:** Test structure formation analysis.

### test_cosmological_parameters_computation(self)
**Описание:** Test cosmological parameters computation.

### test_phase_field_initialization(self)
**Описание:** Test phase field initialization.

### test_phase_field_evolution_step(self)
**Описание:** Test phase field evolution step.

### test_structure_analysis_at_time(self)
**Описание:** Test structure analysis at specific time.

### test_correlation_length_computation(self)
**Описание:** Test correlation length computation.

### test_topological_defects_counting(self)
**Описание:** Test topological defects counting.

### test_structure_growth_rate_computation(self)
**Описание:** Test structure growth rate computation.

### test_parameter_evolution_consistency(self)
**Описание:** Test parameter evolution consistency.

### test_energy_conservation(self)
**Описание:** Test energy conservation.

### test_cosmological_parameters_physical_meaning(self)
**Описание:** Test cosmological parameters physical meaning.

## ./tests/unit/test_level_g_validation.py
Methods: 16

### test_inversion_initialization(self)
**Описание:** Test inversion initialization.

### test_parameter_initialization(self)
**Описание:** Test parameter initialization.

### test_loss_function_computation(self)
**Описание:** Test loss function computation.

### test_model_predictions_computation(self)
**Описание:** Test model predictions computation.

### test_distance_metric_computation(self)
**Описание:** Test distance metric computation.

### test_regularization_computation(self)
**Описание:** Test regularization computation.

### test_gradients_computation(self)
**Описание:** Test gradients computation.

### test_parameter_uncertainties_computation(self)
**Описание:** Test parameter uncertainties computation.

### test_inversion_optimization(self)
**Описание:** Test inversion optimization.

### test_validation_initialization(self)
**Описание:** Test validation initialization.

### test_parameter_validation(self)
**Описание:** Test parameter validation.

### test_energy_balance_validation(self)
**Описание:** Test energy balance validation.

### test_physical_constraint_validation(self)
**Описание:** Test physical constraint validation.

### test_experimental_validation(self)
**Описание:** Test experimental validation.

### test_overall_validation(self)
**Описание:** Test overall validation.

### test_full_validation_process(self)
**Описание:** Test full validation process.

## ./tests/unit/test_models/test_level_b_power_law_core_fixes.py
Methods: 13

### domain_3d(self)
**Описание:** Create 3D domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb06490>

### bvp_core(self, domain_3d)
**Описание:** Create BVP core for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fbc1350>

### power_law_core(self, bvp_core)
**Описание:** Create power law core for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fbc2410>

### test_envelope_3d(self, domain_3d)
**Описание:** Create test 3D envelope field.
**Декораторы:** <ast.Attribute object at 0x753e6fa22950>

### test_7d_correlation_function_implementation(self, power_law_core, test_envelope_3d)
**Описание:** Test that 7D correlation function is properly implemented.

Physical Meaning:
    Verifies that the correlation function preserves 7D structure
    and computes proper spatial correlations C(r) = ⟨a(x...

### test_full_critical_exponents_implementation(self, power_law_core, test_envelope_3d)
**Описание:** Test that full critical exponents are properly implemented.

Physical Meaning:
    Verifies that all standard critical exponents are computed:
    ν (correlation length), β (order parameter), γ (susce...

### test_full_scaling_regions_implementation(self, power_law_core, test_envelope_3d)
**Описание:** Test that full scaling regions analysis is properly implemented.

Physical Meaning:
    Verifies that scaling regions are identified using multi-scale
    decomposition, wavelet analysis, and renormal...

### test_mathematical_correctness_correlation_function(self, power_law_core, test_envelope_3d)
**Описание:** Test mathematical correctness of 7D correlation function.

Mathematical Foundation:
    Verifies that C(r) = ⟨a(x)a(x+r)⟩ is computed correctly
    and satisfies mathematical properties of correlation...

### test_mathematical_correctness_critical_exponents(self, power_law_core, test_envelope_3d)
**Описание:** Test mathematical correctness of critical exponents.

Mathematical Foundation:
    Verifies that critical exponents satisfy scaling relations
    and are computed using proper statistical methods.

### test_physical_meaning_preservation(self, power_law_core, test_envelope_3d)
**Описание:** Test that physical meaning is preserved in corrected implementations.

Physical Meaning:
    Verifies that the corrected methods maintain the physical
    interpretation of the 7D BVP field analysis.

### test_performance_and_stability(self, power_law_core, test_envelope_3d)
**Описание:** Test that corrected implementations are stable and performant.

Physical Meaning:
    Verifies that the full implementations are numerically stable
    and perform within reasonable time limits.

### test_backward_compatibility(self, power_law_core, test_envelope_3d)
**Описание:** Test that corrected implementations maintain backward compatibility.

Physical Meaning:
    Verifies that the corrected methods can still be used
    in existing code without breaking changes.

### check_finite_recursive(obj, path)
**Описание:** Нет докстринга

## ./tests/unit/test_solvers/test_abstract_solver_comprehensive.py
Methods: 26

### __init__(self, domain, parameters)
**Описание:** Initialize concrete solver.

### solve(self, source)
**Описание:** Solve the phase field equation.

### solve_time_evolution(self, initial_field, source, time_steps, dt)
**Описание:** Solve time evolution of the phase field.

### domain(self)
**Описание:** Create domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa53e50>

### parameters(self)
**Описание:** Create parameters for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa50a90>

### solver(self, domain, parameters)
**Описание:** Create concrete solver for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fb76090>

### test_solver_initialization(self, solver, domain, parameters)
**Описание:** Test solver initialization.

### test_solver_solve(self, solver)
**Описание:** Test solver solve method.

### test_solver_solve_time_evolution(self, solver)
**Описание:** Test solver time evolution method.

### test_solver_validate_input(self, solver)
**Описание:** Test input validation.

### test_solver_compute_residual(self, solver)
**Описание:** Test residual computation.

### test_solver_get_energy(self, solver)
**Описание:** Test energy computation.

### test_solver_is_initialized(self, solver)
**Описание:** Test initialization status.

### test_solver_repr(self, solver)
**Описание:** Test solver string representation.

### test_solver_abstract_methods(self)
**Описание:** Test that abstract methods raise NotImplementedError.

### test_solver_residual_physics(self, solver)
**Описание:** Test residual computation physics.

### test_solver_energy_physics(self, solver)
**Описание:** Test energy computation physics.

### test_solver_spectral_coefficients(self, solver)
**Описание:** Test spectral coefficients computation.

### test_solver_domain_properties(self, solver)
**Описание:** Test domain properties access.

### test_solver_parameters_properties(self, solver)
**Описание:** Test parameters properties access.

### test_solver_fft_operations(self, solver)
**Описание:** Test FFT operations in residual computation.

### test_solver_energy_conservation(self, solver)
**Описание:** Test energy conservation properties.

### test_solver_time_evolution_properties(self, solver)
**Описание:** Test time evolution properties.

### test_solver_error_handling(self, solver)
**Описание:** Test error handling.

### test_solver_numerical_stability(self, solver)
**Описание:** Test numerical stability.

### test_solver_large_values(self, solver)
**Описание:** Test with large values.

## ./tests/unit/test_solvers/test_time_integrator_comprehensive.py
Methods: 33

### __init__(self, domain, config, bvp_core)
**Описание:** Initialize concrete time integrator.

### step(self, field, dt)
**Описание:** Perform one time step.

### get_integrator_type(self) -> str
**Описание:** Get integrator type.

### domain(self)
**Описание:** Create domain for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa21d50>

### config(self)
**Описание:** Create config for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fa211d0>

### mock_bvp_core(self)
**Описание:** Create mock BVP core.
**Декораторы:** <ast.Attribute object at 0x753e6fccee10>

### integrator(self, domain, config)
**Описание:** Create time integrator for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fcfc250>

### integrator_with_bvp(self, domain, config, mock_bvp_core)
**Описание:** Create time integrator with BVP core for testing.
**Декораторы:** <ast.Attribute object at 0x753e6fcfe410>

### test_integrator_initialization(self, integrator, domain, config)
**Описание:** Test integrator initialization.

### test_integrator_initialization_with_bvp(self, integrator_with_bvp, domain, config, mock_bvp_core)
**Описание:** Test integrator initialization with BVP core.

### test_integrator_step(self, integrator)
**Описание:** Test integrator step method.

### test_integrator_get_integrator_type(self, integrator)
**Описание:** Test integrator type retrieval.

### test_integrator_get_domain(self, integrator, domain)
**Описание:** Test domain retrieval.

### test_integrator_get_config(self, integrator, config)
**Описание:** Test config retrieval.

### test_integrator_detect_quenches_with_detector(self, integrator_with_bvp)
**Описание:** Test quench detection with detector.

### test_integrator_detect_quenches_without_detector(self, integrator)
**Описание:** Test quench detection without detector.

### test_integrator_get_bvp_core(self, integrator_with_bvp, mock_bvp_core)
**Описание:** Test BVP core retrieval.

### test_integrator_get_bvp_core_none(self, integrator)
**Описание:** Test BVP core retrieval when None.

### test_integrator_set_bvp_core(self, integrator, mock_bvp_core)
**Описание:** Test BVP core setting.

### test_integrator_set_bvp_core_none(self, integrator_with_bvp)
**Описание:** Test BVP core setting to None.

### test_integrator_repr(self, integrator)
**Описание:** Test integrator string representation.

### test_integrator_abstract_methods(self)
**Описание:** Test that abstract methods raise NotImplementedError.

### test_integrator_step_physics(self, integrator)
**Описание:** Test integrator step physics.

### test_integrator_quench_detection_physics(self, integrator_with_bvp)
**Описание:** Test quench detection physics.

### test_integrator_config_handling(self, domain)
**Описание:** Test config handling.

### test_integrator_domain_properties(self, integrator, domain)
**Описание:** Test domain properties access.

### test_integrator_error_handling(self, integrator)
**Описание:** Test error handling.

### test_integrator_numerical_stability(self, integrator)
**Описание:** Test numerical stability.

### test_integrator_large_dt(self, integrator)
**Описание:** Test with large dt.

### test_integrator_quench_detector_initialization(self, domain, config)
**Описание:** Test quench detector initialization.

### test_integrator_quench_detector_without_config(self, domain)
**Описание:** Test quench detector without config.

### test_integrator_bvp_core_integration(self, domain, config, mock_bvp_core)
**Описание:** Test BVP core integration.

### test_integrator_config_copy(self, integrator, config)
**Описание:** Test that config is copied.

## Сводка
Всего методов: 4092
Файлов с методами: 386

### Топ файлов по количеству методов:
- ./bhlff/testing/automated_reporting.py: 55 методов
- ./tests/unit/test_core/test_bvp_constants_comprehensive.py: 41 методов
- ./tests/unit/test_core/test_solvers_coverage.py: 38 методов
- ./tests/unit/test_solvers/test_time_integrator_comprehensive.py: 33 методов
- ./tests/unit/test_core/test_domain_comprehensive.py: 32 методов
- ./bhlff/testing/automated_testing.py: 32 методов
- ./bhlff/testing/quality_monitor.py: 32 методов
- ./bhlff/models/level_f/nonlinear.py: 30 методов
- ./tests/unit/test_level_f/test_nonlinear.py: 29 методов
- ./bhlff/models/level_f/multi_particle.py: 29 методов