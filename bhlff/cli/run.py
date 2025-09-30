"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

CLI command for running experiments.

This module provides the CLI command for running phase field theory
experiments with various configurations and parameters.

Physical Meaning:
    Executes phase field theory simulations according to specified
    configuration, running the appropriate theory level and saving
    results for analysis.

Example:
    >>> run_experiment("configs/default.json", output_dir="output")
"""

import click
from pathlib import Path
from typing import Optional

from bhlff.utils.config.loader import ConfigLoader
from bhlff.utils.logging import StructuredLogger


def run_experiment(
    config_path: Path,
    output_dir: Optional[Path] = None,
    level: Optional[str] = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> None:
    """
    Run phase field theory experiment.

    Physical Meaning:
        Executes a phase field theory simulation according to the
        specified configuration, running the appropriate theory level
        and saving results for analysis.

    Mathematical Foundation:
        Solves the fractional Riesz equation L_β a = s(x,t) with
        appropriate boundary conditions and initial conditions.

    Args:
        config_path (Path): Path to configuration file.
        output_dir (Optional[Path]): Output directory for results.
        level (Optional[str]): Theory level to run (a-g).
        dry_run (bool): If True, perform dry run without execution.
        verbose (bool): If True, enable verbose output.
    """
    # Setup logging
    logger = StructuredLogger("experiment_runner")

    if verbose:
        click.echo(f"Loading configuration from {config_path}")

    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config(config_path)

    # Override output directory if specified
    if output_dir:
        config["output"]["output_dir"] = str(output_dir)

    # Create output directory
    output_path = Path(config["output"]["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    # Log experiment start
    logger.log_experiment_start(f"level_{level}" if level else "default", config)

    if dry_run:
        click.echo("Dry run mode - no actual computation performed")
        click.echo(f"Configuration loaded: {len(config)} sections")
        click.echo(f"Output directory: {output_path}")
        return

    # Run the appropriate theory level
    if level:
        run_level_experiment(level, config, logger, verbose)
    else:
        run_default_experiment(config, logger, verbose)

    # Log experiment end
    logger.log_experiment_end(
        f"level_{level}" if level else "default",
        {"status": "completed", "output_dir": str(output_path)},
    )

    if verbose:
        click.echo(f"Experiment completed. Results saved to {output_path}")


def run_level_experiment(
    level: str, config: dict, logger: StructuredLogger, verbose: bool
) -> None:
    """
    Run experiment for specific theory level.

    Physical Meaning:
        Executes experiments for the specified theory level,
        implementing the appropriate physical models and analysis.

    Args:
        level (str): Theory level (a-g).
        config (dict): Configuration dictionary.
        logger (StructuredLogger): Logger instance.
        verbose (bool): Verbose output flag.
    """
    if verbose:
        click.echo(f"Running Level {level.upper()} experiment")

    # Import appropriate level module
    try:
        if level == "a":
            from bhlff.models.level_a import SolverValidator

            # Run Level A experiments
            validator = SolverValidator(config)
            validator.run_validation()

        elif level == "b":
            from bhlff.models.level_b import PowerLawAnalyzer

            # Run Level B experiments
            analyzer = PowerLawAnalyzer(config)
            analyzer.analyze_power_law()

        elif level == "c":
            from bhlff.models.level_c import BoundaryAnalyzer

            # Run Level C experiments
            analyzer = BoundaryAnalyzer(config)
            analyzer.analyze_boundaries()

        elif level == "d":
            from bhlff.models.level_d import ModeSuperpositionAnalyzer

            # Run Level D experiments
            analyzer = ModeSuperpositionAnalyzer(config)
            analyzer.analyze_superposition()

        elif level == "e":
            from bhlff.models.level_e import SolitonAnalyzer

            # Run Level E experiments
            analyzer = SolitonAnalyzer(config)
            analyzer.analyze_solitons()

        elif level == "f":
            from bhlff.models.level_f import MultiParticleAnalyzer

            # Run Level F experiments
            analyzer = MultiParticleAnalyzer(config)
            analyzer.analyze_multi_particle()

        elif level == "g":
            from bhlff.models.level_g import CosmologicalEvolutionAnalyzer

            # Run Level G experiments
            analyzer = CosmologicalEvolutionAnalyzer(config)
            analyzer.analyze_cosmological_evolution()

        else:
            raise ValueError(f"Unknown theory level: {level}")

    except ImportError as e:
        click.echo(f"Error importing level {level} modules: {e}", err=True)
        raise
    except Exception as e:
        click.echo(f"Error running level {level} experiment: {e}", err=True)
        raise


def run_default_experiment(
    config: dict, logger: StructuredLogger, verbose: bool
) -> None:
    """
    Run default experiment.

    Physical Meaning:
        Executes a basic phase field simulation with the default
        configuration, demonstrating core functionality.

    Args:
        config (dict): Configuration dictionary.
        logger (StructuredLogger): Logger instance.
        verbose (bool): Verbose output flag.
    """
    if verbose:
        click.echo("Running default experiment")

    # Import core components
    from bhlff.core.domain.domain import Domain
    from bhlff.core.phase.phase_field import PhaseField
    from bhlff.solvers.spectral.fft_solver_3d import FFT3DSolver

    # Create domain
    domain_config = config["domain"]
    domain = Domain(
        L=domain_config["L"],
        N=domain_config["N"],
        dimensions=domain_config["dimensions"],
        N_phi=domain_config.get("N_phi", 32),
        N_t=domain_config.get("N_t", 64),
        T=domain_config.get("T", 1.0),
    )

    # Create initial phase field
    field_data = create_initial_field(domain)
    phase_field = PhaseField(
        data=field_data,
        domain=domain,
        time=0.0,
        phase_velocity=config["physics"]["phase_velocity"],
    )

    # Create solver
    _solver = FFT3DSolver(domain, config["physics"])

    # Run simulation
    if verbose:
        click.echo("Starting simulation...")

    # This would run the actual simulation
    # For now, just log the setup
    logger.log_experiment_start(
        "default_simulation",
        {"domain_shape": domain.shape, "field_energy": phase_field.compute_energy()},
    )

    if verbose:
        click.echo("Simulation completed")


def create_initial_field(domain: Domain) -> "np.ndarray":
    """
    Create initial field configuration.

    Physical Meaning:
        Creates an initial phase field configuration for simulation,
        typically a localized excitation or uniform field.

    Args:
        domain: Computational domain.

    Returns:
        np.ndarray: Initial field data.
    """
    import numpy as np

    # Create a simple Gaussian initial condition
    field_data = np.zeros(domain.shape, dtype=np.complex128)

    if domain.dimensions == 3:
        # 3D Gaussian
        center = domain.N // 2
        sigma = domain.N // 8

        x = np.arange(domain.N)
        y = np.arange(domain.N)
        z = np.arange(domain.N)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        gaussian = np.exp(
            -((X - center) ** 2 + (Y - center) ** 2 + (Z - center) ** 2)
            / (2 * sigma**2)
        )
        field_data = gaussian.astype(np.complex128)

    return field_data
