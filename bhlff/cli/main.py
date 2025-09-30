"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Main CLI interface for BHLFF.

This module provides the main command-line interface for the 7D phase field
theory implementation, including experiment running, analysis, and reporting.

Physical Meaning:
    CLI interface provides access to all functionality of the 7D phase field
    theory implementation, enabling users to run experiments, analyze results,
    and generate reports from the command line.

Example:
    >>> bhlff --help
    >>> bhlff run --config configs/default.json
    >>> bhlff analyze --input output/results.h5
"""

import click
import sys
from pathlib import Path
from typing import Optional

from .run import run_experiment
from .analyze import analyze_results
from .report import generate_report


@click.group()
@click.version_option(version="0.1.0")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration file path",
)
@click.pass_context
def main(ctx: click.Context, verbose: bool, config: Optional[Path]) -> None:
    """
    BHLFF: 7D Phase Field Theory Implementation.

    Physical Meaning:
        Main CLI interface for the 7D phase field theory implementation,
        providing access to all simulation, analysis, and reporting
        functionality.

    Mathematical Foundation:
        Implements the fractional Riesz operator L_β = μ(-Δ)^β + λ
        and related equations for 7D phase field theory.
    """
    # Ensure that ctx.obj exists and is a dict
    ctx.ensure_object(dict)

    # Store global options
    ctx.obj["verbose"] = verbose
    ctx.obj["config"] = config

    if verbose:
        click.echo("BHLFF: 7D Phase Field Theory Implementation")
        click.echo("Verbose mode enabled")


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Configuration file path",
)
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), help="Output directory path"
)
@click.option(
    "--level",
    "-l",
    type=click.Choice(["a", "b", "c", "d", "e", "f", "g"]),
    help="Theory level to run",
)
@click.option("--dry-run", is_flag=True, help="Perform a dry run without executing")
@click.pass_context
def run(
    ctx: click.Context,
    config: Path,
    output: Optional[Path],
    level: Optional[str],
    dry_run: bool,
) -> None:
    """
    Run phase field theory experiments.

    Physical Meaning:
        Executes phase field theory simulations according to the
        specified configuration and theory level.

    Mathematical Foundation:
        Solves the fractional Riesz equation L_β a = s(x,t) with
        appropriate boundary conditions and initial conditions.
    """
    try:
        run_experiment(
            config_path=config,
            output_dir=output,
            level=level,
            dry_run=dry_run,
            verbose=ctx.obj.get("verbose", False),
        )
    except Exception as e:
        click.echo(f"Error running experiment: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input data file path",
)
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), help="Output analysis file path"
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "hdf5", "csv"]),
    default="json",
    help="Output format",
)
@click.option("--metrics", multiple=True, help="Specific metrics to compute")
@click.pass_context
def analyze(
    ctx: click.Context, input: Path, output: Optional[Path], format: str, metrics: tuple
) -> None:
    """
    Analyze phase field theory results.

    Physical Meaning:
        Analyzes simulation results to extract physical quantities
        such as energy, topological charge, and phase coherence.

    Mathematical Foundation:
        Computes various metrics from the phase field data including
        energy functionals, topological invariants, and statistical
        properties.
    """
    try:
        analyze_results(
            input_path=input,
            output_path=output,
            output_format=format,
            metrics=list(metrics) if metrics else None,
            verbose=ctx.obj.get("verbose", False),
        )
    except Exception as e:
        click.echo(f"Error analyzing results: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input analysis file path",
)
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), help="Output report file path"
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["html", "pdf", "latex"]),
    default="html",
    help="Report format",
)
@click.option(
    "--template",
    "-t",
    type=click.Path(exists=True, path_type=Path),
    help="Custom template file",
)
@click.pass_context
def report(
    ctx: click.Context,
    input: Path,
    output: Optional[Path],
    format: str,
    template: Optional[Path],
) -> None:
    """
    Generate analysis reports.

    Physical Meaning:
        Generates comprehensive reports from analysis results,
        including visualizations and statistical summaries.

    Mathematical Foundation:
        Creates reports with mathematical formulations, results
        analysis, and physical interpretations of the data.
    """
    try:
        generate_report(
            input_path=input,
            output_path=output,
            report_format=format,
            template_path=template,
            verbose=ctx.obj.get("verbose", False),
        )
    except Exception as e:
        click.echo(f"Error generating report: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration file to validate",
)
@click.option(
    "--schema",
    "-s",
    type=click.Path(exists=True, path_type=Path),
    help="Schema file for validation",
)
def validate(config: Optional[Path], schema: Optional[Path]) -> None:
    """
    Validate configuration files.

    Physical Meaning:
        Validates configuration files to ensure they contain
        physically reasonable parameters and are properly formatted.

    Mathematical Foundation:
        Checks that all parameters are within valid ranges and
        that the configuration is consistent with the mathematical
        requirements of the theory.
    """
    try:
        from bhlff.utils.config.loader import ConfigLoader

        if config:
            loader = ConfigLoader()
            config_data = loader.load_config(config)
            click.echo(f"Configuration file {config} is valid")
            click.echo(f"Loaded configuration with {len(config_data)} sections")
        else:
            click.echo("No configuration file specified")
            sys.exit(1)

    except Exception as e:
        click.echo(f"Configuration validation failed: {e}", err=True)
        sys.exit(1)


@main.command()
def info() -> None:
    """
    Display system information.

    Physical Meaning:
        Displays information about the BHLFF installation,
        including version, dependencies, and system capabilities.
    """
    try:
        import bhlff
        import numpy as np
        import scipy
        import matplotlib

        click.echo("BHLFF: 7D Phase Field Theory Implementation")
        click.echo(
            f"Version: {bhlff.__version__ if hasattr(bhlff, '__version__') else '0.1.0'}"
        )
        click.echo(f"NumPy version: {np.__version__}")
        click.echo(f"SciPy version: {scipy.__version__}")
        click.echo(f"Matplotlib version: {matplotlib.__version__}")

        # Check for optional dependencies
        try:
            import pyfftw

            click.echo(f"PyFFTW version: {pyfftw.__version__}")
        except ImportError:
            click.echo("PyFFTW: not available")

        try:
            import numba

            click.echo(f"Numba version: {numba.__version__}")
        except ImportError:
            click.echo("Numba: not available")

    except Exception as e:
        click.echo(f"Error getting system info: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
