"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

CLI command for analyzing results.

This module provides the CLI command for analyzing phase field theory
simulation results and extracting physical quantities.

Physical Meaning:
    Analyzes simulation results to extract physical quantities such as
    energy, topological charge, phase coherence, and other relevant
    metrics for the 7D phase field theory.

Example:
    >>> analyze_results("output/results.h5", output_path="analysis.json")
"""

import click
from pathlib import Path
from typing import Optional, List

from bhlff.utils.logging import StructuredLogger


def analyze_results(
    input_path: Path,
    output_path: Optional[Path] = None,
    output_format: str = "json",
    metrics: Optional[List[str]] = None,
    verbose: bool = False,
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

    Args:
        input_path (Path): Path to input data file.
        output_path (Optional[Path]): Path to output analysis file.
        output_format (str): Output format (json, hdf5, csv).
        metrics (Optional[List[str]]): Specific metrics to compute.
        verbose (bool): If True, enable verbose output.
    """
    # Setup logging
    logger = StructuredLogger("result_analyzer")

    if verbose:
        click.echo(f"Analyzing results from {input_path}")

    # Load input data
    data = load_simulation_data(input_path, verbose)

    # Determine output path
    if output_path is None:
        output_path = input_path.parent / f"analysis.{output_format}"

    # Set default metrics if none specified
    if metrics is None:
        metrics = [
            "energy",
            "topological_charge",
            "phase_coherence",
            "field_statistics",
            "spectral_analysis",
        ]

    # Perform analysis
    results = {}

    for metric in metrics:
        if verbose:
            click.echo(f"Computing {metric}...")

        try:
            if metric == "energy":
                results[metric] = compute_energy_analysis(data)
            elif metric == "topological_charge":
                results[metric] = compute_topological_charge_analysis(data)
            elif metric == "phase_coherence":
                results[metric] = compute_phase_coherence_analysis(data)
            elif metric == "field_statistics":
                results[metric] = compute_field_statistics(data)
            elif metric == "spectral_analysis":
                results[metric] = compute_spectral_analysis(data)
            else:
                click.echo(f"Unknown metric: {metric}", err=True)
                continue

        except Exception as e:
            click.echo(f"Error computing {metric}: {e}", err=True)
            logger.log_error(
                "analysis_error", f"Failed to compute {metric}", {"error": str(e)}
            )
            continue

    # Save results
    save_analysis_results(results, output_path, output_format, verbose)

    if verbose:
        click.echo(f"Analysis completed. Results saved to {output_path}")


def load_simulation_data(input_path: Path, verbose: bool) -> dict:
    """
    Load simulation data from file.

    Physical Meaning:
        Loads phase field simulation data from various file formats,
        including field data, parameters, and metadata.

    Args:
        input_path (Path): Path to input file.
        verbose (bool): Verbose output flag.

    Returns:
        dict: Loaded simulation data.
    """
    import h5py
    import numpy as np

    if verbose:
        click.echo(f"Loading data from {input_path}")

    if input_path.suffix == ".h5" or input_path.suffix == ".hdf5":
        with h5py.File(input_path, "r") as f:
            data = {}
            for key in f.keys():
                data[key] = f[key][:]
    elif input_path.suffix == ".npz":
        data = np.load(input_path)
        data = {key: data[key] for key in data.keys()}
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    return data


def compute_energy_analysis(data: dict) -> dict:
    """
    Compute energy analysis.

    Physical Meaning:
        Computes energy-related metrics from the phase field data,
        including total energy, energy density, and energy evolution.

    Returns:
        dict: Energy analysis results.
    """
    import numpy as np

    # Extract field data
    if "field" in data:
        field = data["field"]
    elif "phase_field" in data:
        field = data["phase_field"]
    else:
        raise ValueError("No field data found in input")

    # Compute energy metrics
    energy_density = np.abs(field) ** 2
    total_energy = np.sum(energy_density)

    # Compute energy statistics
    energy_stats = {
        "total_energy": float(total_energy),
        "mean_energy_density": float(np.mean(energy_density)),
        "max_energy_density": float(np.max(energy_density)),
        "min_energy_density": float(np.min(energy_density)),
        "std_energy_density": float(np.std(energy_density)),
    }

    return energy_stats


def compute_topological_charge_analysis(data: dict) -> dict:
    """
    Compute topological charge analysis.

    Physical Meaning:
        Computes topological charge and related metrics from the
        phase field data, identifying topological defects.

    Returns:
        dict: Topological charge analysis results.
    """
    import numpy as np

    # Extract field data
    if "field" in data:
        field = data["field"]
    elif "phase_field" in data:
        field = data["phase_field"]
    else:
        raise ValueError("No field data found in input")

    # Compute phase
    phase = np.angle(field)

    # Compute topological charge (simplified)
    # This is a basic implementation - full implementation would
    # use proper line integration
    charge_density = np.gradient(phase)
    total_charge = np.sum(charge_density)

    charge_stats = {
        "total_charge": float(total_charge),
        "mean_charge_density": float(np.mean(charge_density)),
        "max_charge_density": float(np.max(charge_density)),
        "min_charge_density": float(np.min(charge_density)),
    }

    return charge_stats


def compute_phase_coherence_analysis(data: dict) -> dict:
    """
    Compute phase coherence analysis.

    Physical Meaning:
        Computes phase coherence metrics from the phase field data,
        measuring the degree of phase alignment across the field.

    Returns:
        dict: Phase coherence analysis results.
    """
    import numpy as np

    # Extract field data
    if "field" in data:
        field = data["field"]
    elif "phase_field" in data:
        field = data["phase_field"]
    else:
        raise ValueError("No field data found in input")

    # Compute phase coherence
    phase = np.angle(field)
    coherence = np.abs(np.mean(np.exp(1j * phase)))

    # Compute phase statistics
    phase_stats = {
        "phase_coherence": float(coherence),
        "mean_phase": float(np.mean(phase)),
        "std_phase": float(np.std(phase)),
        "phase_range": float(np.max(phase) - np.min(phase)),
    }

    return phase_stats


def compute_field_statistics(data: dict) -> dict:
    """
    Compute field statistics.

    Physical Meaning:
        Computes basic statistical properties of the phase field,
        including amplitude and phase distributions.

    Returns:
        dict: Field statistics.
    """
    import numpy as np

    # Extract field data
    if "field" in data:
        field = data["field"]
    elif "phase_field" in data:
        field = data["phase_field"]
    else:
        raise ValueError("No field data found in input")

    # Compute amplitude and phase
    amplitude = np.abs(field)
    phase = np.angle(field)

    # Compute statistics
    stats = {
        "amplitude": {
            "mean": float(np.mean(amplitude)),
            "std": float(np.std(amplitude)),
            "min": float(np.min(amplitude)),
            "max": float(np.max(amplitude)),
        },
        "phase": {
            "mean": float(np.mean(phase)),
            "std": float(np.std(phase)),
            "min": float(np.min(phase)),
            "max": float(np.max(phase)),
        },
        "field_shape": list(field.shape),
        "field_dtype": str(field.dtype),
    }

    return stats


def compute_spectral_analysis(data: dict) -> dict:
    """
    Compute spectral analysis.

    Physical Meaning:
        Computes spectral properties of the phase field, including
        power spectrum and dominant frequencies.

    Returns:
        dict: Spectral analysis results.
    """
    import numpy as np

    # Extract field data
    if "field" in data:
        field = data["field"]
    elif "phase_field" in data:
        field = data["phase_field"]
    else:
        raise ValueError("No field data found in input")

    # Compute FFT
    field_fft = np.fft.fftn(field)
    power_spectrum = np.abs(field_fft) ** 2

    # Compute spectral statistics
    spectral_stats = {
        "total_power": float(np.sum(power_spectrum)),
        "mean_power": float(np.mean(power_spectrum)),
        "max_power": float(np.max(power_spectrum)),
        "spectral_centroid": float(
            np.sum(power_spectrum * np.arange(len(power_spectrum)))
            / np.sum(power_spectrum)
        ),
    }

    return spectral_stats


def save_analysis_results(
    results: dict, output_path: Path, output_format: str, verbose: bool
) -> None:
    """
    Save analysis results to file.

    Physical Meaning:
        Saves analysis results to file in the specified format
        for further processing and visualization.

    Args:
        results (dict): Analysis results.
        output_path (Path): Output file path.
        output_format (str): Output format.
        verbose (bool): Verbose output flag.
    """
    if verbose:
        click.echo(f"Saving results to {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        import json

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    elif output_format == "hdf5":
        import h5py

        with h5py.File(output_path, "w") as f:
            for key, value in results.items():
                if isinstance(value, dict):
                    group = f.create_group(key)
                    for subkey, subvalue in value.items():
                        group.attrs[subkey] = subvalue
                else:
                    f.attrs[key] = value
    elif output_format == "csv":
        import pandas as pd

        # Flatten results for CSV
        flattened = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flattened[f"{key}_{subkey}"] = subvalue
            else:
                flattened[key] = value

        df = pd.DataFrame([flattened])
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
