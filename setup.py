"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Setup script for BHLFF package.

This module provides the setup configuration for the BHLFF package,
which implements the 7D phase field theory for elementary particles.

The setup script handles package installation, dependencies, and
metadata configuration for PyPI distribution.

Physical Meaning:
    The setup script ensures proper installation of the BHLFF package
    with all necessary dependencies for computational physics simulations
    of phase field theory in 7D space-time.

Example:
    pip install -e .
    pip install -e .[dev]
    pip install -e .[docs,visualization]
"""

import os
import sys
from pathlib import Path

from setuptools import setup, find_packages

# Read the README file
def read_readme():
    """Read the README file for long description."""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return "7D Phase Field Theory Implementation for Elementary Particles"

# Read version from __init__.py
def get_version():
    """Get version from bhlff/__init__.py."""
    init_path = Path(__file__).parent / "bhlff" / "__init__.py"
    if init_path.exists():
        with open(init_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

# Check Python version
if sys.version_info < (3, 9):
    sys.exit("BHLFF requires Python 3.9 or higher")

# Get package data
def get_package_data():
    """Get package data files."""
    package_data = {
        "bhlff": [
            "configs/*.json",
            "configs/**/*.json",
            "data/*.h5",
            "data/*.npz",
            "templates/*.html",
            "templates/*.tex",
        ]
    }
    return package_data

# Get entry points
def get_entry_points():
    """Get console script entry points."""
    return {
        "console_scripts": [
            "bhlff=bhlff.cli.main:main",
            "bhlff-run=bhlff.cli.run:main",
            "bhlff-analyze=bhlff.cli.analyze:main",
            "bhlff-report=bhlff.cli.report:main",
        ]
    }

# Main setup configuration
setup(
    name="bhlff",
    version=get_version(),
    author="Vasiliy Zdanovskiy",
    author_email="vasilyvz@gmail.com",
    maintainer="Vasiliy Zdanovskiy",
    maintainer_email="vasilyvz@gmail.com",
    description="7D Phase Field Theory Implementation for Elementary Particles",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/vasilyvz/bhlff",
    project_urls={
        "Homepage": "https://github.com/vasilyvz/bhlff",
        "Documentation": "https://bhlff.readthedocs.io",
        "Repository": "https://github.com/vasilyvz/bhlff.git",
        "Issues": "https://github.com/vasilyvz/bhlff/issues",
        "Changelog": "https://github.com/vasilyvz/bhlff/blob/main/CHANGELOG.md",
    },
    packages=find_packages(),
    package_data=get_package_data(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "h5py>=3.1.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "click>=8.0.0",
        "rich>=12.0.0",
        "pytest>=6.2.0",
        "pytest-cov>=3.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.950",
        "isort>=5.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "pytest-xdist>=2.4.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "isort>=5.10.0",
            "pre-commit>=2.17.0",
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0",
            "sphinx-autodoc-typehints>=1.12.0",
            "sphinx-copybutton>=0.5.0",
        ],
        "visualization": [
            "plotly>=5.0.0",
            "mayavi>=4.7.0",
            "vtk>=9.1.0",
            "pyvista>=0.35.0",
        ],
        "performance": [
            "numba>=0.56.0",
            "cupy-cuda11x>=10.0.0; platform_machine=='x86_64'",
            "cupy-cuda12x>=12.0.0; platform_machine=='x86_64'",
        ],
    },
    entry_points=get_entry_points(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "phase field theory",
        "fractional calculus",
        "elementary particles",
        "7d physics",
        "topological defects",
        "solitons",
        "computational physics",
    ],
    license="MIT",
    zip_safe=False,
    platforms=["any"],
)
