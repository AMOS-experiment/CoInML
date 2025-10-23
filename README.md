# SCULPT - Supervised Clustering and Uncovering Latent Patterns with Training

SCULPT is a comprehensive machine learning framework for analyzing high-dimensional coincidence spectroscopy data, particularly from COLTRIMS experiments.

## Features

- **Interactive UMAP visualization** for high-dimensional momentum data
- **Adaptive confidence scoring** for clustering validation
- **Genetic programming** for automated feature discovery
- **Deep autoencoders** for non-linear feature extraction
- **Flexible molecular configuration** system
- **Real-time filtering** and selection tools

## Requirements

- Python 3.8 or higher (3.11 recommended)
- 8GB RAM minimum (16GB recommended)
- Modern web browser

## Installation

### Quick Start

1. Clone this repository:
   ```bash
   git clone https://github.com/AMOS-experiment/CoInML
   cd sculpt
   ```

2. Create and activate a new conda environment:
   ```bash
   conda create --name sculpt_env python=3.11 -y
   conda activate sculpt_env
   ```

3. Install the package:
   ```bash
   pip install .
   ```

4. Run the application:
   ```bash
   python src/sculpt/app.py
   ```

5. Open your browser to:
   [http://localhost:9000](http://localhost:9000)

## Developer Setup

### Prerequisites
In addition to the basic installation steps above, developers should execute the following:

1. Install development dependencies:
   ```bash
   pip install ".[dev]"
   ```

2. Set up `pre-commit` in the repository:
   ```bash
   pre-commit install
   ```
   This ensures hooks run automatically before each commit.

### Running Pre-commit Hooks
- Run all hooks on all files:
  ```bash
  pre-commit run --all-files
  ```
- Run hooks only on changed files:
  ```bash
  pre-commit run
  ```

### Setting Up GGShield
GGShield is a tool for detecting secrets in your code before they get committed. To install and configure it:
1. Install GGShield:
   ```bash
   pip install ggshield
   ```
2. Authenticate with GitGuardian:
   ```bash
   ggshield auth login
   ```

### Common Issues
- **Hooks failing during commit**: Fix reported issues and re-stage files before committing.
- **Pre-commit not installed**: Run `pre-commit install`.
- **Skipping pre-commit checks** (use only if necessary):
  ```bash
  git commit --no-verify
  ```

### Configuration
Pre-commit hooks are defined in `.pre-commit-config.yaml`. Example:
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
```
Modify this file to add or remove hooks as needed.

## Usage

### 1. Data Upload

- Drag and drop CSV files containing momentum data
- Expected format: `Px, Py, Pz` for each particle

### 2. Configure Molecular System

- Select or create a molecular profile (e.g., D₂O, HDO)
- Assign profiles to uploaded files

### 3. Run Analysis

- Choose features for UMAP embedding
- Adjust parameters (`n_neighbors`, `min_dist`)
- Click **"Run UMAP"**

### 4. Explore Results

- Use lasso/box selection tools
- Apply filters (density, physics parameters)
- Export selected data

## Example Data

Example COLTRIMS datasets are provided in:
`data/examples/`

## Documentation

See the `docs/` directory for detailed documentation.

## Additional Resources
- [Pre-commit Documentation](https://pre-commit.com/)
- [Git Hooks Documentation](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks)

## Copyright Notice

SCULPT (Supervised Clustering and Uncovering Latent Patterns with Training) Copyright (c) 2025,
The Regents of the University of California, through Lawrence Berkeley National Laboratory
(“Berkeley Lab”) subject to receipt of any required approvals from the U.S. Dept. of Energy. All
rights reserved.

If you have questions about your rights to use, distribute this software, or use for commercial
purposes, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE: This Software was developed under Contract No. DE-AC02-05CH11231 with the
Department of Energy (“DOE”). During the period of commercialization or such other time period
specified by DOE, the U.S. Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable, worldwide license in the Software to reproduce, prepare
derivative works, and perform publicly and display publicly, by or on behalf of the U.S.
Government. Subsequent to that period, the U.S. Government is granted for itself and others
acting on its behalf a nonexclusive, paid-up, irrevocable, worldwide license in the Software to
reproduce, prepare derivative works, distribute copies to the public, perform publicly and display
publicly, and to permit others to do so. The specific term of the license can be identified by inquiry
made to Lawrence Berkeley National Laboratory or DOE. NEITHER THE UNITED STATES NOR THE
UNITED STATES DEPARTMENT OF ENERGY, NOR ANY OF THEIR EMPLOYEES, MAKES ANY
WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR
THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY DATA, APPARATUS, PRODUCT, OR
PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
RIGHTS.
