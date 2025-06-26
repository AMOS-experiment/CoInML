# SCULPT - Supervised Clustering and Uncovering Latent Patterns with Training

SCULPT is a comprehensive machine learning framework for analyzing high-dimensional coincidence spectroscopy data, particularly from COLTRIMS experiments.

## Features

- **Interactive UMAP visualization** for high-dimensional momentum data
- **Adaptive confidence scoring** for clustering validation
- **Genetic programming** for automated feature discovery
- **Deep autoencoders** for non-linear feature extraction
- **Flexible molecular configuration** system
- **Real-time filtering** and selection tools

## Installation

### Requirements
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- Modern web browser

### Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt

2. Run the application:
python src/app.py

3. Open your browser to http://localhost:9000

## Usage

1. Data Upload

Drag and drop CSV files containing momentum data
Expected format: Px, Py, Pz for each particle

2. Configure Molecular System

Select or create a molecular profile (e.g., D2O, HDO)
Assign profiles to uploaded files

3. Run Analysis

Choose features for UMAP embedding
Adjust parameters (n_neighbors, min_dist)
Click "Run UMAP"

4. Explore Results

Use lasso/box selection tools
Apply filters (density, physics parameters)
Export selected data

## Example Data
Example COLTRIMS datasets are provided in data/examples/

## Documentation
See the docs/ directory for detailed documentation.

## License
MIT License
