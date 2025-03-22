# Correlation in Coincidence Data with Machine Learning

This repository provides Machine Learning (ML) methods to detect rare events showing particle entanglement, utilizing high repetition-rate light source technology and REaction MIcroscopy (REMI)

## Getting Started

To install this source code follow the next steps:
1. Clone this repository
2. In command line, go to the source code:
    ```sh
    cd CoInML
    ```
3. Create and activate a new conda environment:
    ```sh
    conda create --name my_env python=3.11 -y
    conda activate my_env
    ```
4. Install source code:
    ```sh
    pip install .
    ```

## Developer Setup

### Prerequisites
In addition to the steps listed above, please make sure to execute the following:

1. Install development dependencies:
   ```sh
   pip install ".[dev]"
   ```
3. Set up `pre-commit` in the repository:
   ```sh
   pre-commit install
   ```
   This ensures hooks run automatically before each commit.

### Running Pre-commit Hooks
- Run all hooks on all files:
  ```sh
  pre-commit run --all-files
  ```
- Run hooks only on changed files:
  ```sh
  pre-commit run
  ```

### Setting Up GGShield
GGShield is a tool for detecting secrets in your code before they get committed. To install and configure it:
1. Install GGShield:
   ```sh
   pip install ggshield
   ```
2. Authenticate with GitGuardian:
   ```sh
   ggshield auth login
   ```

### Common Issues
- **Hooks failing during commit**: Fix reported issues and re-stage files before committing.
- **Pre-commit not installed**: Run `pre-commit install`.
- **Skipping pre-commit checks** (use only if necessary):
  ```sh
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

### Additional Resources
- [Pre-commit Documentation](https://pre-commit.com/)
- [Git Hooks Documentation](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks)
