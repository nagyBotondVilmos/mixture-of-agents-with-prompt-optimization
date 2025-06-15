# Mixture of Agents (MOA) Implementation

A Python implementation of the Mixture of Agents model for solving coding problems.

## Features

- **MOA Model**: Multi-layer neural network implementation using multiple AI agents
- **Single LLM Baseline**: Comparison with traditional single language model approach
- **Training Framework**: Comprehensive training system with evaluation metrics
- **API Server**: Flask-based server for model interaction
- **Result Analysis**: Tools for analyzing and visualizing model performance
- **Test Case Generation**: Automated test case generation for coding problems

## Installation

### From Source

1. Clone the repository
2. Install the package:

```bash
cd imp
```

```bash
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Set up your API keys

Copy `common/secret_example.py` to `common/secret.py` and edit it to include your OpenAI-compatible API keys (OpenAI, DeepSeek, other compatible API providers, etc.):

### 2. Start the API server and trainer

```bash
python trainer/api.py
```

```bash
python trainer/trainer.py -h
```

### 3. Compare models

```bash
python comparison/compare_models.py -h
```

### 4. Analyze results

```bash
python analysis/analyze_training_results.py -h
```

### Note
Don't use the `analyze_results.py` or `analyze_results_old.py` scripts, they are the old versions of the script.

## Package Structure

- `mymoa/`: Core MOA implementation
- `trainer/`: Training framework and API server
- `singlellm/`: Single LLM baseline implementation
- `common/`: Shared utilities and configuration
- `dataset/`: Dataset management utilities

## Dependencies

- `openai>=1.0.0`: For API interactions
- `flask>=2.0.0`: Web framework for API server
- `requests>=2.25.0`: HTTP client
- `numpy>=1.20.0`: Numerical computing
- `matplotlib>=3.3.0`: Plotting and visualization
- `tabulate>=0.8.0`: Table formatting

## Github

https://github.com/nagyBotondVilmos/mixture-of-agents-with-prompt-optimization
