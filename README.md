# DELPHI: Dynamic Ensemble Leveraging Probabilistic Hybrid Inference - Fashion Trend Forecasting

A hybrid probabilistic forecasting framework that combines classical statistical models, deep learning ensembles, and latent-state inference to produce robust, interpretable, and uncertainty-aware predictions. Delphi is built specifically to handle the volatility of fashion trends and was trained on publicly available fashion item time series data. 

---

## Overview

DELPHI (Dynamic Ensemble Leveraging Probabilistic Hybrid Inference) is a modular system designed for complex, non-stationary time series forecasting.

The central idea is straightforward. Instead of relying on a single model, DELPHI combines a strong statistical baseline with adaptive neural models, then uses probabilistic inference to decide which components to trust at any point in time.

This allows the system to remain stable when patterns are predictable while still adapting when the data shifts.

DELPHI integrates the following components:

* Classical forecasting using TBATS
* Deep learning ensembles based on LSTM-style architectures
* Probabilistic regime detection with a variational Hidden Markov Model
* An explainability and uncertainty layer referred to as xAIUQ

This hybrid design improves both performance and transparency.

---

## Key Features

### Hybrid Forecasting

* Uses a TBATS baseline to capture seasonality and trend
* Applies neural ensemble corrections to improve accuracy
* Retains interpretability from the statistical model

### Probabilistic Gating

* Uses a variational Hidden Markov Model
* Learns latent regimes directly from the data
* Dynamically weights model contributions

### Deep Learning Core

* Ensemble-based recurrent architectures
* Captures nonlinear temporal structure
* Supports multi-step forecasting

### Explainability and Uncertainty

* SHAP-based feature attribution
* Confidence estimation through ensemble variance
* Regime-aware explanations

### Hardware-Aware Utilities

* GPU monitoring tools
* Memory-efficient batching
* CUDA-compatible execution

---

## Architecture

### Directory Structure

```
delphi/
├── data/                  # Raw and processed datasets
├── models/                # Model definitions (TBATS, neural nets, HMM)
├── ensemble/              # Ensemble logic
├── xai/                   # Explainability and uncertainty modules
├── utils/                 # Utilities (monitoring, logging)
├── config/                # Configuration files
├── evaluation_results/    # Metrics and outputs
├── explain.py             # Explanation entry point
├── predict.py             # Inference pipeline
└── train.py               # Training pipeline
```

---

### Data Flow

```
Raw Data
   ↓
Preprocessing
   ↓
TBATS Baseline
   ↓
HMM Regime Detection
   ↓
Ensemble Correction
   ↓
Final Prediction + Uncertainty
   ↓
Explanation Layer
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/millingtonsully/delphi.git
cd delphi
```

---

### 2. Create Environment

```bash
conda create -n delphi python=3.10
conda activate delphi
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Optional GPU Setup

* CUDA version 12.1 recommended
* Compatible with PyTorch GPU builds

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

### Development vs Production

| Mode        | Description                                |
| ----------- | ------------------------------------------ |
| Development | Includes debugging and visualization tools |
| Production  | Minimal dependencies for deployment        |

---

## Configuration Guide

Main configuration file:

```
config/delphi_config.yaml
```

---

### HMM Configuration

```yaml
hmm:
  num_states: 4
  transition_prior: 0.1
  emission_dim: 16
```

* num_states defines the number of latent regimes
* transition_prior regularizes switching behavior
* emission_dim sets embedding size

---

### Ensemble Configuration

```yaml
ensemble:
  members: 5
  hidden_size: 128
  num_layers: 2
```

* members controls ensemble size
* hidden_size defines model capacity
* num_layers determines depth

---

### Training Parameters

```yaml
training:
  epochs: 50
  batch_size: 64
  learning_rate: 0.001
  kl_weight: 0.01
```

* kl_weight controls the strength of the variational objective

---

## Usage

### Training Workflow

DELPHI uses a two-stage training process.

Stage 1 trains representations and establishes the baseline.

```bash
python train.py --stage 1
```

Stage 2 performs joint optimization across the HMM and ensemble.

```bash
python train.py --stage 2
```

---

### Inference

```bash
python predict.py --config config/delphi_config.yaml
```

---

### Sampling Predictions

```bash
python predict.py --sample 100
```

This produces probabilistic forecasts with uncertainty estimates.

---

### Generating Explanations

```bash
python explain.py --input sample_data.csv
```

Outputs include:

* Feature importance
* Regime attribution
* Uncertainty metrics

---

## Explainability Layer

The explainability system provides insight into both model behavior and confidence.

### Regime Detection

* Identifies shifts in underlying patterns
* Useful for detecting structural changes

### Feature Attribution

* Uses SHAP values
* Provides both local and global explanations

### Uncertainty Quantification

* Based on ensemble variation
* Produces prediction intervals
* Separates model uncertainty from data noise

---

## Monitoring and Utilities

### GPU Monitoring

```bash
python utils/gpu_monitor.py
```

Tracks:

* Memory usage
* GPU utilization
* Device performance

---

### Memory Management

* Adaptive batching
* Efficient tensor allocation
* Optional gradient checkpointing

---

## Results and Evaluation

Outputs are stored in:

```
evaluation_results/
```

Includes:

* RMSE, MAE, MAPE
* Calibration plots
* Forecast intervals

---

### Interpreting Results

* Lower RMSE indicates better accuracy
* Narrow intervals indicate higher confidence
* Stable regimes suggest consistent behavior

---

## Technical Specifications

### Frameworks

* PyTorch 2.0 or newer
* NumPy
* Pandas
* SHAP

---

### Hardware Requirements

| Component | Minimum  | Recommended               |
| --------- | -------- | ------------------------- |
| CPU       | 4 cores  | 8 or more cores           |
| RAM       | 8 GB     | 32 GB                     |
| GPU       | Optional | NVIDIA GPU with CUDA 12.1 |

---

## License

This project is licensed under the MIT License. See the repository for full license text.

---

## Future Work

* Integration with transformer-based models
* Online and streaming learning support
* Distributed training capabilities
* Extended causal inference modules
* Cost-efficient explainability layer

---

## Contact

millingtonsully@gmail.com, [sullymillington.com](https://www.sullymillington.com/)

---

DELPHI combines statistical structure with adaptive learning to produce forecasts that are both accurate and interpretable, overcoming the challenges of traditional black box models.
