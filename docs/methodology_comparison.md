# Methodology Comparison: DELPHI vs Variational Quantization Paper

## Overview

This document compares the current DELPHI implementation with the Variational Quantization for State Space Models approach (arXiv:2404.11117).

## Key Differences

### 1. Data Preprocessing

**Current DELPHI Approach:**
- Uses explicit deseasonalization (STL/LOESS) before model training
- Works on deseasonalized residuals: `z_t = y_t - seasonal_t`
- Re-seasonalizes predictions after inference: `ŷ_t = ŷ_deseasonalized + seasonal_t`
- Benefits: Removes strong seasonal patterns, model focuses on trend/residuals
- Challenges: Re-seasonalization must be accurate, potential scale mismatches

**Variational Quantization Paper:**
- Works directly on original scale data
- Model learns to handle seasonality through HMM states and emission distributions
- Each emission law can capture different seasonal patterns
- Benefits: No re-seasonalization needed, model learns seasonality naturally
- Challenges: Requires more model capacity to learn seasonality

### 2. Model Architecture

**Current DELPHI:**
- HMM with 4 states
- Each state routes to a specialized corrector (ensemble member)
- Corrections are added to TBATS parametric forecast
- Input: `[residuals, weak_signal_ratio, parametric_forecast]`

**Variational Quantization:**
- HMM with multiple states (typically 3-4)
- Each state has its own emission distribution (learned neural network)
- Emission laws are specialized for different behaviors
- Input includes external signals directly in hidden states

### 3. Training Procedure

**Current DELPHI:**
- Two-stage ELBO training:
  1. Stage 1: Train emissions and posterior (fix uniform prior)
  2. Stage 2: Train prior (freeze emissions/posterior)
- Uses KL divergence between prior and posterior
- Entropy regularization on state probabilities

**Variational Quantization:**
- Two-stage training:
  1. Stage 1: Train emission distributions and variational posterior
  2. Stage 2: Train HMM prior (transition matrices and initial probabilities)
- Similar ELBO objective but works on original scale

### 4. Evaluation

**Current DELPHI:**
- MASE computed on original scale (after re-seasonalization)
- Potential issues:
  - If seasonal components don't match forecast period → scale mismatch
  - Re-seasonalization errors compound forecast errors

**Variational Quantization:**
- MASE computed directly on original scale predictions
- No re-seasonalization needed
- More straightforward evaluation

## Recommendations

### Short-term Fixes (Implemented)

1. **Improved Re-seasonalization:**
   - Added `forecast_seasonal_components()` function to forecast seasonal components for future periods
   - Uses seasonal naive method when test period extends beyond available data

2. **Enhanced Diagnostics:**
   - Added MASE component breakdown (numerator, denominator, scaling factor)
   - Added re-seasonalization diagnostics
   - Added scale comparison diagnostics

3. **Validation Checks:**
   - Verify seasonal components are not zero when deseasonalization was used
   - Check scale consistency between predictions and true values
   - Verify re-seasonalization arithmetic

### Long-term Improvements (Consider)

1. **Direct Scale Approach:**
   - Consider working directly on original scale (like the paper)
   - Let HMM states learn to handle seasonality
   - Requires model retraining but eliminates re-seasonalization issues

2. **Better Seasonal Modeling:**
   - If keeping deseasonalization, use TBATS seasonal components for re-seasonalization
   - Or learn seasonal components jointly with the model

3. **External Signal Integration:**
   - Paper shows strong improvements with external signals
   - Current implementation includes weak signals but could be enhanced

## MASE Interpretation

The high MASE (29.67) indicates:
- Forecast errors are ~29.7x larger than seasonal naive baseline
- This could mean:
  1. Model predictions are genuinely poor
  2. Seasonal naive baseline is very strong (small denominator)
  3. Re-seasonalization is introducing errors

The diagnostic tools will help identify which case applies.









