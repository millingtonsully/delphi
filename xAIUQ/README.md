# xAIUQ: Hybrid Explainability and Uncertainty Quantification for DELPHI

This module provides comprehensive explainability and uncertainty quantification for DELPHI model predictions, leveraging the existing variational inference framework.

## Components

### 1. VariationalExplainer
Leverages ELBO variational posteriors to explain regime shifts with uncertainty:
- `explain_regime_shifts()`: Compute regime shift probabilities
- `compute_state_attribution()`: Attribute forecast variance to HMM states
- `decompose_elbo_contributions()`: Break down ELBO loss by state

### 2. TimeSeriesFeatureAttribution
SHAP-based attribution for the 3 input features:
- `compute_shap_values()`: Compute SHAP values for time-series
- `attribute_features()`: Attribute forecast to input features
- `temporal_importance()`: Analyze temporal importance patterns

### 3. RegimeExplainer
State-specific explanations for each of the 4 correctors:
- `explain_state_activation()`: Explain why states activate
- `corrector_contributions()`: Quantify corrector contributions
- `regime_transition_explanation()`: Explain regime transitions

### 4. CausalAnalyzer
Intervention-based analysis for external signal impact:
- `intervention_analysis()`: "What-if" scenarios for weak signal changes
- `causal_impact()`: Measure causal impact of influencer data
- `signal_sensitivity()`: Quantify forecast sensitivity to weak signal

### 5. UncertaintyQuantifier
Enhanced uncertainty quantification:
- `decompose_uncertainty()`: Separate aleatoric and epistemic uncertainty
- `regime_uncertainty()`: Quantify uncertainty in regime detection
- `enhanced_confidence_intervals()`: Regime-aware confidence intervals

### 6. DelphiExplainer (Unified Interface)
Main API combining all components:
- `explain()`: Generate comprehensive explanation report
- `explain_regime_shifts()`: Regime shift explanations
- `explain_features()`: Feature attribution
- `explain_external_signal()`: External signal analysis
- `quantify_uncertainty()`: Uncertainty quantification

## Usage

### Command Line (Recommended)

```bash
# Explain all series from evaluation results
python explain.py --model_path checkpoints/delphi_final.pt

# Explain specific series
python explain.py --model_path checkpoints/delphi_final.pt --series_id br_female_outerwear_0

# Explain with specific components
python explain.py --model_path checkpoints/delphi_final.pt --components regime_shifts uncertainty

# Limit number of series
python explain.py --model_path checkpoints/delphi_final.pt --num_series 10
```

### Python API

```python
from delphi.xAIUQ import DelphiExplainer
from delphi.models.delphi_core import DELPHICore
import torch

# Load model
model = DELPHICore(...)
model.load_state_dict(torch.load('checkpoints/delphi_final.pt')['model_state_dict'])
model.eval()

# Initialize explainer
explainer = DelphiExplainer(model, device='cpu')

# Prepare input (batch, seq_len, 3)
x = torch.tensor(...)  # Input tensor
parametric_forecast = torch.tensor(...)  # Optional parametric forecast

# Generate comprehensive explanation
report = explainer.explain(x, parametric_forecast, include_uncertainty=True)

# Or get specific explanations
regime_shifts = explainer.explain_regime_shifts(x, parametric_forecast)
features = explainer.explain_features(x, parametric_forecast)
uncertainty = explainer.quantify_uncertainty(x, parametric_forecast)
```

## Output Structure

Explanations are saved to `explanation_results/`:

```
explanation_results/
├── explanations_summary.json          # High-level summary
├── explanations_by_series/           # Per-series detailed explanations
│   ├── {series_id}_regime_shifts.json
│   ├── {series_id}_feature_attribution.json
│   ├── {series_id}_uncertainty.json
│   ├── {series_id}_external_signal.json
│   └── {series_id}_complete.json     # All explanations combined
```

## Dependencies

- `torch` (required)
- `numpy` (required)
- `pandas` (required, for result_loader)
- `shap` (optional, for enhanced feature attribution)
- `matplotlib` (optional, for visualization)

## Notes

- The xAIUQ layer is non-invasive and does not modify existing Delphi code
- All explanations work with the existing DELPHICore model structure
- The explainer uses the same data pipeline as `evaluate.py` for consistency
- Explanations leverage existing variational posteriors and state probabilities

