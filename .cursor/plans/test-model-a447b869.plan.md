<!-- a447b869-a6c1-4f64-ade6-385570e57761 f3656b4a-2d48-446c-be08-948882f63d51 -->
# Fix Model Implementation and Codebase Cleanup

## Critical Bugs to Fix

### 1. Input Dimension Mismatch (`delphi/data/preprocessing.py`)

**Problem:** `prepare_model_inputs()` returns shape `(seq_len, 1)` when weak_signal/parametric_forecast is missing, but model expects `(seq_len, 3)`.

**Fix:** Always return 3-dimensional input, filling missing features with zeros or the main signal.

### 2. HMM States vs Ensemble Members (`delphi/models/ensemble_correctors.py`)

**Problem:** HMM has 4 states but ensemble has 5 members. `F.one_hot(states, num_classes=5)` fails when state >= 4.

**Fix:** Map 4 HMM states to 5 ensemble members properly, or make them equal.

### 3. OWA Metric (`delphi/evaluation/metrics.py`)

**Problem:** Current OWA divides by MAE-from-mean, giving inflated values (13.0 instead of ~0.8).

**Fix:** Replace with proper relative MAE or remove entirely since it's misleading.

### 4. Output Scaling (`delphi/models/delphi_core.py`)

**Problem:** When `parametric_forecast=None`, raw corrections are returned without scaling.

**Fix:** Apply sigmoid or other scaling to bound outputs to [0,1] for normalized data.

### 5. Predictor Input Handling (`delphi/inference/predictor.py`)

**Problem:** `predict_series()` may not properly pass parametric forecast to input preparation.

**Fix:** Ensure 3-dim input is always created with proper feature values.

## Files to Delete

- All `__pycache__/` directories
- `EVALUATION_GUIDE.md`, `IMPLEMENTATION_SUMMARY.md`, `INTEGRATION_GUIDE.md`
- `META_ENSEMBLE_GUIDE.md`, `QUICK_START.md`, `USAGE_GUIDE.md`
- `visualize.py`, `example_usage.py`
- `logs/` directory

## Files to Keep

- Core model: `delphi/models/`, `delphi/training/`, `delphi/inference/`, `delphi/evaluation/`, `delphi/data/`
- Scripts: `train.py`, `evaluate.py`, `inference.py`
- Config: `configs/delphi_config.yaml`
- Data: `data/` (keep main files)
- Checkpoints: `checkpoints/`
- `README.md`, `requirements.txt`

### To-dos

- [ ] Add pda() function to delphi/evaluation/metrics.py
- [ ] Integrate PDA into compute_all_metrics() function
- [ ] Add pda to metrics list in config file
- [ ] Fix prepare_model_inputs to always return 3-dim input
- [ ] Fix HMM state to ensemble member routing mismatch
- [ ] Replace OWA with proper relative MAE calculation
- [ ] Add output scaling when parametric_forecast is None
- [ ] Fix predictor to properly handle input preparation
- [ ] Delete unnecessary files and __pycache__ directories