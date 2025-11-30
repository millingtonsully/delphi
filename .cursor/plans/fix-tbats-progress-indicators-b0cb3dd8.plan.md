<!-- b0cb3dd8-652d-4894-a411-c0e3020e60cf 0ad1c3d1-98de-4d92-81a2-26e62b02ee14 -->
# Fix Config Parsing and Align with HERMES/Next Methodology

## Issues Identified

1. **Immediate Error**: YAML parsing `1e-3` as string causing TypeError
2. **Data Splits**: Need to verify rolling origin evaluation methodology
3. **Weak Signal Handling**: Ensure weak signals are properly included in test evaluation
4. **Config Type Safety**: Add type conversion for all numeric config values

## Implementation Plan

### 1. Fix Config Type Parsing (`train.py`)

**Problem**: YAML's `safe_load` may parse scientific notation (`1e-3`) as strings.

**Solution**: Add type conversion utility function and apply to all numeric config values.

- Create `_ensure_numeric()` helper function to convert strings to floats/ints
- Apply to: `learning_rate`, `weight_decay`, `kl_weight`, `entropy_weight`, `batch_size`, etc.
- Add validation with clear error messages

**Files**: `train.py` (lines ~305-357)

### 2. Ensure Proper Train/Val/Test Splits (`delphi/data/loader.py`)

**HERMES Methodology**:

- Training: Historical data up to train_end_week
- Validation: Used for hyperparameter tuning (not for final evaluation)
- Test: Rolling origin evaluation from val_end_week onwards

**Current Issue**: Validation split is too short (52 weeks) to create sequences properly.

**Solution**:

- Keep current split logic (correct per HERMES)
- Document that validation sequences come from training data (standard practice)
- Ensure test set uses rolling origin correctly
- Add comments explaining HERMES methodology alignment

**Files**: `delphi/data/loader.py` (lines ~374-441), `train.py` (lines ~154-185)

### 3. Fix Weak Signal Handling in Test Evaluation (`evaluate.py`, `train.py`)

**HERMES Methodology**: Weak signals (fashion forward data) should be available for test predictions.

**Current Issue**: Weak signals may not be properly passed to test evaluation.

**Solution**:

- Ensure `splits['test'] `includes `weak_signal_ratio` if available
- Update `generate_final_predictions()` to use test weak signals
- Update `evaluate.py` to load and use weak signals from test split

**Files**: `train.py` (lines ~197-287), `evaluate.py` (lines ~119-141), `delphi/data/loader.py` (lines ~428-439)

### 4. Add Config Validation (`train.py`)

**Solution**: Add comprehensive config validation after loading YAML:

- Check required keys exist
- Validate numeric ranges (learning_rate > 0, etc.)
- Ensure train_end_week < val_end_week
- Validate forecast_horizon is positive

**Files**: `train.py` (after line ~307)

### 5. Improve Rolling Origin Evaluation (`evaluate.py`)

**HERMES Methodology**: Test evaluation should use rolling origin approach.

**Current**: Single forecast from end of test data.

**Solution**:

- Add option for rolling origin evaluation (multiple forecast origins)
- Keep single forecast as default for simplicity
- Document rolling origin usage

**Files**: `evaluate.py` (lines ~119-141)

### 6. Ensure Weak Signal Ratio is Preserved in Splits (`delphi/data/loader.py`)

**Current Issue**: `create_train_val_test_split()` may not preserve `weak_signal_ratio` from preprocessed data.

**Solution**:

- Check if `weak_signal_ratio` exists in preprocessed data
- Include it in splits if available
- Ensure it's passed through correctly

**Files**: `delphi/data/loader.py` (lines ~374-441)

## Implementation Details

### Type Conversion Function

```python
def _ensure_numeric(value, default=None, min_val=None, max_val=None):
    """Convert config value to numeric type with validation."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        result = float(value) if isinstance(value, float) or '.' in str(value) else int(value)
    else:
        try:
            result = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid numeric value: {value}")
    if min_val is not None and result < min_val:
        raise ValueError(f"Value {result} below minimum {min_val}")
    if max_val is not None and result > max_val:
        raise ValueError(f"Value {result} above maximum {max_val}")
    return result
```

### Config Loading with Type Safety

```python
# After yaml.safe_load()
config['training']['learning_rate'] = _ensure_numeric(
    config['training']['learning_rate'], 
    default=1e-3, 
    min_val=1e-6
)
# ... apply to all numeric configs
```

## Testing Checklist

- [ ] Config loads without type errors
- [ ] Learning rate is float, not string
- [ ] Weak signals available in test evaluation
- [ ] Train/val/test splits follow HERMES methodology
- [ ] Rolling origin evaluation works (if implemented)
- [ ] All numeric configs are properly typed

## Notes

- Keep DELPHI architecture unchanged (HMM gating, ensemble correctors, etc.)
- Follow HERMES data management practices
- Ensure reproducibility with proper seed setting
- Maintain backward compatibility with existing configs

### To-dos

- [x] Fix Config Type Parsing - add _ensure_numeric() helper and apply to all numeric configs
- [x] Ensure Proper Train/Val/Test Splits with HERMES methodology documentation
- [x] Fix Weak Signal Handling in Test Evaluation - pass weak_signal_ratio to predictions
- [x] Add Config Validation - check required keys, validate ranges
- [x] Improve Rolling Origin Evaluation in evaluate.py