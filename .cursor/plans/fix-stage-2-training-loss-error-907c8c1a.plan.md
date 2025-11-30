<!-- 907c8c1a-3ca5-4248-9674-3610e5d8264e d3a35cc2-5834-4a63-bb62-ae083ed2b315 -->
# Replace sMAPE with MASE in Validation Metrics

## Problem

sMAPE is misleading for deseasonalized/normalized data because small values near zero inflate the percentage error dramatically, even when absolute errors are tiny.

## Solution

Replace sMAPE with MASE (Mean Absolute Scaled Error) which:

- Scales error by naive forecast error (more robust)
- Standard metric in forecasting competitions
- Works better with normalized/deseasonalized data

## Files to Modify

### `delphi/training/trainer.py`

1. Change import from `smape` to `mase`
2. Replace `smape()` call with `mase()` call
3. Update metrics dictionary key from `val_smape` to `val_mase`
4. Update print statements in `train_stage1()` and `train_stage2()` to show MASE instead of sMAPE

## Implementation Details

The `mase()` function already exists in `delphi/evaluation/metrics.py`. Without training data available in the validation context, it will use the fallback: scaling by mean of true values, which is still more robust than sMAPE for small values.

### To-dos

- [ ] Change import from smape to mase in trainer.py validate()
- [ ] Replace smape() with mase() in metrics computation
- [ ] Update print statements in train_stage1() and train_stage2() to show MASE