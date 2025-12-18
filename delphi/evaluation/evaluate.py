"""
Evaluation script for DELPHI.

Uses holdout evaluation: last forecast_horizon weeks as test data.
Outputs metrics to text file and predictions to CSV for readability.
"""

import argparse
import yaml
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
from typing import Any, Union, Optional, Dict

# Add project root to path for imports when running from subdirectory
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from delphi.models.delphi_core import DELPHICore
from delphi.models.parametric import TBATSBaseline
from delphi.data import (
    DataLoader,
    preprocess_data,
    create_train_val_test_split
)
from delphi.data.preprocessing import prepare_model_inputs, forecast_seasonal_components
from delphi.evaluation.metrics import compute_all_metrics, mase as mase_score


def _ensure_numeric(
    value: Any, 
    default: Optional[Union[int, float]] = None, 
    value_type: str = 'float'
) -> Union[int, float]:
    """Convert config value to numeric type (handles YAML parsing issues)."""
    if value is None:
        return default if default is not None else 0
    try:
        if value_type == 'int':
            return int(float(value))
        return float(value)
    except (ValueError, TypeError):
        return default if default is not None else 0


def load_model(checkpoint_path: str, model_config: Dict, device: str) -> DELPHICore:
    """
    Load DELPHI model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_config: Model configuration dictionary
        device: Device for inference
    
    Returns:
        Loaded model in eval mode
    """
    model = DELPHICore(**model_config)
    checkpoint = torch.load(
        checkpoint_path, 
        map_location=torch.device(device),
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def predict_series(
    model: DELPHICore,
    device: str,
    main_signal: np.ndarray,
    residuals: Optional[np.ndarray] = None,
    weak_signal_ratio: Optional[np.ndarray] = None,
    parametric_forecast: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Generate prediction for a single time series.
    
    Args:
        model: DELPHI model
        device: Device for inference
        main_signal: Main time series signal
        residuals: TBATS residuals (z_t = y_t - ŷ_t)
        weak_signal_ratio: Weak signal ratio
        parametric_forecast: Parametric baseline forecast
    
    Returns:
        Dictionary with forecast results
    """
    # Prepare input tensor
    input_tensor = prepare_model_inputs(
        main_signal,
        residuals=residuals,
        weak_signal_ratio=weak_signal_ratio,
        parametric_forecast=parametric_forecast
    )
    
    # Add batch dimension and convert to torch tensor
    inputs = torch.FloatTensor(input_tensor).unsqueeze(0).to(device)
    
    if parametric_forecast is not None:
        param_tensor = torch.FloatTensor(parametric_forecast).unsqueeze(0).to(device)
    else:
        param_tensor = None
    
    # Run inference
    with torch.no_grad():
        results = model(inputs, parametric_forecast=param_tensor)
        return {
            'forecast': results['forecast'].cpu().numpy(),
            'correction': results['correction'].cpu().numpy(),
            'state_probs': results['state_probs'].cpu().numpy()
        }


# Guard against duplicate execution (Windows python -m issue)
_main_executed = False

def _filter_series_dict(series_dict: Dict[str, np.ndarray], allowed_ids: list) -> Dict[str, np.ndarray]:
    """Return a dict containing only allowed series IDs."""
    return {k: v for k, v in series_dict.items() if k in allowed_ids}


def main():
    global _main_executed
    if _main_executed:
        return
    _main_executed = True
    
    parser = argparse.ArgumentParser(description='Evaluate DELPHI model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, default=None,
                       help='Path to test data directory (if None, uses config data_dir)')
    parser.add_argument('--config', type=str, default='configs/delphi_config.yaml',
                       help='Path to config file (relative to project root)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--num_series', type=int, default=10000,
                       help='Number of series to evaluate (uses random subset with seed)')
    # NOTE: Shuffling of series is currently disabled (see selection logic below),
    # so the seed does not affect which series are chosen at the moment. It is
    # retained for compatibility and potential future re-enabling of shuffling.
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for series subsampling (defaults to config seed or 42)')
    args = parser.parse_args()
    
    # Resolve config path relative to project root
    if not os.path.isabs(args.config):
        config_path = project_root / args.config
    else:
        config_path = Path(args.config)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure numeric values are properly typed
    forecast_horizon = _ensure_numeric(config['data']['forecast_horizon'], 26, 'int')
    train_end_week = _ensure_numeric(config['data'].get('train_end_week', 209), 209, 'int')
    val_end_week = _ensure_numeric(config['data'].get('val_end_week', 261), 261, 'int')
    seasonal_period = _ensure_numeric(config['data'].get('seasonal_period', 52), 52, 'int')
    context_length = 52  # Fixed context length
    
    # Create output directory (relative to project root)
    output_dir = project_root / args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    # Initialize RNG for potential future use. Currently, selection is
    # deterministic (no shuffle), so this seed does not affect which series
    # are evaluated; it is kept here for compatibility and possible re-enabling
    # of random subsampling.
    seed = args.seed if args.seed is not None else _ensure_numeric(config.get('seed', 42), 42, 'int')
    rng = np.random.default_rng(int(seed))

    model_config = {
        'input_dim': 3,
        'n_states': config['hmm']['n_states'],
        'hmm_hidden_size': config['hmm']['hidden_size'],
        'hmm_num_layers': config['hmm']['num_layers'],
        'ensemble_hidden_size': config['ensemble']['hidden_size'],
        'ensemble_num_layers': config['ensemble']['num_layers'],
        'output_dim': forecast_horizon,
        'n_ensemble_members': config['ensemble']['n_members'],
        'dropout': config['hmm']['dropout']
    }
    
    # Load model
    print("Loading DELPHI model...")
    print(f"  Device: {device}")
    model = load_model(args.model_path, model_config, device)
    print("  Model loaded successfully!")
    
    # Load data
    print("\nLoading data...")
    data_dir = args.test_data if args.test_data else config['data']['data_dir']
    loader = DataLoader(data_dir=data_dir)
    data = loader.load_all_series(load_weak_signal=True)
    
    # Select subset of series before preprocessing to reduce runtime
    all_series_ids = list(data['main_signal'].keys())
    # Disable random shuffling so evaluation uses the same deterministic series
    # ordering as training (first N series). This ensures that when using
    # training_series_limit=N, evaluation on num_series=N aligns with those
    # same series.
    # rng.shuffle(all_series_ids)
    selected_ids = all_series_ids[:args.num_series]
    # Note: Selection is currently deterministic (first N series); the seed is
    # reported below for reproducibility/logging only and does not affect which
    # series are chosen while shuffling remains disabled.
    if len(selected_ids) < len(all_series_ids):
        print(f"Subsampling series: {len(selected_ids)} of {len(all_series_ids)} (seed={seed})")
    else:
        print(f"Using all available series: {len(selected_ids)} (seed={seed})")

    # Filter data to the selected subset (main + weak signals)
    data = {
        'main_signal': _filter_series_dict(data['main_signal'], selected_ids),
        'weak_signal': _filter_series_dict(data.get('weak_signal', {}), selected_ids)
    }

    # Preserve original data for MASE computation (before deseasonalization) on the subset
    # MASE is evaluated on original scale, not deseasonalized
    original_data = {
        'main_signal': {k: v.copy() for k, v in data['main_signal'].items()},
        'weak_signal': {k: v.copy() for k, v in data.get('weak_signal', {}).items()}
    }
    
    # Preprocess
    print("Preprocessing data...")
    preprocessed = preprocess_data(
        data,
        deseasonalize=config['preprocessing'].get('deseasonalize', True),
        compute_weak_ratio=config['preprocessing'].get('compute_weak_ratio', True),
        seasonal_period=_ensure_numeric(config['data'].get('seasonal_period', 52), 52, 'int'),
        verbose=False
    )
    
    # Create splits for preprocessed data (used for model inference)
    splits = create_train_val_test_split(
        preprocessed,
        train_end_week=train_end_week,
        val_end_week=val_end_week,
        forecast_horizon=forecast_horizon
    )
    
    # Create splits for original data (used for MASE computation)
    original_splits = create_train_val_test_split(
        original_data,
        train_end_week=train_end_week,
        val_end_week=val_end_week,
        forecast_horizon=forecast_horizon
    )
    
    train_split = splits['train']
    val_split = splits['val']
    test_split = splits['test']
    
    train_main = train_split['main_signal']
    val_main = val_split['main_signal']
    test_main = test_split['main_signal']
    weak_train = train_split.get('weak_signal_ratio', {})
    weak_val = val_split.get('weak_signal_ratio', {})
    
    # Get seasonal components for re-seasonalization
    seasonal_train = train_split.get('seasonal_components', {})
    seasonal_test = test_split.get('seasonal_components', {})
    
    # Original data splits for MASE computation
    orig_train_split = original_splits['train']
    orig_test_split = original_splits['test']
    orig_train_main = orig_train_split['main_signal']
    orig_test_main = orig_test_split['main_signal']
    
    # Determine series with test data available (relaxed filtering to match paper methodology)
    # Paper evaluates all 10,000 series. All series in the dataset have 261 weeks,
    # so we only exclude series with no test data at all (minimal filtering).
    # This matches the paper's approach of evaluating on the full dataset.
    valid_series = []
    for series_id in preprocessed['main_signal'].keys():
        test_seq = test_main.get(series_id, np.array([]))
        
        # Only exclude series with no test data at all
        if len(test_seq) == 0:
            continue
        
        valid_series.append(series_id)
    
    if not valid_series:
        print("\nERROR: No series have test data for the requested split.")
        return
    
    # Respect the preselected subset while ensuring test availability
    eval_series = [sid for sid in selected_ids if sid in valid_series][:args.num_series]
    dropped = len(selected_ids) - len(eval_series)
    print(f"\nEvaluating on {len(eval_series)} series (seed={seed})...")
    if dropped > 0:
        print(f"  Skipped {dropped} selected series without test data.")
    if len(valid_series) > len(eval_series):
        print(f"  (Limited from {len(valid_series)} available valid series)")
    
    # Generate parametric baseline forecasts
    print("\nGenerating TBATS baseline forecasts...")
    # Get TBATS parallelization settings from config (with defaults for backwards compatibility)
    n_parallel_workers = config['parametric'].get('n_parallel_workers', 8)
    n_jobs = config['parametric'].get('n_jobs', 1)
    
    tbats = TBATSBaseline(
        use_box_cox=config['parametric'].get('use_box_cox', True),
        use_trend=config['parametric'].get('use_trend', True),
        use_arma_errors=config['parametric'].get('use_arma_errors', True),
        seasonal_periods=config['parametric'].get('seasonal_periods', [52]),
        n_jobs=n_jobs,
        n_parallel_workers=n_parallel_workers
    )
    
    # Use history up to the beginning of the test window for TBATS fitting
    tbats_input = {}
    for series_id in eval_series:
        history_main = np.concatenate([
            train_main.get(series_id, np.array([])),
            val_main.get(series_id, np.array([]))
        ])
        tbats_input[series_id] = history_main
    param_forecasts, param_residuals = tbats.fit_and_forecast(
        tbats_input,
        forecast_horizon=forecast_horizon,
        verbose=True  # Enable verbose to see fitting stats
    )
    
    # Display TBATS fitting statistics
    tbats_stats = tbats.get_fitting_stats()
    print(f"\nTBATS Fitting Statistics:")
    print(f"  Successful fits: {tbats_stats['successful_fits']}/{len(eval_series)}")
    print(f"  Failed fits: {tbats_stats['failed_fits']}")
    print(f"  Fallback to mean: {tbats_stats['fallback_to_mean']}")
    print(f"  Offset applied: {tbats_stats['offset_applied']}")
    
    # Check a sample of TBATS forecasts to verify scale
    sample_series_ids = list(eval_series[:3])
    print(f"\nTBATS Forecast Scale Check (sample series):")
    for series_id in sample_series_ids:
        if series_id in param_forecasts:
            tbats_forecast = param_forecasts[series_id]
            original_data = tbats_input.get(series_id, np.array([]))
            if len(original_data) > 0:
                data_mean = np.mean(np.abs(original_data))
                forecast_mean = np.mean(np.abs(tbats_forecast))
                scale_ratio = forecast_mean / data_mean if data_mean > 0 else 0
                print(f"  {series_id}:")
                print(f"    Original data mean: {data_mean:.6f}")
                print(f"    TBATS forecast mean: {forecast_mean:.6f}")
                print(f"    Scale ratio: {scale_ratio:.3f} (should be ~1.0)")
    
    # Run evaluation
    print("\nRunning DELPHI predictions...")
    predictions = {}
    true_values = {}
    train_histories = {}
    original_train_histories = {}  # For MASE denominator (original scale)
    original_true_values = {}  # For MASE numerator (original scale)
    
    # Diagnostic data storage
    diagnostic_data = {
        'reseasonalization_info': [],
        'scale_comparisons': [],
        'mase_components': [],
        'scale_analysis': []  # Detailed scale diagnostics
    }
    
    for idx, series_id in enumerate(eval_series, 1):
        train_history = train_main.get(series_id, np.array([]))
        val_history = val_main.get(series_id, np.array([]))
        history_main = np.concatenate([train_history, val_history])
        test_sequence = test_main.get(series_id, np.array([]))[:forecast_horizon]
        
        # Skip only if no test data at all (shouldn't happen after initial filtering)
        if len(test_sequence) == 0:
            continue
        
        # Get original scale data for MASE
        orig_train_history = orig_train_main.get(series_id, np.array([]))
        orig_test_sequence = orig_test_main.get(series_id, np.array([]))[:forecast_horizon]
        
        param_forecast = param_forecasts.get(series_id)
        series_residuals = param_residuals.get(series_id)
        
        # Use the last context_length points before the test window as model input
        input_seq = history_main[-context_length:]
        target_seq = test_sequence
        
        # Store ground truth (deseasonalized for MSE/MAE)
        true_values[series_id] = target_seq
        train_histories[series_id] = train_history
        
        # Store original scale data for MASE
        original_true_values[series_id] = orig_test_sequence
        original_train_histories[series_id] = orig_train_history
        
        # Get residuals for input context
        input_residuals = None
        if series_residuals is not None and len(series_residuals) >= context_length:
            input_residuals = series_residuals[-context_length:]
        
        # Get weak ratio history up to the test boundary, if available
        weak_history_segments = []
        if series_id in weak_train and len(weak_train[series_id]) > 0:
            weak_history_segments.append(weak_train[series_id])
        if series_id in weak_val and len(weak_val[series_id]) > 0:
            weak_history_segments.append(weak_val[series_id])
        
        input_weak_ratio = None
        if weak_history_segments:
            weak_history = np.concatenate(weak_history_segments)
            if len(weak_history) >= context_length:
                input_weak_ratio = weak_history[-context_length:]
        
        # Predict
        result = predict_series(
            model=model,
            device=device,
            main_signal=input_seq,
            residuals=input_residuals,
            weak_signal_ratio=input_weak_ratio,
            parametric_forecast=param_forecast
        )
        
        # Get deseasonalized forecast from model (on deseasonalized scale)
        deseasonalized_forecast = result['forecast'].flatten()
        
        # Re-seasonalize forecast for MASE computation (evaluated on original scale)
        # Get seasonal components from test split
        seasonal_components_forecast = seasonal_test.get(series_id, np.array([]))
        seasonal_train_full = seasonal_train.get(series_id, np.array([]))
        
        # Check if we need to forecast seasonal components for future periods
        needs_forecasting = False
        if len(seasonal_components_forecast) < len(deseasonalized_forecast):
            needs_forecasting = True
            # Forecast seasonal components using seasonal naive method
            # Combine train and val seasonal components to get full historical pattern
            seasonal_val = val_split.get('seasonal_components', {}).get(series_id, np.array([]))
            if len(seasonal_train_full) > 0 and len(seasonal_val) > 0:
                historical_seasonal = np.concatenate([seasonal_train_full, seasonal_val])
            elif len(seasonal_train_full) > 0:
                historical_seasonal = seasonal_train_full
            elif len(seasonal_val) > 0:
                historical_seasonal = seasonal_val
            else:
                historical_seasonal = np.array([])
            
            if len(historical_seasonal) > 0:
                # Forecast missing seasonal components
                missing_length = len(deseasonalized_forecast) - len(seasonal_components_forecast)
                forecasted_seasonal = forecast_seasonal_components(
                    historical_seasonal,
                    missing_length,
                    seasonal_period
                )
                # Append forecasted components to existing ones
                seasonal_components_forecast = np.concatenate([
                    seasonal_components_forecast,
                    forecasted_seasonal[:missing_length]
                ])
        
        # Ensure seasonal components match forecast length
        if len(seasonal_components_forecast) > len(deseasonalized_forecast):
            seasonal_components_forecast = seasonal_components_forecast[:len(deseasonalized_forecast)]
        elif len(seasonal_components_forecast) < len(deseasonalized_forecast):
            # If still shorter, pad with zeros (shouldn't happen with forecasting above)
            padding = np.zeros(len(deseasonalized_forecast) - len(seasonal_components_forecast))
            seasonal_components_forecast = np.concatenate([seasonal_components_forecast, padding])
        
        # Re-seasonalize: Add seasonal component back to get original scale forecast
        reseasonalized_forecast = deseasonalized_forecast + seasonal_components_forecast
        
        # Validation: Check scale consistency
        # Verify that re-seasonalization makes sense
        validation_errors = []
        
        # Check 1: Seasonal components should not be zero if deseasonalization was used
        if config['preprocessing'].get('deseasonalize', True):
            if np.allclose(seasonal_components_forecast, 0, atol=1e-6):
                validation_errors.append(f"Seasonal components are all zero but deseasonalization was used")
        
        # Check 2: Re-seasonalized forecast should be on a similar scale as true values
        if len(orig_test_sequence) == len(reseasonalized_forecast):
            true_mean = np.mean(np.abs(orig_test_sequence))
            pred_mean = np.mean(np.abs(reseasonalized_forecast))
            if true_mean > 1e-6:
                scale_ratio = pred_mean / true_mean
                if scale_ratio < 0.1 or scale_ratio > 10:
                    validation_errors.append(f"Scale mismatch: pred/true ratio = {scale_ratio:.4f}")
        
        # Check 2b: Deseasonalized forecast should match test sequence scale
        if len(test_sequence) == len(deseasonalized_forecast):
            test_mean = np.mean(np.abs(test_sequence))
            pred_mean = np.mean(np.abs(deseasonalized_forecast))
            if test_mean > 1e-6:
                scale_ratio = pred_mean / test_mean
                if scale_ratio < 0.5 or scale_ratio > 2.0:
                    validation_errors.append(f"Potential scale mismatch detected: pred/test deseasonalized ratio = {scale_ratio:.4f}")
        
        # Check 3: Re-seasonalized forecast = deseasonalized + seasonal (should be exact)
        recomputed = deseasonalized_forecast + seasonal_components_forecast
        if not np.allclose(recomputed, reseasonalized_forecast, atol=1e-6):
            validation_errors.append(f"Re-seasonalization check failed: recomputed != stored")
        
        # Check 4: If TBATS forecast is available, verify scale consistency
        if param_forecast is not None and len(param_forecast) == len(deseasonalized_forecast):
            # TBATS forecast should be on deseasonalized scale (since model is fit on deseasonalized data)
            # The correction should be small relative to TBATS forecast
            if np.mean(np.abs(param_forecast)) > 1e-6:
                correction_ratio = np.mean(np.abs(result['correction'].flatten())) / np.mean(np.abs(param_forecast))
                if correction_ratio > 10:
                    validation_errors.append(f"Correction is very large relative to TBATS forecast: ratio = {correction_ratio:.4f}")
        
        # Log validation errors for first few series
        if validation_errors and idx <= 5:
            print(f"\n  Validation warnings for {series_id}:")
            for err in validation_errors:
                print(f"    - {err}")
        
        # Store diagnostic information (for first few series only to avoid clutter)
        if idx <= 5:
            # Compute statistics for scale comparison
            orig_train_stats = {
                'mean': np.mean(orig_train_history) if len(orig_train_history) > 0 else 0,
                'std': np.std(orig_train_history) if len(orig_train_history) > 0 else 0,
                'min': np.min(orig_train_history) if len(orig_train_history) > 0 else 0,
                'max': np.max(orig_train_history) if len(orig_train_history) > 0 else 0
            }
            
            train_history_stats = {
                'mean': np.mean(train_history) if len(train_history) > 0 else 0,
                'std': np.std(train_history) if len(train_history) > 0 else 0,
                'min': np.min(train_history) if len(train_history) > 0 else 0,
                'max': np.max(train_history) if len(train_history) > 0 else 0
            }
            
            param_forecast_stats = {
                'mean': np.mean(param_forecast) if param_forecast is not None and len(param_forecast) > 0 else 0,
                'std': np.std(param_forecast) if param_forecast is not None and len(param_forecast) > 0 else 0,
                'min': np.min(param_forecast) if param_forecast is not None and len(param_forecast) > 0 else 0,
                'max': np.max(param_forecast) if param_forecast is not None and len(param_forecast) > 0 else 0
            }
            
            correction_stats = {
                'mean': np.mean(result['correction'].flatten()) if 'correction' in result else 0,
                'std': np.std(result['correction'].flatten()) if 'correction' in result else 0,
                'min': np.min(result['correction'].flatten()) if 'correction' in result else 0,
                'max': np.max(result['correction'].flatten()) if 'correction' in result else 0
            }
            
            diagnostic_data['scale_analysis'].append({
                'series_id': series_id,
                'original_train_stats': orig_train_stats,
                'deseasonalized_train_stats': train_history_stats,
                'deseasonalized_test_stats': {
                    'mean': np.mean(test_sequence),
                    'std': np.std(test_sequence),
                    'min': np.min(test_sequence),
                    'max': np.max(test_sequence)
                },
                'original_test_stats': {
                    'mean': np.mean(orig_test_sequence),
                    'std': np.std(orig_test_sequence),
                    'min': np.min(orig_test_sequence),
                    'max': np.max(orig_test_sequence)
                },
                'tbats_forecast_stats': param_forecast_stats,
                'model_correction_stats': correction_stats,
                'deseasonalized_forecast_stats': {
                    'mean': np.mean(deseasonalized_forecast),
                    'std': np.std(deseasonalized_forecast),
                    'min': np.min(deseasonalized_forecast),
                    'max': np.max(deseasonalized_forecast)
                },
                'seasonal_components_stats': {
                    'mean': np.mean(seasonal_components_forecast),
                    'std': np.std(seasonal_components_forecast),
                    'min': np.min(seasonal_components_forecast),
                    'max': np.max(seasonal_components_forecast)
                },
                'scale_ratios': {
                    'pred_to_test_deseasonalized': np.mean(deseasonalized_forecast) / (np.mean(test_sequence) + 1e-8),
                    'pred_to_test_original': np.mean(reseasonalized_forecast) / (np.mean(orig_test_sequence) + 1e-8),
                    'train_deseasonalized_to_original': train_history_stats['mean'] / (orig_train_stats['mean'] + 1e-8) if orig_train_stats['mean'] > 1e-8 else np.nan,
                    'test_deseasonalized_to_original': np.mean(test_sequence) / (np.mean(orig_test_sequence) + 1e-8)
                }
            })
            
            diagnostic_data['reseasonalization_info'].append({
                'series_id': series_id,
                'needs_forecasting': needs_forecasting,
                'seasonal_components_length': len(seasonal_components_forecast),
                'forecast_length': len(deseasonalized_forecast),
                'seasonal_mean': np.mean(seasonal_components_forecast),
                'seasonal_std': np.std(seasonal_components_forecast),
                'deseasonalized_forecast_mean': np.mean(deseasonalized_forecast),
                'reseasonalized_forecast_mean': np.mean(reseasonalized_forecast),
                'true_values_mean': np.mean(orig_test_sequence)
            })
            
            # Store scale comparison
            diagnostic_data['scale_comparisons'].append({
                'series_id': series_id,
                'deseasonalized_range': (np.min(deseasonalized_forecast), np.max(deseasonalized_forecast)),
                'reseasonalized_range': (np.min(reseasonalized_forecast), np.max(reseasonalized_forecast)),
                'true_range': (np.min(orig_test_sequence), np.max(orig_test_sequence)),
                'scale_diff_ratio': np.mean(np.abs(reseasonalized_forecast - orig_test_sequence)) / (np.mean(np.abs(orig_test_sequence)) + 1e-8)
            })
        
        # Store re-seasonalized predictions (original scale) for MASE computation
        predictions[series_id] = reseasonalized_forecast
        
        if idx % 20 == 0 or idx == len(eval_series):
            print(f"  Processed {idx}/{len(eval_series)} series")
    
    # Compute metrics on original scale
    # All metrics (MSE, MAE, MASE) should be computed on original scale for fair comparison
    print("\n" + "="*70)
    print("EVALUATION METRICS")
    print("="*70)
    
    metrics, metric_stats = compute_all_metrics(
        y_true=original_true_values,  # Original scale test data
        y_pred=predictions,  # Re-seasonalized predictions
        training_data=original_train_histories,  # Original scale training data for MASE denominator
        seasonal_period=seasonal_period,
        forecast_horizon=forecast_horizon,
        return_stats=True
    )
    
    # Compute MASE diagnostics for first few series
    print("\nComputing MASE diagnostics...")
    sample_series = list(original_true_values.keys())[:5]
    for series_id in sample_series:
        if series_id in predictions and series_id in original_true_values:
            true_vals = original_true_values[series_id]
            pred_vals = predictions[series_id]
            train_series = original_train_histories.get(series_id)
            if train_series is not None and len(train_series) > seasonal_period:
                mase_diag = mase_score(
                    true_vals, pred_vals, train_series,
                    seasonal_period, forecast_horizon,
                    return_diagnostics=True
                )
                if isinstance(mase_diag, dict) and 'error' not in mase_diag:
                    diagnostic_data['mase_components'].append({
                        'series_id': series_id,
                        **mase_diag
                    })
    
    # Print results
    print("\nResults:")
    print("-" * 40)
    for metric, value in metrics.items():
        if isinstance(value, float) and not np.isnan(value):
            print(f"  {metric.upper():20s}: {value:.4f}")
    
    # Print metric computation statistics
    print("\nMetric Computation Statistics:")
    print("-" * 40)
    print(f"  Total series with predictions: {metric_stats['series_with_predictions']}")
    print(f"  Series with valid MSE/MAE: {metric_stats['series_with_valid_mse_mae']}")
    if 'mase' in metrics:
        print(f"  Series with valid MASE: {metric_stats['series_with_valid_mase']}")
        if metric_stats['series_with_failed_mase'] > 0:
            print(f"  Series with failed MASE: {metric_stats['series_with_failed_mase']}")
            if metric_stats['mase_failure_reasons']:
                print("  MASE failure reasons:")
                for reason, count in metric_stats['mase_failure_reasons'].items():
                    print(f"    - {reason}: {count}")
    
    # Print diagnostic information
    if diagnostic_data['mase_components']:
        print("\nMASE Diagnostic Components (sample series):")
        print("-" * 70)
        for diag in diagnostic_data['mase_components'][:3]:
            print(f"\n  Series: {diag['series_id']}")
            print(f"    MASE: {diag['mase']:.4f}")
            print(f"    Scaling Factor (T-m)/h: {diag['scaling_factor']:.4f}")
            print(f"    Error Ratio: {diag['error_ratio']:.4f}")
            print(f"    Mean Forecast Error: {diag['mean_forecast_error']:.4f}")
            print(f"    Mean Seasonal Naive Error: {diag['mean_seasonal_naive_error']:.4f}")
    
    if diagnostic_data['reseasonalization_info']:
        print("\nRe-seasonalization Diagnostics (sample series):")
        print("-" * 70)
        for diag in diagnostic_data['reseasonalization_info'][:3]:
            print(f"\n  Series: {diag['series_id']}")
            print(f"    Needed Forecasting: {diag['needs_forecasting']}")
            print(f"    Seasonal Component Mean: {diag['seasonal_mean']:.4f} ± {diag['seasonal_std']:.4f}")
            print(f"    Deseasonalized Forecast Mean: {diag['deseasonalized_forecast_mean']:.4f}")
            print(f"    Re-seasonalized Forecast Mean: {diag['reseasonalized_forecast_mean']:.4f}")
            print(f"    True Values Mean: {diag['true_values_mean']:.4f}")
    
    if diagnostic_data['scale_comparisons']:
        print("\nScale Comparison Diagnostics (sample series):")
        print("-" * 70)
        for diag in diagnostic_data['scale_comparisons'][:3]:
            print(f"\n  Series: {diag['series_id']}")
            print(f"    Scale Difference Ratio: {diag['scale_diff_ratio']:.4f}")
            print(f"    Re-seasonalized Range: [{diag['reseasonalized_range'][0]:.4f}, {diag['reseasonalized_range'][1]:.4f}]")
            print(f"    True Range: [{diag['true_range'][0]:.4f}, {diag['true_range'][1]:.4f}]")
    
    if diagnostic_data['scale_analysis']:
        print("\nDetailed Scale Analysis (sample series):")
        print("-" * 70)
        for diag in diagnostic_data['scale_analysis'][:3]:
            print(f"\n  Series: {diag['series_id']}")
            print(f"    Original Train: mean={diag['original_train_stats']['mean']:.6f}, std={diag['original_train_stats']['std']:.6f}")
            print(f"    Deseasonalized Train: mean={diag['deseasonalized_train_stats']['mean']:.6f}, std={diag['deseasonalized_train_stats']['std']:.6f}")
            print(f"    Deseasonalized Test: mean={diag['deseasonalized_test_stats']['mean']:.6f}, std={diag['deseasonalized_test_stats']['std']:.6f}")
            print(f"    Original Test: mean={diag['original_test_stats']['mean']:.6f}, std={diag['original_test_stats']['std']:.6f}")
            print(f"    TBATS Forecast: mean={diag['tbats_forecast_stats']['mean']:.6f}, std={diag['tbats_forecast_stats']['std']:.6f}")
            print(f"    Model Correction: mean={diag['model_correction_stats']['mean']:.6f}, std={diag['model_correction_stats']['std']:.6f}")
            print(f"    Model Forecast (deseasonalized): mean={diag['deseasonalized_forecast_stats']['mean']:.6f}")
            print(f"    Scale Ratios:")
            for key, val in diag['scale_ratios'].items():
                if not np.isnan(val) and not np.isinf(val):
                    print(f"      {key}: {val:.4f}")
    
    # Save metrics to text file
    metrics_path = output_dir / 'evaluation_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("DELPHI Model Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Series evaluated: {len(eval_series)}\n")
        f.write(f"Forecast horizon: {forecast_horizon}\n")
        f.write(f"Context length: {context_length}\n")
        f.write(f"Seasonal period: {seasonal_period}\n\n")
        f.write("Metrics:\n")
        f.write("-" * 50 + "\n")
        for metric, value in metrics.items():
            if isinstance(value, float) and not np.isnan(value):
                f.write(f"  {metric.upper():20s}: {value:.4f}\n")
        
        # Add metric computation statistics
        f.write("\n\nMetric Computation Statistics:\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Total series with predictions: {metric_stats['series_with_predictions']}\n")
        f.write(f"  Series with valid MSE/MAE: {metric_stats['series_with_valid_mse_mae']}\n")
        if 'mase' in metrics:
            f.write(f"  Series with valid MASE: {metric_stats['series_with_valid_mase']}\n")
            if metric_stats['series_with_failed_mase'] > 0:
                f.write(f"  Series with failed MASE: {metric_stats['series_with_failed_mase']}\n")
                if metric_stats['mase_failure_reasons']:
                    f.write("  MASE failure reasons:\n")
                    for reason, count in metric_stats['mase_failure_reasons'].items():
                        f.write(f"    - {reason}: {count}\n")
        
        # Add diagnostic information
        if diagnostic_data['mase_components']:
            f.write("\n\nMASE Diagnostic Components (sample series):\n")
            f.write("-" * 50 + "\n")
            for diag in diagnostic_data['mase_components'][:5]:
                f.write(f"\nSeries: {diag['series_id']}\n")
                f.write(f"  MASE: {diag['mase']:.4f}\n")
                f.write(f"  Scaling Factor (T-m)/h: {diag['scaling_factor']:.4f}\n")
                f.write(f"  Error Ratio: {diag['error_ratio']:.4f}\n")
                f.write(f"  Mean Forecast Error: {diag['mean_forecast_error']:.4f}\n")
                f.write(f"  Mean Seasonal Naive Error: {diag['mean_seasonal_naive_error']:.4f}\n")
                f.write(f"  Sum Forecast Errors: {diag['sum_forecast_errors']:.4f}\n")
                f.write(f"  Sum Seasonal Naive Errors: {diag['sum_seasonal_naive_errors']:.4f}\n")
        
        if diagnostic_data['reseasonalization_info']:
            f.write("\n\nRe-seasonalization Diagnostics (sample series):\n")
            f.write("-" * 50 + "\n")
            for diag in diagnostic_data['reseasonalization_info'][:5]:
                f.write(f"\nSeries: {diag['series_id']}\n")
                f.write(f"  Needed Forecasting: {diag['needs_forecasting']}\n")
                f.write(f"  Seasonal Component Mean: {diag['seasonal_mean']:.4f} ± {diag['seasonal_std']:.4f}\n")
                f.write(f"  Deseasonalized Forecast Mean: {diag['deseasonalized_forecast_mean']:.4f}\n")
                f.write(f"  Re-seasonalized Forecast Mean: {diag['reseasonalized_forecast_mean']:.4f}\n")
                f.write(f"  True Values Mean: {diag['true_values_mean']:.4f}\n")
    
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Save detailed diagnostics to JSON file
    diagnostics_path = output_dir / 'evaluation_diagnostics.json'
    diagnostics_json = {
        'mase_components': diagnostic_data['mase_components'],
        'reseasonalization_info': diagnostic_data['reseasonalization_info'],
        'scale_comparisons': diagnostic_data['scale_comparisons'],
        'scale_analysis': diagnostic_data['scale_analysis']
    }
    with open(diagnostics_path, 'w') as f:
        json.dump(diagnostics_json, f, indent=2, default=str)
    print(f"Detailed diagnostics saved to: {diagnostics_path}")
    
    # Save predictions to CSV (human readable, using original scale)
    predictions_rows = []
    for series_id in predictions.keys():
        pred = predictions[series_id]  # Already re-seasonalized (original scale)
        true = original_true_values.get(series_id, np.array([]))  # Original scale
        
        if len(true) != len(pred):
            continue
        
        for t in range(len(pred)):
            predictions_rows.append({
                'series_id': series_id,
                'horizon': t + 1,
                'predicted': pred[t],
                'actual': true[t],
                'error': pred[t] - true[t],
                'abs_error': abs(pred[t] - true[t])
            })
    
    predictions_df = pd.DataFrame(predictions_rows)
    predictions_path = output_dir / 'predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to: {predictions_path}")
    
    # Save summary by series (using original scale)
    from delphi.evaluation.metrics import mase as mase_func
    summary_rows = []
    for series_id in predictions.keys():
        pred = predictions[series_id]  # Already re-seasonalized (original scale)
        true = original_true_values.get(series_id, np.array([]))  # Original scale
        
        if len(true) != len(pred):
            continue
        
        mse_val = np.mean((pred - true) ** 2)
        mae_val = np.mean(np.abs(pred - true))
        mase_val = np.nan
        train_series = original_train_histories.get(series_id)  # Original scale training data
        if train_series is not None and len(train_series) > seasonal_period:
            mase_val = mase_func(true, pred, train_series, seasonal_period, forecast_horizon)
        
        summary_rows.append({
            'series_id': series_id,
            'mse': mse_val,
            'mae': mae_val,
            'mase': mase_val
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / 'series_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Series summary saved to: {summary_path}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()

