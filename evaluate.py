"""
Evaluation script for DELPHI.

Uses holdout evaluation: last forecast_horizon weeks as test data.
Outputs metrics to text file and predictions to CSV for readability.
"""

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import os
from typing import Any, Union, Optional, Dict

from delphi.inference.predictor import DELPHIPredictor
from delphi.models.delphi_core import DELPHICore
from delphi.models.parametric import TBATSBaseline
from delphi.data import HERMESDataLoader, preprocess_hermes_data
from delphi.evaluation.metrics import compute_all_metrics


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


def main():
    parser = argparse.ArgumentParser(description='Evaluate DELPHI model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, default=None,
                       help='Path to test data directory (if None, uses config data_dir)')
    parser.add_argument('--config', type=str, default='configs/delphi_config.yaml',
                       help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--num_series', type=int, default=100,
                       help='Maximum number of series to evaluate (default: 100)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure numeric values are properly typed
    forecast_horizon = _ensure_numeric(config['data']['forecast_horizon'], 26, 'int')
    context_length = 52  # Fixed context length
    min_series_length = context_length + forecast_horizon
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    model_config = {
        'input_dim': 3,
        'n_states': config['hmm']['n_states'],
        'hmm_hidden_size': config['hmm']['hidden_size'],
        'hmm_num_layers': config['hmm']['num_layers'],
        'ensemble_hidden_size': config['ensemble']['hidden_size'],
        'ensemble_num_layers': config['ensemble']['num_layers'],
        'output_dim': forecast_horizon,
        'n_ensemble_members': config['ensemble']['n_members'],
        'dropout': config['hmm']['dropout'],
        'use_xlstm_for_trend': config['ensemble'].get('use_xlstm_for_trend', False)
    }
    
    # Load predictor
    print("Loading DELPHI model...")
    print(f"  Device: {device}")
    predictor = DELPHIPredictor.from_checkpoint(
        args.model_path,
        model_config,
        device=device
    )
    print("  Model loaded successfully!")
    
    # Load data
    print("\nLoading data...")
    data_dir = args.test_data if args.test_data else config['data']['data_dir']
    loader = HERMESDataLoader(data_dir=data_dir)
    data = loader.load_all_series(load_weak_signal=True)
    
    # Preprocess
    print("Preprocessing data...")
    preprocessed = preprocess_hermes_data(
        data,
        deseasonalize=config['preprocessing'].get('deseasonalize', True),
        compute_weak_ratio=config['preprocessing'].get('compute_weak_ratio', True),
        seasonal_period=_ensure_numeric(config['data'].get('seasonal_period', 52), 52, 'int'),
        verbose=False
    )
    
    main_signals = preprocessed['main_signal']
    weak_ratios = preprocessed.get('weak_signal_ratio', {})
    
    # Filter series with sufficient length for holdout evaluation
    valid_series = [
        sid for sid, ts in main_signals.items() 
        if len(ts) >= min_series_length
    ]
    
    if not valid_series:
        print(f"\nERROR: No series have sufficient length (need {min_series_length} points).")
        print(f"  Context length: {context_length}")
        print(f"  Forecast horizon: {forecast_horizon}")
        return
    
    # Limit number of series
    eval_series = valid_series[:args.num_series]
    print(f"\nEvaluating on {len(eval_series)} series...")
    if len(valid_series) > args.num_series:
        print(f"  (Limited from {len(valid_series)} valid series)")
    
    # Generate parametric baseline forecasts
    print("\nGenerating TBATS baseline forecasts...")
    tbats = TBATSBaseline(
        use_box_cox=config['parametric'].get('use_box_cox', True),
        use_trend=config['parametric'].get('use_trend', True),
        use_arma_errors=config['parametric'].get('use_arma_errors', True),
        seasonal_periods=config['parametric'].get('seasonal_periods', [52])
    )
    
    # Use all data except last forecast_horizon for TBATS fitting
    tbats_input = {sid: main_signals[sid][:-forecast_horizon] for sid in eval_series}
    param_forecasts, _ = tbats.fit_and_forecast(
        tbats_input,
        forecast_horizon=forecast_horizon,
        verbose=False
    )
    
    # Run evaluation
    print("\nRunning DELPHI predictions...")
    predictions = {}
    uncertainties = {}
    true_values = {}
    train_data = {}
    
    num_samples = _ensure_numeric(config.get('inference', {}).get('num_samples', 100), 100, 'int')
    
    for idx, series_id in enumerate(eval_series, 1):
        main_signal = main_signals[series_id]
        weak_ratio = weak_ratios.get(series_id)
        param_forecast = param_forecasts.get(series_id)
        
        # Holdout: last forecast_horizon as test, previous context_length as input
        input_seq = main_signal[-(context_length + forecast_horizon):-forecast_horizon]
        target_seq = main_signal[-forecast_horizon:]
        
        # Store ground truth and training data
        true_values[series_id] = target_seq
        train_data[series_id] = main_signal[:-forecast_horizon]
        
        # Get weak ratio for input if available
        input_weak_ratio = None
        if weak_ratio is not None and len(weak_ratio) >= context_length + forecast_horizon:
            input_weak_ratio = weak_ratio[-(context_length + forecast_horizon):-forecast_horizon]
        
        # Predict
        result = predictor.predict_series(
            main_signal=input_seq,
            weak_signal_ratio=input_weak_ratio,
            parametric_forecast=param_forecast,
            num_samples=num_samples
        )
        
        predictions[series_id] = result['mean'].flatten()
        uncertainties[series_id] = result['std'].flatten()
        
        if idx % 20 == 0 or idx == len(eval_series):
            print(f"  Processed {idx}/{len(eval_series)} series")
    
    # Compute metrics
    print("\n" + "="*70)
    print("EVALUATION METRICS")
    print("="*70)
    
    metrics = compute_all_metrics(
        y_true=true_values,
        y_pred=predictions,
        y_train=train_data,
        y_std=uncertainties,
        confidence_level=config.get('inference', {}).get('confidence_level', 0.95)
    )
    
    # Print results
    print("\nResults:")
    print("-" * 40)
    for metric, value in metrics.items():
        if isinstance(value, float) and not np.isnan(value):
            print(f"  {metric.upper():20s}: {value:.4f}")
    
    # Save metrics to text file
    metrics_path = os.path.join(args.output_dir, 'evaluation_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("DELPHI Model Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Series evaluated: {len(eval_series)}\n")
        f.write(f"Forecast horizon: {forecast_horizon}\n")
        f.write(f"Context length: {context_length}\n\n")
        f.write("Metrics:\n")
        f.write("-" * 50 + "\n")
        for metric, value in metrics.items():
            if isinstance(value, float) and not np.isnan(value):
                f.write(f"  {metric.upper():20s}: {value:.4f}\n")
    
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Save predictions to CSV (human readable)
    predictions_rows = []
    for series_id in predictions.keys():
        pred = predictions[series_id]
        true = true_values[series_id]
        std = uncertainties[series_id]
        
        for t in range(len(pred)):
            predictions_rows.append({
                'series_id': series_id,
                'horizon': t + 1,
                'predicted': pred[t],
                'actual': true[t],
                'std': std[t],
                'lower_95': pred[t] - 1.96 * std[t],
                'upper_95': pred[t] + 1.96 * std[t],
                'error': pred[t] - true[t],
                'abs_error': abs(pred[t] - true[t])
            })
    
    predictions_df = pd.DataFrame(predictions_rows)
    predictions_path = os.path.join(args.output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to: {predictions_path}")
    
    # Save summary by series
    summary_rows = []
    for series_id in predictions.keys():
        pred = predictions[series_id]
        true = true_values[series_id]
        
        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        
        # Direction accuracy
        last_known = train_data[series_id][-1]
        true_dir = np.sign(true - last_known)
        pred_dir = np.sign(pred - last_known)
        pda = np.mean(true_dir == pred_dir)
        
        summary_rows.append({
            'series_id': series_id,
            'mae': mae,
            'rmse': rmse,
            'pda': pda
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, 'series_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Series summary saved to: {summary_path}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
