"""
Production Inference script for DELPHI.

Provides full DELPHI system inference including:
- DELPHI core model predictions with uncertainty quantification
- Meta-ensemble integration (PatchTST, N-BEATS, N-HiTS, Chronos-2, xLSTMTime, ETS, SNAIVE)
- MinT hierarchical reconciliation
- Multiple output formats (JSON, CSV, NPY)
"""

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, Optional, Any, Union
from datetime import datetime
import warnings

from delphi.inference.predictor import DELPHIPredictor
from delphi.models.delphi_core import DELPHICore
from delphi.models import MetaEnsemble, MinTReconciliation
from delphi.models.parametric import TBATSBaseline
from delphi.data import HERMESDataLoader, preprocess_hermes_data

warnings.filterwarnings('ignore')


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


def load_production_data(
    data_path: str,
    config: Dict,
    verbose: bool = True
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load and validate production data for inference.
    
    Args:
        data_path: Path to production data (CSV, pickle, or directory)
        config: Configuration dictionary
        verbose: Print progress messages
    
    Returns:
        Dictionary with preprocessed data:
            - 'main_signal': Main time series
            - 'weak_signal_ratio': Weak signal ratios (if available)
            - 'series_ids': List of series IDs
    """
    if verbose:
        print(f"Loading production data from: {data_path}")
    
    # Initialize loader with data path
    loader = HERMESDataLoader(data_dir=data_path)
    
    # Try to load data - loader auto-detects format
    try:
        data = loader.load_all_series(load_weak_signal=True)
    except FileNotFoundError:
        # Try as direct file path if directory load failed
        data_file = Path(data_path)
        if data_file.exists() and data_file.suffix == '.csv':
            # Load single CSV file
            loader = HERMESDataLoader(data_dir=str(data_file.parent))
            data = loader.load_from_csv_files(
                main_csv=data_file.name,
                load_weak_signal=False
            )
        elif data_file.exists() and data_file.suffix == '.pkl':
            loader = HERMESDataLoader(data_dir=str(data_file.parent))
            data = loader.load_from_pickle(data_file.name, load_weak_signal=False)
        else:
            raise FileNotFoundError(
                f"Could not load data from {data_path}. "
                "Supported formats: CSV files, pickle files, or data directory."
            )
    
    if verbose:
        print(f"Loaded {len(data['main_signal'])} time series")
    
    # Preprocess data
    if verbose:
        print("Preprocessing data...")
    
    preprocessed = preprocess_hermes_data(
        data,
        deseasonalize=config['preprocessing'].get('deseasonalize', True),
        compute_weak_ratio=config['preprocessing'].get('compute_weak_ratio', True),
        seasonal_period=_ensure_numeric(config['data'].get('seasonal_period', 52), 52, 'int'),
        verbose=False
    )
    
    # Validate data length
    context_length = 52  # Required context length
    valid_series = {}
    skipped_series = []
    
    for series_id, ts in preprocessed['main_signal'].items():
        if len(ts) >= context_length:
            valid_series[series_id] = ts
        else:
            skipped_series.append(series_id)
    
    if skipped_series and verbose:
        print(f"Warning: {len(skipped_series)} series skipped (less than {context_length} data points)")
    
    if len(valid_series) == 0:
        raise ValueError(
            f"No series have sufficient data length (minimum {context_length} points required). "
            f"Check your input data."
        )
    
    if verbose:
        print(f"Valid series for inference: {len(valid_series)}")
    
    return {
        'main_signal': valid_series,
        'weak_signal_ratio': preprocessed.get('weak_signal_ratio', {}),
        'series_ids': list(valid_series.keys())
    }


def load_meta_ensemble(
    meta_ensemble_path: str,
    config: Dict,
    delphi_model: DELPHICore,
    device: str,
    verbose: bool = True
) -> Optional[MetaEnsemble]:
    """
    Load meta-ensemble from checkpoint if available.
    
    Args:
        meta_ensemble_path: Path to meta-ensemble checkpoint
        config: Configuration dictionary
        delphi_model: Trained DELPHI core model
        device: Device for inference
        verbose: Print progress messages
    
    Returns:
        MetaEnsemble instance or None if not available
    """
    if not os.path.exists(meta_ensemble_path):
        if verbose:
            print(f"Meta-ensemble checkpoint not found at {meta_ensemble_path}")
            print("Running inference with DELPHI core only.")
        return None
    
    try:
        if verbose:
            print(f"Loading meta-ensemble from: {meta_ensemble_path}")
        
        checkpoint = torch.load(meta_ensemble_path, map_location=device, weights_only=False)
        
        # Get model list from checkpoint or config
        meta_models = checkpoint.get('meta_models', config.get('meta_ensemble', {}).get('models', [
            'delphi_core', 'patchtst', 'nbeats', 'nhits', 'chronos2', 'xlstmtime', 'ets', 'snaive'
        ]))
        
        # Get model configs
        model_configs = checkpoint.get('model_configs', config.get('meta_ensemble', {}).get('model_configs', {}))
        if not model_configs:
            model_configs = config.get('models', {})
        
        # Initialize meta-ensemble
        meta_ensemble = MetaEnsemble(
            models=meta_models,
            forecast_horizon=_ensure_numeric(config['data'].get('forecast_horizon', 26), 26, 'int'),
            model_configs=model_configs,
            meta_learner_config=config.get('meta_ensemble', {}).get('meta_learner', {}),
            device=device
        )
        
        # Set DELPHI core model
        meta_ensemble.set_delphi_core(delphi_model)
        
        if verbose:
            print(f"Meta-ensemble initialized with models: {', '.join(meta_models)}")
        
        return meta_ensemble
        
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to load meta-ensemble: {e}")
            print("Running inference with DELPHI core only.")
        return None


def run_inference(
    predictor: DELPHIPredictor,
    data: Dict[str, Dict[str, np.ndarray]],
    config: Dict,
    meta_ensemble: Optional[MetaEnsemble] = None,
    reconciler: Optional[MinTReconciliation] = None,
    num_samples: int = 100,
    verbose: bool = True
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run complete DELPHI inference pipeline.
    
    Args:
        predictor: DELPHI predictor instance
        data: Preprocessed data dictionary
        config: Configuration dictionary
        meta_ensemble: Optional meta-ensemble for combined forecasts
        reconciler: Optional MinT reconciler for hierarchical coherence
        num_samples: Number of samples for uncertainty quantification
        verbose: Print progress messages
    
    Returns:
        Dictionary with predictions:
            - 'mean': Mean forecasts
            - 'std': Standard deviations
            - 'lower_ci': Lower confidence interval (95%)
            - 'upper_ci': Upper confidence interval (95%)
    """
    forecast_horizon = _ensure_numeric(config['data'].get('forecast_horizon', 26), 26, 'int')
    
    # Step 1: Generate TBATS parametric forecasts
    if verbose:
        print("\nStep 1: Generating parametric baseline forecasts...")
    
    tbats = TBATSBaseline(
        use_box_cox=config['parametric'].get('use_box_cox', True),
        use_trend=config['parametric'].get('use_trend', True),
        use_arma_errors=config['parametric'].get('use_arma_errors', True),
        seasonal_periods=config['parametric'].get('seasonal_periods', [52])
    )
    
    param_forecasts, _ = tbats.fit_and_forecast(
        data['main_signal'],
        forecast_horizon=forecast_horizon,
        verbose=verbose
    )
    
    # Step 2: Generate DELPHI core predictions
    if verbose:
        print("\nStep 2: Generating DELPHI core predictions with uncertainty...")
    
    delphi_forecasts = {}
    delphi_uncertainty = {}
    
    series_ids = data['series_ids']
    total_series = len(series_ids)
    
    for idx, series_id in enumerate(series_ids, 1):
        main_signal = data['main_signal'][series_id]
        weak_ratio = data['weak_signal_ratio'].get(series_id)
        param_forecast = param_forecasts.get(series_id)
        
        # Use last 52 weeks as context
        input_seq = main_signal[-52:]
        input_weak_ratio = weak_ratio[-52:] if weak_ratio is not None and len(weak_ratio) >= 52 else None
        
        result = predictor.predict_series(
            main_signal=input_seq,
            weak_signal_ratio=input_weak_ratio,
            parametric_forecast=param_forecast,
            num_samples=num_samples
        )
        
        delphi_forecasts[series_id] = result['mean'].flatten()
        delphi_uncertainty[series_id] = result['std'].flatten()
        
        # Progress indicator
        if verbose and (idx % 100 == 0 or idx == total_series):
            print(f"  Processed {idx}/{total_series} series")
    
    # Step 3: Meta-ensemble combination (if available)
    if meta_ensemble is not None:
        if verbose:
            print("\nStep 3: Generating meta-ensemble predictions...")
        
        try:
            # Get predictions from all models
            all_model_forecasts = meta_ensemble.predict_all_models(
                data=data['main_signal'],
                delphi_forecast=delphi_forecasts
            )
            
            # Combine forecasts
            combined_forecasts = meta_ensemble.combine_forecasts(all_model_forecasts)
            
            if verbose:
                models_with_forecasts = [m for m, f in all_model_forecasts.items() if len(f) > 0]
                print(f"  Combined forecasts from: {', '.join(models_with_forecasts)}")
            
        except Exception as e:
            if verbose:
                print(f"  Warning: Meta-ensemble failed: {e}")
                print("  Using DELPHI core forecasts only.")
            combined_forecasts = delphi_forecasts
    else:
        combined_forecasts = delphi_forecasts
    
    # Step 4: MinT reconciliation (if hierarchy defined)
    hierarchy = config.get('reconciliation', {}).get('hierarchy', None)
    
    if reconciler is not None and hierarchy:
        if verbose:
            print("\nStep 4: Applying MinT reconciliation...")
        
        try:
            final_forecasts = reconciler.reconcile(
                forecasts=combined_forecasts,
                hierarchy=hierarchy
            )
            if verbose:
                print(f"  Reconciliation applied to {len(final_forecasts)} series")
        except Exception as e:
            if verbose:
                print(f"  Warning: Reconciliation failed: {e}")
                print("  Using unreconciled forecasts.")
            final_forecasts = combined_forecasts
    else:
        final_forecasts = combined_forecasts
        if verbose:
            if hierarchy is None:
                print("\nStep 4: No hierarchy defined - skipping reconciliation")
            else:
                print("\nStep 4: Reconciliation disabled")
    
    # Compile results
    results = {
        'mean': {},
        'std': {},
        'lower_ci': {},
        'upper_ci': {}
    }
    
    for series_id in series_ids:
        if series_id in final_forecasts:
            mean_forecast = final_forecasts[series_id]
            std_forecast = delphi_uncertainty.get(series_id, np.zeros_like(mean_forecast))
            
            results['mean'][series_id] = mean_forecast
            results['std'][series_id] = std_forecast
            results['lower_ci'][series_id] = mean_forecast - 1.96 * std_forecast
            results['upper_ci'][series_id] = mean_forecast + 1.96 * std_forecast
    
    if verbose:
        print(f"\nInference complete: {len(results['mean'])} series forecasted")
    
    return results


def save_predictions(
    predictions: Dict[str, Dict[str, np.ndarray]],
    output_path: str,
    output_format: str = 'json',
    config: Dict = None,
    verbose: bool = True
):
    """
    Save predictions to file in specified format.
    
    Args:
        predictions: Dictionary with predictions
        output_path: Path to save predictions
        output_format: Format to save (json, csv, npy)
        config: Configuration dictionary (for metadata)
        verbose: Print progress messages
    """
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Change extension based on format
    if output_format == 'json':
        output_path = output_path.with_suffix('.json')
    elif output_format == 'csv':
        output_path = output_path.with_suffix('.csv')
    elif output_format == 'npy':
        output_path = output_path.with_suffix('.npz')
    
    if verbose:
        print(f"\nSaving predictions to: {output_path}")
    
    if output_format == 'json':
        # JSON format - human readable with metadata
        output_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'model': 'DELPHI',
                'forecast_horizon': config['data']['forecast_horizon'] if config else 26,
                'num_series': len(predictions['mean']),
                'confidence_level': 0.95
            },
            'predictions': {}
        }
        
        for series_id in predictions['mean'].keys():
            output_data['predictions'][series_id] = {
                'mean': predictions['mean'][series_id].tolist(),
                'std': predictions['std'][series_id].tolist(),
                'lower_ci_95': predictions['lower_ci'][series_id].tolist(),
                'upper_ci_95': predictions['upper_ci'][series_id].tolist()
            }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    elif output_format == 'csv':
        # CSV format - flat table format
        rows = []
        
        for series_id in predictions['mean'].keys():
            mean = predictions['mean'][series_id]
            std = predictions['std'][series_id]
            lower = predictions['lower_ci'][series_id]
            upper = predictions['upper_ci'][series_id]
            
            for t in range(len(mean)):
                rows.append({
                    'series_id': series_id,
                    'horizon': t + 1,
                    'forecast_mean': mean[t],
                    'forecast_std': std[t],
                    'lower_ci_95': lower[t],
                    'upper_ci_95': upper[t]
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
    
    elif output_format == 'npy':
        # NPY format - efficient numpy format
        np.savez(
            output_path,
            series_ids=np.array(list(predictions['mean'].keys())),
            mean=np.array([predictions['mean'][sid] for sid in predictions['mean'].keys()]),
            std=np.array([predictions['std'][sid] for sid in predictions['mean'].keys()]),
            lower_ci=np.array([predictions['lower_ci'][sid] for sid in predictions['mean'].keys()]),
            upper_ci=np.array([predictions['upper_ci'][sid] for sid in predictions['mean'].keys()])
        )
    
    else:
        raise ValueError(f"Unsupported output format: {output_format}. Use 'json', 'csv', or 'npy'.")
    
    if verbose:
        print(f"Predictions saved successfully ({output_format} format)")


def main():
    parser = argparse.ArgumentParser(
        description='Run DELPHI production inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference with JSON output
  python inference.py --model_path checkpoints/delphi_final.pt --data_path data/

  # Inference with CSV output
  python inference.py --model_path checkpoints/delphi_final.pt --data_path data/ --output_format csv

  # Inference without meta-ensemble (faster)
  python inference.py --model_path checkpoints/delphi_final.pt --data_path data/ --no_meta_ensemble

  # Custom output path and more uncertainty samples
  python inference.py --model_path checkpoints/delphi_final.pt --data_path data/ \\
      --output_path results/forecasts.json --num_samples 200
        """
    )
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to DELPHI core model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to production data (CSV, pickle, or directory)')
    
    # Optional arguments
    parser.add_argument('--config', type=str, default='configs/delphi_config.yaml',
                        help='Path to config file (default: configs/delphi_config.yaml)')
    parser.add_argument('--output_path', type=str, default='predictions.json',
                        help='Path to save predictions (default: predictions.json)')
    parser.add_argument('--output_format', type=str, default='json',
                        choices=['json', 'csv', 'npy'],
                        help='Output format: json, csv, or npy (default: json)')
    parser.add_argument('--meta_ensemble_path', type=str, default=None,
                        help='Path to meta-ensemble checkpoint (auto-detected if not specified)')
    parser.add_argument('--no_meta_ensemble', action='store_true',
                        help='Disable meta-ensemble (use DELPHI core only)')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples for uncertainty quantification (default: 100)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress messages')
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    # Print header
    if verbose:
        print("=" * 70)
        print("DELPHI PRODUCTION INFERENCE")
        print("=" * 70)
    
    # Load config
    if verbose:
        print(f"\nLoading configuration from: {args.config}")
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        raise SystemExit(1)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure numeric config values
    config['data']['forecast_horizon'] = _ensure_numeric(
        config['data'].get('forecast_horizon', 26), 26, 'int'
    )
    config['data']['seasonal_period'] = _ensure_numeric(
        config['data'].get('seasonal_period', 52), 52, 'int'
    )
    
    # Set device
    requested_device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if requested_device == 'cuda' and not torch.cuda.is_available():
        if verbose:
            print("Warning: CUDA requested but not available. Using CPU.")
        device = 'cpu'
    else:
        device = requested_device if torch.cuda.is_available() else 'cpu'
    
    if verbose:
        print(f"Using device: {device}")
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found: {args.model_path}")
        raise SystemExit(1)
    
    # Build model config
    model_config = {
        'input_dim': 3,
        'n_states': _ensure_numeric(config['hmm']['n_states'], 4, 'int'),
        'hmm_hidden_size': _ensure_numeric(config['hmm']['hidden_size'], 64, 'int'),
        'hmm_num_layers': _ensure_numeric(config['hmm']['num_layers'], 2, 'int'),
        'ensemble_hidden_size': _ensure_numeric(config['ensemble']['hidden_size'], 64, 'int'),
        'ensemble_num_layers': _ensure_numeric(config['ensemble']['num_layers'], 2, 'int'),
        'output_dim': _ensure_numeric(config['data']['forecast_horizon'], 26, 'int'),
        'n_ensemble_members': _ensure_numeric(config['ensemble']['n_members'], 5, 'int'),
        'dropout': _ensure_numeric(config['hmm']['dropout'], 0.2),
        'use_xlstm_for_trend': config['ensemble'].get('use_xlstm_for_trend', False)
    }
    
    # Load DELPHI predictor
    if verbose:
        print(f"\nLoading DELPHI model from: {args.model_path}")
    
    try:
        predictor = DELPHIPredictor.from_checkpoint(
            args.model_path,
            model_config,
            device=device
        )
        if verbose:
            print("DELPHI core model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise SystemExit(1)
    
    # Load meta-ensemble (if enabled)
    meta_ensemble = None
    if not args.no_meta_ensemble:
        # Auto-detect meta-ensemble path if not specified
        if args.meta_ensemble_path:
            meta_path = args.meta_ensemble_path
        else:
            # Try same directory as model checkpoint
            model_dir = Path(args.model_path).parent
            meta_path = model_dir / 'meta_ensemble.pt'
        
        meta_ensemble = load_meta_ensemble(
            str(meta_path),
            config,
            predictor.model,
            device,
            verbose=verbose
        )
    elif verbose:
        print("\nMeta-ensemble disabled by user request")
    
    # Initialize reconciler
    reconciler = MinTReconciliation(
        method=config.get('reconciliation', {}).get('method', 'mint'),
        hierarchy_levels=config.get('reconciliation', {}).get('hierarchy_levels', [])
    )
    
    # Load production data
    try:
        data = load_production_data(args.data_path, config, verbose=verbose)
    except Exception as e:
        print(f"Error loading data: {e}")
        raise SystemExit(1)
    
    # Run inference
    if verbose:
        print("\n" + "=" * 70)
        print("RUNNING INFERENCE")
        print("=" * 70)
    
    try:
        predictions = run_inference(
            predictor=predictor,
            data=data,
            config=config,
            meta_ensemble=meta_ensemble,
            reconciler=reconciler,
            num_samples=args.num_samples,
            verbose=verbose
        )
    except Exception as e:
        print(f"Error during inference: {e}")
        raise SystemExit(1)
    
    # Save predictions
    try:
        save_predictions(
            predictions,
            args.output_path,
            args.output_format,
            config,
            verbose=verbose
        )
    except Exception as e:
        print(f"Error saving predictions: {e}")
        raise SystemExit(1)
    
    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("INFERENCE COMPLETE")
        print("=" * 70)
        print(f"\nSummary:")
        print(f"  Series forecasted: {len(predictions['mean'])}")
        print(f"  Forecast horizon: {config['data']['forecast_horizon']} periods")
        print(f"  Output file: {args.output_path}")
        print(f"  Output format: {args.output_format}")
        print(f"  Meta-ensemble: {'Enabled' if meta_ensemble else 'Disabled'}")
        print(f"  Uncertainty samples: {args.num_samples}")
        print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
