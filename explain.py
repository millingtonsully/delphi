"""
Explanation script for DELPHI model predictions.

Reads evaluation_results/predictions.csv to identify series to explain,
then generates comprehensive explanations using the xAIUQ layer.
"""

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
import os
from typing import Any, Union, Optional, Dict

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Memory monitoring disabled.")

from delphi.models.delphi_core import DELPHICore
from delphi.models.parametric import TBATSBaseline
from delphi.data import (
    DataLoader,
    preprocess_data,
    create_train_val_test_split
)
from delphi.data.preprocessing import prepare_model_inputs
from xAIUQ import DelphiExplainer
from xAIUQ.result_loader import (
    get_series_ids,
    get_series_horizons,
    filter_series_by_criteria,
    get_evaluation_metadata
)


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


def get_memory_usage_mb() -> float:
    """
    Get current memory usage in MB.
    
    Returns:
        Memory usage in megabytes (MB), or 0.0 if psutil is not available
    """
    if not PSUTIL_AVAILABLE:
        return 0.0
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert bytes to MB


def check_memory_threshold(current_memory_mb: float, warning_threshold_mb: float = 8192, 
                           critical_threshold_mb: float = 16384) -> bool:
    """
    Check if memory usage exceeds thresholds and alert.
    
    Args:
        current_memory_mb: Current memory usage in MB
        warning_threshold_mb: Memory threshold for warning (default: 8GB)
        critical_threshold_mb: Memory threshold for critical alert (default: 16GB)
    
    Returns:
        True if memory usage is critical, False otherwise
    """
    if current_memory_mb >= critical_threshold_mb:
        print(f"    ⚠️  CRITICAL: Memory usage {current_memory_mb:.1f} MB exceeds critical threshold {critical_threshold_mb:.1f} MB")
        return True
    elif current_memory_mb >= warning_threshold_mb:
        print(f"    ⚠️  WARNING: Memory usage {current_memory_mb:.1f} MB exceeds warning threshold {warning_threshold_mb:.1f} MB")
        return False
    return False


def load_model(checkpoint_path: str, model_config: Dict, device: str) -> DELPHICore:
    """
    Load DELPHI model from checkpoint.
    
    Handles compatibility with older checkpoint formats.
    
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
    
    state_dict = checkpoint['model_state_dict']
    
    # Try loading with strict=False first (handles architecture mismatches)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        # If strict loading fails, try with strict=False
        print(f"Warning: Some state_dict keys don't match. Attempting flexible loading...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys (will use random initialization): {len(missing_keys)} keys")
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    print(f"  - {key}")
        
        if unexpected_keys:
            print(f"Unexpected keys (ignored): {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    print(f"  - {key}")
    
    model.to(device)
    model.eval()
    return model


def save_explanation_report(
    report: Dict,
    series_id: str,
    output_dir: str,
    component: str = 'complete'
):
    """
    Save explanation report to JSON file.
    
    Args:
        report: Explanation report dictionary
        series_id: Series identifier
        output_dir: Output directory
        component: Component name (e.g., 'regime_shifts', 'complete')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return obj
    
    serializable_report = convert_to_serializable(report)
    
    filename = f"{series_id}_{component}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_report, f, indent=2)
    
    return filepath


def main():
    parser = argparse.ArgumentParser(description='Explain DELPHI model predictions')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--results_dir', type=str, default='evaluation_results',
                       help='Directory containing evaluation results')
    parser.add_argument('--config', type=str, default='configs/delphi_config.yaml',
                       help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='explanation_results',
                       help='Directory to save explanation results')
    parser.add_argument('--num_series', type=int, default=None,
                       help='Maximum number of series to explain (default: all)')
    parser.add_argument('--series_id', type=str, default=None,
                       help='Specific series ID to explain (overrides num_series)')
    parser.add_argument('--components', type=str, nargs='+', 
                       default=['all'],
                       choices=['all', 'regime_shifts', 'features', 'regime', 'external_signal', 'uncertainty'],
                       help='Which explanation components to generate')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure numeric values are properly typed
    forecast_horizon = _ensure_numeric(config['data']['forecast_horizon'], 26, 'int')
    train_end_week = _ensure_numeric(config['data'].get('train_end_week', 209), 209, 'int')
    val_end_week = _ensure_numeric(config['data'].get('val_end_week', 261), 261, 'int')
    seasonal_period = _ensure_numeric(config['data'].get('seasonal_period', 52), 52, 'int')
    context_length = 52  # Fixed context length
    
    # Create output directory structure
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    explanations_dir = output_dir / "explanations_by_series"
    explanations_dir.mkdir(exist_ok=True)
    
    # Initialize device (force CPU for explanations to avoid GPU OOM)
    device = 'cpu'
    print("Forcing device to CPU for explanations to avoid GPU out-of-memory issues.")
    print(f"Using device: {device}")
    
    # Model configuration
    model_config = {
        'input_dim': 4,
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
    model = load_model(args.model_path, model_config, device)
    print("Model loaded successfully!")
    
    # Initialize explainer
    explainer = DelphiExplainer(model, device=device)
    
    # Get series to explain
    if args.series_id:
        series_to_explain = [args.series_id]
    else:
        series_to_explain = filter_series_by_criteria(
            results_dir=args.results_dir,
            max_series=args.num_series
        )
    
    print(f"\nGenerating explanations for {len(series_to_explain)} series...")
    
    # Load data
    print("\nLoading data...")
    data_dir = config['data']['data_dir']
    loader = DataLoader(data_dir=data_dir)
    data = loader.load_all_series(load_weak_signal=True)
    
    # Preprocess
    print("Preprocessing data...")
    preprocessed = preprocess_data(
        data,
        deseasonalize=config['preprocessing'].get('deseasonalize', True),
        compute_weak_ratio=config['preprocessing'].get('compute_weak_ratio', True),
        seasonal_period=seasonal_period,
        verbose=False
    )
    
    # Create splits
    splits = create_train_val_test_split(
        preprocessed,
        train_end_week=train_end_week,
        val_end_week=val_end_week,
        forecast_horizon=forecast_horizon
    )
    
    train_split = splits['train']
    val_split = splits['val']
    
    train_main = train_split['main_signal']
    val_main = val_split['main_signal']
    weak_train = train_split.get('weak_signal_ratio', {})
    weak_val = val_split.get('weak_signal_ratio', {})
    
    # Generate parametric baseline forecasts
    print("\nGenerating TBATS baseline forecasts...")
    tbats = TBATSBaseline(
        use_box_cox=config['parametric'].get('use_box_cox', True),
        use_trend=config['parametric'].get('use_trend', True),
        use_arma_errors=config['parametric'].get('use_arma_errors', True),
        seasonal_periods=config['parametric'].get('seasonal_periods', [52]),
        n_parallel_workers=8
    )
    
    # Prepare TBATS input
    tbats_input = {}
    for series_id in series_to_explain:
        history_main = np.concatenate([
            train_main.get(series_id, np.array([])),
            val_main.get(series_id, np.array([]))
        ])
        if len(history_main) >= context_length:
            tbats_input[series_id] = history_main
    
    param_forecasts, param_residuals = tbats.fit_and_forecast(
        tbats_input,
        forecast_horizon=forecast_horizon,
        verbose=False
    )
    
    # Generate explanations
    print("\nGenerating explanations...")
    all_explanations = []
    
    for idx, series_id in enumerate(series_to_explain, 1):
        print(f"  Processing {idx}/{len(series_to_explain)}: {series_id}")
        
        # Memory monitoring: before processing
        memory_before = get_memory_usage_mb()
        print(f"    Memory before: {memory_before:.1f} MB")
        is_critical = check_memory_threshold(memory_before)
        if is_critical:
            print(f"    ⚠️  Skipping {series_id} due to critical memory usage")
            continue
        
        train_history = train_main.get(series_id, np.array([]))
        val_history = val_main.get(series_id, np.array([]))
        history_main = np.concatenate([train_history, val_history])
        
        if len(history_main) < context_length:
            print(f"    Skipping {series_id}: insufficient history")
            continue
        
        # Prepare model input
        input_seq = history_main[-context_length:]
        param_forecast = param_forecasts.get(series_id)
        series_residuals = param_residuals.get(series_id)
        
        # Get weak ratio history
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
        
        # Get residuals for input context
        input_residuals = None
        if series_residuals is not None and len(series_residuals) >= context_length:
            input_residuals = series_residuals[-context_length:]
        
        # Prepare model inputs
        input_tensor = prepare_model_inputs(
            input_seq,
            residuals=input_residuals,
            weak_signal_ratio=input_weak_ratio,
            parametric_forecast=param_forecast[:forecast_horizon] if param_forecast is not None else None
        )
        
        # Convert to tensor
        inputs = torch.FloatTensor(input_tensor).unsqueeze(0).to(device)
        if param_forecast is not None:
            param_tensor = torch.FloatTensor(param_forecast[:forecast_horizon]).unsqueeze(0).to(device)
        else:
            param_tensor = None
        
        # Generate explanations
        try:
            if 'all' in args.components:
                print("    Running full explanation (regime_shifts, features, regime, external_signal, uncertainty)...")
                report = explainer.explain(inputs, param_tensor, include_uncertainty=True, series_id=series_id)
                explanation_dict = report.to_dict()
            else:
                explanation_dict = {'series_id': series_id}
                
                if 'regime_shifts' in args.components:
                    print("    Running component: regime_shifts")
                    explanation_dict['regime_shifts'] = explainer.explain_regime_shifts(inputs, param_tensor)
                
                if 'features' in args.components:
                    print("    Running component: features")
                    explanation_dict['feature_attribution'] = explainer.explain_features(inputs, param_tensor)
                
                if 'regime' in args.components:
                    print("    Running component: regime")
                    explanation_dict['regime_explanation'] = explainer._get_regime_explanation(inputs, param_tensor)
                
                if 'external_signal' in args.components:
                    print("    Running component: external_signal")
                    explanation_dict['external_signal'] = explainer.explain_external_signal(inputs, param_tensor)
                
                if 'uncertainty' in args.components:
                    print("    Running component: uncertainty")
                    explanation_dict['uncertainty'] = explainer.quantify_uncertainty(inputs, param_tensor)
            
            # Save individual components
            if 'regime_shifts' in explanation_dict:
                save_explanation_report(
                    explanation_dict['regime_shifts'],
                    series_id,
                    str(explanations_dir),
                    'regime_shifts'
                )
            
            if 'feature_attribution' in explanation_dict:
                save_explanation_report(
                    explanation_dict['feature_attribution'],
                    series_id,
                    str(explanations_dir),
                    'feature_attribution'
                )
            
            if 'uncertainty' in explanation_dict:
                save_explanation_report(
                    explanation_dict['uncertainty'],
                    series_id,
                    str(explanations_dir),
                    'uncertainty'
                )
            
            if 'external_signal' in explanation_dict:
                save_explanation_report(
                    explanation_dict['external_signal'],
                    series_id,
                    str(explanations_dir),
                    'external_signal'
                )
            
            # Save complete report
            save_explanation_report(
                explanation_dict,
                series_id,
                str(explanations_dir),
                'complete'
            )
            
            all_explanations.append(explanation_dict)
            
            # Memory cleanup after each series
            del explanation_dict, inputs, param_tensor
            if 'all' in args.components:
                try:
                    del report
                except NameError:
                    pass  # report not defined if using individual components
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear CUDA cache if available
            
            # Memory monitoring: after processing
            memory_after = get_memory_usage_mb()
            memory_delta = memory_after - memory_before
            print(f"    Memory after: {memory_after:.1f} MB (Δ {memory_delta:+.1f} MB)")
            check_memory_threshold(memory_after)
            
        except Exception as e:
            print(f"    Error explaining {series_id}: {e}")
            # Memory cleanup on error
            try:
                del explanation_dict, inputs, param_tensor
                if 'all' in args.components:
                    try:
                        del report
                    except NameError:
                        pass
            except NameError:
                pass  # Variables may not exist if error occurred early
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Memory monitoring: after error
            memory_after = get_memory_usage_mb()
            memory_delta = memory_after - memory_before
            print(f"    Memory after error: {memory_after:.1f} MB (Δ {memory_delta:+.1f} MB)")
            continue
    
    # Save summary
    summary = {
        'num_series_explained': len(all_explanations),
        'series_ids': [exp['series_id'] for exp in all_explanations],
        'components_generated': args.components,
        'model_path': args.model_path,
        'results_dir': args.results_dir
    }
    
    summary_path = output_dir / "explanations_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print("EXPLANATION GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nExplanations saved to: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"Individual explanations: {explanations_dir}")
    print(f"\nExplained {len(all_explanations)} series successfully")


if __name__ == '__main__':
    main()

