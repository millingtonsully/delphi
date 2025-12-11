"""
Main training script for DELPHI.

Implements training methodology:
- Temporal train/val/test splits (no data leakage)
- Rolling origin evaluation for test set
- Proper weak signal (fashion forward) handling
- Two-stage ELBO training for variational HMM
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import os
import sys
from typing import Optional, Dict, Any, Union

# Add project root to path for imports when running from subdirectory
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from delphi.data import DataLoader, create_train_val_test_split, preprocess_data
from delphi.data.preprocessing import prepare_model_inputs
from delphi.models.parametric import TBATSBaseline
from delphi.models.delphi_core import DELPHICore
from delphi.training.trainer import DELPHITrainer, TimeSeriesDataset
from torch.utils.data import DataLoader as TorchDataLoader


def _ensure_numeric(
    value: Any, 
    default: Optional[Union[int, float]] = None, 
    min_val: Optional[float] = None, 
    max_val: Optional[float] = None,
    value_type: str = 'float'
) -> Union[int, float]:
    """
    Convert config value to numeric type with validation.
    
    YAML's safe_load may parse scientific notation (e.g., 1e-3) as strings.
    This function ensures all numeric config values are properly typed.
    
    Args:
        value: Value to convert (may be string, int, or float)
        default: Default value if input is None
        min_val: Minimum allowed value (optional)
        max_val: Maximum allowed value (optional)
        value_type: 'float' or 'int'
    
    Returns:
        Numeric value (float or int)
    
    Raises:
        ValueError: If value cannot be converted or is out of range
    """
    if value is None:
        if default is not None:
            return default
        raise ValueError("Value is None and no default provided")
    
    # Convert to numeric
    try:
        if value_type == 'int':
            result = int(float(value))  # Handle "1e3" -> 1000
        else:
            result = float(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot convert '{value}' to {value_type}: {e}")
    
    # Validate range
    if min_val is not None and result < min_val:
        raise ValueError(f"Value {result} is below minimum {min_val}")
    if max_val is not None and result > max_val:
        raise ValueError(f"Value {result} is above maximum {max_val}")
    
    return result


def _validate_and_convert_config(config: Dict) -> Dict:
    """
    Validate and convert all numeric config values to proper types.
    
    This ensures YAML parsing issues (like 1e-3 as string) don't cause runtime errors.
    """
    # Training hyperparameters
    config['training']['learning_rate'] = _ensure_numeric(
        config['training']['learning_rate'], default=1e-3, min_val=1e-8, max_val=1.0
    )
    config['training']['weight_decay'] = _ensure_numeric(
        config['training']['weight_decay'], default=1e-5, min_val=0.0
    )
    config['training']['kl_weight'] = _ensure_numeric(
        config['training']['kl_weight'], default=0.1, min_val=0.0
    )
    config['training']['entropy_weight'] = _ensure_numeric(
        config['training']['entropy_weight'], default=0.01, min_val=0.0
    )
    config['training']['batch_size'] = _ensure_numeric(
        config['training']['batch_size'], default=32, min_val=1, value_type='int'
    )
    config['training']['stage1_epochs'] = _ensure_numeric(
        config['training']['stage1_epochs'], default=50, min_val=1, value_type='int'
    )
    config['training']['stage2_epochs'] = _ensure_numeric(
        config['training']['stage2_epochs'], default=30, min_val=1, value_type='int'
    )
    config['training']['early_stopping_patience'] = _ensure_numeric(
        config['training'].get('early_stopping_patience', 10), default=10, min_val=1, value_type='int'
    )
    config['training']['early_stopping_min_delta'] = _ensure_numeric(
        config['training'].get('early_stopping_min_delta', 1e-4), default=1e-4, min_val=0.0
    )

    # KL annealing configuration (optional)
    training_cfg = config['training']
    training_cfg['kl_anneal'] = bool(training_cfg.get('kl_anneal', False))
    training_cfg['kl_start'] = _ensure_numeric(
        training_cfg.get('kl_start', training_cfg['kl_weight']),
        default=training_cfg['kl_weight'],
        min_val=0.0
    )
    training_cfg['kl_end'] = _ensure_numeric(
        training_cfg.get('kl_end', training_cfg['kl_weight']),
        default=training_cfg['kl_weight'],
        min_val=0.0
    )
    training_cfg['kl_warmup_epochs'] = _ensure_numeric(
        training_cfg.get('kl_warmup_epochs', 0),
        default=0,
        min_val=0,
        value_type='int'
    )
    
    # Data configuration
    config['data']['train_end_week'] = _ensure_numeric(
        config['data']['train_end_week'], default=209, min_val=1, value_type='int'
    )
    config['data']['val_end_week'] = _ensure_numeric(
        config['data']['val_end_week'], default=261, min_val=1, value_type='int'
    )
    config['data']['forecast_horizon'] = _ensure_numeric(
        config['data']['forecast_horizon'], default=26, min_val=1, value_type='int'
    )
    config['data']['seasonal_period'] = _ensure_numeric(
        config['data']['seasonal_period'], default=52, min_val=1, value_type='int'
    )
    
    # Model configuration
    config['hmm']['n_states'] = _ensure_numeric(
        config['hmm']['n_states'], default=4, min_val=2, value_type='int'
    )
    config['hmm']['hidden_size'] = _ensure_numeric(
        config['hmm']['hidden_size'], default=64, min_val=1, value_type='int'
    )
    config['hmm']['num_layers'] = _ensure_numeric(
        config['hmm']['num_layers'], default=2, min_val=1, value_type='int'
    )
    config['hmm']['dropout'] = _ensure_numeric(
        config['hmm']['dropout'], default=0.2, min_val=0.0, max_val=1.0
    )
    
    config['ensemble']['n_members'] = _ensure_numeric(
        config['ensemble']['n_members'], default=4, min_val=1, value_type='int'
    )
    config['ensemble']['hidden_size'] = _ensure_numeric(
        config['ensemble']['hidden_size'], default=64, min_val=1, value_type='int'
    )
    config['ensemble']['num_layers'] = _ensure_numeric(
        config['ensemble']['num_layers'], default=2, min_val=1, value_type='int'
    )
    config['ensemble']['dropout'] = _ensure_numeric(
        config['ensemble']['dropout'], default=0.2, min_val=0.0, max_val=1.0
    )
    
    # Inference config
    if 'inference' not in config:
        config['inference'] = {}
    config['inference']['num_samples'] = _ensure_numeric(
        config['inference'].get('num_samples', 100), default=100, min_val=1, value_type='int'
    )
    
    # Validate logical constraints
    if config['data']['train_end_week'] >= config['data']['val_end_week']:
        raise ValueError(
            f"train_end_week ({config['data']['train_end_week']}) must be less than "
            f"val_end_week ({config['data']['val_end_week']})"
        )
    
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(config):
    """Prepare training data."""
    # Load data
    loader = DataLoader(data_dir=config['data']['data_dir'])
    
    print("Loading data...")
    data = loader.load_all_series(load_weak_signal=True)
    
    # Preprocess
    print("Preprocessing data (this may take a while for large datasets)...")
    preprocessed = preprocess_data(
        data,
        deseasonalize=config['preprocessing']['deseasonalize'],
        compute_weak_ratio=config['preprocessing']['compute_weak_ratio'],
        seasonal_period=config['data']['seasonal_period'],
        verbose=True  # Show progress
    )
    print(f"Preprocessing complete. Processed {len(preprocessed['main_signal'])} series.")
    
    # Split data
    splits = create_train_val_test_split(
        preprocessed,
        train_end_week=config['data']['train_end_week'],
        val_end_week=config['data']['val_end_week'],
        forecast_horizon=config['data']['forecast_horizon']
    )
    
    return splits, preprocessed


def prepare_datasets(splits, config):
    """Prepare PyTorch datasets."""
    # Get parametric forecasts
    num_train_series = len(splits['train']['main_signal'])
    print(f"Fitting TBATS models for {num_train_series} series (this may take a while)...")
    
    tbats = TBATSBaseline(
        use_box_cox=config['parametric']['use_box_cox'],
        use_trend=config['parametric']['use_trend'],
        use_arma_errors=config['parametric']['use_arma_errors'],
        seasonal_periods=config['parametric']['seasonal_periods'],
        n_parallel_workers=8
    )
    
    train_forecasts, train_residuals = tbats.fit_and_forecast(
        splits['train']['main_signal'],
        forecast_horizon=config['data']['forecast_horizon'],
        verbose=True
    )
    
    # Prepare inputs and targets
    print("Creating training sequences...")
    train_inputs = []
    train_targets = []
    train_param_forecasts = []
    
    # Limit series for sequence creation (can be adjusted)
    series_limit = 100  # Limit for now
    series_list = list(splits['train']['main_signal'].keys())[:series_limit]
    total_series = len(series_list)
    
    import time
    seq_start_time = time.time()
    
    for series_idx, series_id in enumerate(series_list, 1):
        main_signal = splits['train']['main_signal'][series_id]
        weak_ratio = splits['train'].get('weak_signal_ratio', {}).get(series_id)
        param_forecast = train_forecasts.get(series_id)
        series_residuals = train_residuals.get(series_id)
        
        if len(main_signal) < config['data']['forecast_horizon'] + 52:
            continue
        
        # Create sequences
        seq_len = 52
        for i in range(len(main_signal) - seq_len - config['data']['forecast_horizon']):
            input_seq = main_signal[i:i+seq_len]
            target_seq = main_signal[i+seq_len:i+seq_len+config['data']['forecast_horizon']]
            
            # Get residuals for this sequence window
            residual_seq = None
            if series_residuals is not None and len(series_residuals) >= i + seq_len:
                residual_seq = series_residuals[i:i+seq_len]
            
            # Prepare model input with residuals
            weak_seq = weak_ratio[i:i+seq_len] if weak_ratio is not None else None
            param_seq = param_forecast[:config['data']['forecast_horizon']] if param_forecast is not None else None
            
            input_tensor = prepare_model_inputs(
                input_seq,
                residuals=residual_seq,
                weak_signal_ratio=weak_seq,
                parametric_forecast=param_seq
            )
            
            train_inputs.append(input_tensor)
            train_targets.append(target_seq)
            if param_seq is not None:
                train_param_forecasts.append(param_seq)
        
        # Progress indicator every 10 series
        if series_idx % 10 == 0 or series_idx == total_series:
            elapsed = time.time() - seq_start_time
            print(f"Creating sequences... {series_idx}/{total_series} series processed "
                  f"({len(train_inputs)} sequences created, elapsed: {elapsed:.1f}s)")
    
    print(f"Sequence creation complete. Created {len(train_inputs)} training sequences.")
    
    # Safety check: ensure we have sequences
    if len(train_inputs) == 0:
        raise ValueError("No training sequences created! Check data length and forecast_horizon settings.")
    
    # Convert to arrays
    train_inputs = np.array(train_inputs)
    train_targets = np.array(train_targets)
    train_param_forecasts = np.array(train_param_forecasts) if train_param_forecasts else None
    
    print(f"Total sequences created: {len(train_inputs)}")
    print(f"Dataset shapes - Inputs: {train_inputs.shape}, Targets: {train_targets.shape}")
    if train_param_forecasts is not None:
        print(f"Parametric forecasts shape: {train_param_forecasts.shape}")
    
    # Split sequences into train/validation (80/20 split)
    # Note: The validation period (weeks 210-261 = 52 weeks) is too short
    # to create sequences (need seq_len + forecast_horizon = 78 weeks minimum).
    # Following standard practice, we use temporal holdout from training sequences for validation.
    # This ensures no data leakage while providing validation data for early stopping.
    print("\nSplitting sequences into training and validation sets (80/20)...")
    n_total = len(train_inputs)
    val_size = max(1, n_total // 5)  # 20% for validation
    
    # Use last portion for validation (temporal split)
    val_indices = np.arange(n_total - val_size, n_total)
    train_indices = np.arange(n_total - val_size)
    
    # Split the data
    train_inputs_final = train_inputs[train_indices]
    train_targets_final = train_targets[train_indices]
    train_param_forecasts_final = train_param_forecasts[train_indices] if train_param_forecasts is not None else None
    
    val_inputs_array = train_inputs[val_indices]
    val_targets_array = train_targets[val_indices]
    val_param_forecasts_array = train_param_forecasts[val_indices] if train_param_forecasts is not None else None
    
    print(f"Training sequences: {len(train_inputs_final)}, Validation sequences: {len(val_inputs_array)}")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_inputs_final, train_targets_final, train_param_forecasts_final)
    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 0)
    )
    
    # Validation dataset is created from split above
    val_dataset = TimeSeriesDataset(val_inputs_array, val_targets_array, val_param_forecasts_array)
    
    val_loader = TorchDataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 0)
    )
    
    return train_loader, val_loader



def main():
    parser = argparse.ArgumentParser(description='Train DELPHI model')
    parser.add_argument('--config', type=str, default='configs/delphi_config.yaml',
                       help='Path to config file (relative to project root)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from Stage 1 checkpoint (skip Stage 1)')
    args = parser.parse_args()
    
    # Resolve config path relative to project root
    if not os.path.isabs(args.config):
        config_path = project_root / args.config
    else:
        config_path = Path(args.config)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate and convert config values (fixes YAML parsing issues like 1e-3 as string)
    print("Validating configuration...")
    try:
        config = _validate_and_convert_config(config)
        print("Configuration validated successfully.")
    except ValueError as e:
        print(f"Configuration error: {e}")
        raise SystemExit(1)
    
    # Set seed
    set_seed(int(_ensure_numeric(config.get('seed', 42), default=42, value_type='int')))
    
    # Create directories (relative to project root)
    save_dir = project_root / config['training']['save_dir']
    log_dir = project_root / config['training']['log_dir']
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Prepare data
    print("Preparing data...")
    splits, preprocessed = prepare_data(config)
    
    # Prepare datasets
    print("Preparing datasets...")
    train_loader, val_loader = prepare_datasets(splits, config)
    
    # Initialize model
    print("Initializing model...")
    # Check if CUDA is actually available, fallback to CPU if not
    requested_device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if requested_device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
    else:
        device = requested_device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = DELPHICore(
        input_dim=3,
        n_states=config['hmm']['n_states'],
        hmm_hidden_size=config['hmm']['hidden_size'],
        hmm_num_layers=config['hmm']['num_layers'],
        ensemble_hidden_size=config['ensemble']['hidden_size'],
        ensemble_num_layers=config['ensemble']['num_layers'],
        output_dim=config['data']['forecast_horizon'],
        n_ensemble_members=config['ensemble']['n_members'],
        dropout=config['hmm']['dropout']
    )
    
    # Initialize trainer
    training_cfg = config['training']
    trainer = DELPHITrainer(
        model=model,
        device=device,
        learning_rate=training_cfg['learning_rate'],
        weight_decay=training_cfg['weight_decay'],
        kl_weight=training_cfg['kl_weight'],
        entropy_weight=training_cfg['entropy_weight'],
        stage1_epochs=training_cfg['stage1_epochs'],
        stage2_epochs=training_cfg['stage2_epochs'],
        kl_anneal=training_cfg['kl_anneal'],
        kl_start=training_cfg['kl_start'],
        kl_end=training_cfg['kl_end'],
        kl_warmup_epochs=training_cfg['kl_warmup_epochs'],
        early_stopping_patience=training_cfg['early_stopping_patience'],
        early_stopping_min_delta=training_cfg['early_stopping_min_delta'],
        checkpoint_dir=str(save_dir),
        use_scheduler=training_cfg.get('use_scheduler', True),
        scheduler_patience=training_cfg.get('scheduler_patience', 5),
        scheduler_factor=training_cfg.get('scheduler_factor', 0.5),
        scheduler_min_lr=training_cfg.get('scheduler_min_lr', 1e-6),
        use_amp=training_cfg.get('use_amp', True)
    )
    
    # Stage 1 checkpoint paths
    best_stage1_path = save_dir / 'best_model_stage1.pt'
    stage1_checkpoint_path = save_dir / 'delphi_stage1.pt'
    
    # Check for resume mode - prefer best model if available
    if args.resume:
        if best_stage1_path.exists():
            print(f"Resuming from best Stage 1 checkpoint: {best_stage1_path}")
            trainer.load_checkpoint(str(best_stage1_path))
            # Restore best epoch info from checkpoint (already done in load_checkpoint)
            final_epoch_stage1 = trainer.best_epoch_stage1 if trainer.best_epoch_stage1 > 0 else config['training']['stage1_epochs']
            print("Best Stage 1 checkpoint loaded. Skipping Stage 1 training.")
        elif stage1_checkpoint_path.exists():
            print(f"Resuming from Stage 1 checkpoint: {stage1_checkpoint_path}")
            trainer.load_checkpoint(str(stage1_checkpoint_path))
            final_epoch_stage1 = trainer.best_epoch_stage1 if trainer.best_epoch_stage1 > 0 else config['training']['stage1_epochs']
            print("Stage 1 checkpoint loaded. Skipping Stage 1 training.")
        else:
            print("Warning: --resume specified but no checkpoint found. Starting fresh training.")
            args.resume = False  # Disable resume flag
            final_epoch_stage1 = None  # Will be set after Stage 1 training
    else:
        # Train Stage 1
        print("Training Stage 1 (Emissions/Posterior)...")
        trainer.train_stage1(train_loader, val_loader)
        
        # Save Stage 1 checkpoint (use best epoch if early stopping triggered)
        final_epoch_stage1 = trainer.best_epoch_stage1 if trainer.best_epoch_stage1 > 0 else config['training']['stage1_epochs']
        trainer.save_checkpoint(str(stage1_checkpoint_path), epoch=final_epoch_stage1)
        
        # If best model exists, use it for final checkpoint
        best_stage1_path = save_dir / 'best_model_stage1.pt'
        if best_stage1_path.exists():
            print(f"Stage 1 complete! Best model saved at epoch {trainer.best_epoch_stage1} (Val Loss: {trainer.best_val_loss_stage1:.4f})")
            print(f"Best checkpoint: {best_stage1_path}")
            print(f"Final checkpoint: {stage1_checkpoint_path}")
        else:
            print(f"Stage 1 complete! Checkpoint saved to {stage1_checkpoint_path}")
    
    # Train Stage 2
    print("Training Stage 2 (Prior)...")
    trainer.train_stage2(train_loader, val_loader)
    
    # Save DELPHI core model (use best epoch if early stopping triggered)
    final_epoch_stage2 = trainer.best_epoch_stage2 if trainer.best_epoch_stage2 > 0 else config['training']['stage2_epochs']
    final_epoch_total = final_epoch_stage1 + final_epoch_stage2
    
    checkpoint_path = save_dir / 'delphi_final.pt'
    trainer.save_checkpoint(str(checkpoint_path), epoch=final_epoch_total)
    
    # If best model exists for stage 2, recommend using it
    best_stage2_path = save_dir / 'best_model_stage2.pt'
    best_stage1_path = save_dir / 'best_model_stage1.pt'
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nFinal model saved to: {checkpoint_path}")
    
    if best_stage1_path.exists():
        print(f"\nBest Stage 1 model: {best_stage1_path} (Epoch {trainer.best_epoch_stage1}, Val Loss: {trainer.best_val_loss_stage1:.4f})")
    if best_stage2_path.exists():
        print(f"Best Stage 2 model: {best_stage2_path} (Epoch {trainer.best_epoch_stage2}, Val Loss: {trainer.best_val_loss_stage2:.4f})")
        print("\nNote: Consider using best_model_stage2.pt for evaluation if early stopping was triggered.")
    
    print("\nTo evaluate, run:")
    print(f"  python -m delphi.evaluation.evaluate --model_path {checkpoint_path}")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()

