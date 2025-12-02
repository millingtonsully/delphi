"""
Evaluation metrics for DELPHI.

Uses MSE and MAE as the primary evaluation metrics.
"""

import numpy as np
from typing import Dict, Optional


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error (MSE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MSE score
    """
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error (MAE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAE score
    """
    return np.mean(np.abs(y_true - y_pred))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    in_sample: Optional[np.ndarray],
    seasonal_period: int = 1
) -> float:
    """
    Mean Absolute Scaled Error (MASE) for seasonal data.
    
    Args:
        y_true: Ground-truth values for the forecast horizon
        y_pred: Predicted values for the forecast horizon
        in_sample: Historical in-sample values used to compute the
            seasonal naïve scaling term
        seasonal_period: Seasonal period (e.g., 52 for weekly annual seasonality)
    
    Returns:
        MASE score (np.nan if insufficient history or zero scaling term)
    """
    if in_sample is None or len(in_sample) <= seasonal_period:
        return np.nan
    
    diff = np.abs(in_sample[seasonal_period:] - in_sample[:-seasonal_period])
    if diff.size == 0:
        return np.nan
    
    scale = np.mean(diff)
    if scale == 0 or np.isnan(scale):
        return np.nan
    
    return np.mean(np.abs(y_true - y_pred)) / scale


def compute_all_metrics(
    y_true: Dict[str, np.ndarray],
    y_pred: Dict[str, np.ndarray],
    training_data: Optional[Dict[str, np.ndarray]] = None,
    seasonal_period: int = 1
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Computes MSE and MAE averaged across all series.
    
    Args:
        y_true: Dictionary of true values (keyed by series_id)
        y_pred: Dictionary of predicted values (keyed by series_id)
    
    Returns:
        Dictionary with 'mse' and 'mae' scores
    """
    all_mse = []
    all_mae = []
    all_mase = []
    
    for series_id in y_true.keys():
        if series_id not in y_pred:
            continue
        
        true_vals = y_true[series_id]
        pred_vals = y_pred[series_id]
        
        if len(true_vals) != len(pred_vals):
            continue
        
        # Compute metrics for this series
        all_mse.append(mse(true_vals, pred_vals))
        all_mae.append(mae(true_vals, pred_vals))
        
        if training_data is not None:
            train_series = training_data.get(series_id)
            mase_val = mase(true_vals, pred_vals, train_series, seasonal_period)
            if not np.isnan(mase_val):
                all_mase.append(mase_val)
    
    metrics = {
        'mse': np.mean(all_mse) if all_mse else np.nan,
        'mae': np.mean(all_mae) if all_mae else np.nan
    }
    
    if all_mase:
        metrics['mase'] = np.mean(all_mase)
    
    return metrics
