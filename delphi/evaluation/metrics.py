"""
Evaluation metrics for DELPHI.

Uses MSE and MAE as the primary evaluation metrics.
"""

import numpy as np
from typing import Dict


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


def compute_all_metrics(
    y_true: Dict[str, np.ndarray],
    y_pred: Dict[str, np.ndarray]
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
    
    metrics = {
        'mse': np.mean(all_mse) if all_mse else np.nan,
        'mae': np.mean(all_mae) if all_mae else np.nan
    }
    
    return metrics
