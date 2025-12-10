"""
Evaluation metrics for DELPHI.

Uses MSE and MAE as the primary evaluation metrics.
"""

import numpy as np
from typing import Dict, Optional, Union


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
    seasonal_period: int = 1,
    forecast_horizon: Optional[int] = None,
    return_diagnostics: bool = False
) -> Union[float, Dict[str, float]]:
    """
    Mean Absolute Scaled Error (MASE) for seasonal data.
    
    Formula:
    MASE = (T - m) / h * sum(|Y_T+j - Ŷ_T+j|) / sum(|Y_i - Y_{i-m}|)
    
    Where:
    - T is training length, m is seasonal period, h is forecast horizon
    - Numerator: sum of absolute forecast errors
    - Denominator: sum of absolute seasonal naive errors on training data
    
    Args:
        y_true: Ground-truth values for the forecast horizon
        y_pred: Predicted values for the forecast horizon
        in_sample: Historical in-sample values used to compute the
            seasonal naïve scaling term
        seasonal_period: Seasonal period (e.g., 52 for weekly annual seasonality)
        forecast_horizon: Forecast horizon h (if None, uses len(y_true))
        return_diagnostics: If True, return dict with MASE components for debugging
    
    Returns:
        MASE score (np.nan if insufficient history or zero scaling term), or
        dict with diagnostics if return_diagnostics=True
    """
    if in_sample is None or len(in_sample) <= seasonal_period:
        if return_diagnostics:
            return {'mase': np.nan, 'error': 'insufficient_history'}
        return np.nan
    
    T = len(in_sample)
    h = forecast_horizon if forecast_horizon is not None else len(y_true)
    m = seasonal_period
    
    # Compute seasonal naive errors on training data (denominator)
    diff = np.abs(in_sample[m:] - in_sample[:-m])
    if diff.size == 0:
        if return_diagnostics:
            return {'mase': np.nan, 'error': 'empty_diff'}
        return np.nan
    
    sum_seasonal_naive_errors = np.sum(diff)
    mean_seasonal_naive_error = np.mean(diff)
    if sum_seasonal_naive_errors == 0 or np.isnan(sum_seasonal_naive_errors):
        if return_diagnostics:
            return {'mase': np.nan, 'error': 'zero_denominator', 'sum_seasonal_naive_errors': sum_seasonal_naive_errors}
        return np.nan
    
    # Compute forecast errors (numerator)
    forecast_errors = np.abs(y_true - y_pred)
    sum_forecast_errors = np.sum(forecast_errors)
    mean_forecast_error = np.mean(forecast_errors)
    
    # MASE formula: (T - m) / h * sum(|errors|) / sum(|seasonal_naive_errors|)
    if h == 0:
        if return_diagnostics:
            return {'mase': np.nan, 'error': 'zero_horizon'}
        return np.nan
    
    mase_val = ((T - m) / h) * (sum_forecast_errors / sum_seasonal_naive_errors)
    
    if return_diagnostics:
        return {
            'mase': mase_val,
            'T': T,
            'm': m,
            'h': h,
            'sum_forecast_errors': sum_forecast_errors,
            'mean_forecast_error': mean_forecast_error,
            'sum_seasonal_naive_errors': sum_seasonal_naive_errors,
            'mean_seasonal_naive_error': mean_seasonal_naive_error,
            'scaling_factor': (T - m) / h,
            'error_ratio': sum_forecast_errors / sum_seasonal_naive_errors
        }
    
    return mase_val


def compute_all_metrics(
    y_true: Dict[str, np.ndarray],
    y_pred: Dict[str, np.ndarray],
    training_data: Optional[Dict[str, np.ndarray]] = None,
    seasonal_period: int = 1,
    forecast_horizon: Optional[int] = None,
    return_stats: bool = False
) -> Union[Dict[str, float], tuple]:
    """
    Compute all evaluation metrics.
    
    Computes MSE and MAE averaged across all series.
    
    Args:
        y_true: Dictionary of true values (keyed by series_id)
        y_pred: Dictionary of predicted values (keyed by series_id)
        training_data: Training data for MASE computation (original scale)
        seasonal_period: Seasonal period for MASE
        forecast_horizon: Forecast horizon for MASE
        return_stats: If True, also return statistics about metric computation
    
    Returns:
        Dictionary with 'mse' and 'mae' scores (and 'mase' if training_data provided),
        or tuple of (metrics_dict, stats_dict) if return_stats=True
    """
    all_mse = []
    all_mae = []
    all_mase = []
    
    # Statistics tracking
    stats = {
        'total_series': len(y_true),
        'series_with_predictions': 0,
        'series_with_valid_mse_mae': 0,
        'series_with_valid_mase': 0,
        'series_with_failed_mase': 0,
        'mase_failure_reasons': {}
    }
    
    for series_id in y_true.keys():
        if series_id not in y_pred:
            continue
        
        stats['series_with_predictions'] += 1
        true_vals = y_true[series_id]
        pred_vals = y_pred[series_id]
        
        if len(true_vals) != len(pred_vals):
            continue
        
        # Compute metrics for this series
        all_mse.append(mse(true_vals, pred_vals))
        all_mae.append(mae(true_vals, pred_vals))
        stats['series_with_valid_mse_mae'] += 1
        
        if training_data is not None:
            train_series = training_data.get(series_id)
            mase_val = mase(true_vals, pred_vals, train_series, seasonal_period, forecast_horizon, return_diagnostics=True)
            
            if isinstance(mase_val, dict):
                if 'error' in mase_val:
                    # MASE computation failed
                    stats['series_with_failed_mase'] += 1
                    error_type = mase_val.get('error', 'unknown')
                    stats['mase_failure_reasons'][error_type] = stats['mase_failure_reasons'].get(error_type, 0) + 1
                else:
                    # MASE computation succeeded
                    all_mase.append(mase_val['mase'])
                    stats['series_with_valid_mase'] += 1
            elif not np.isnan(mase_val):
                all_mase.append(mase_val)
                stats['series_with_valid_mase'] += 1
            else:
                stats['series_with_failed_mase'] += 1
                stats['mase_failure_reasons']['nan'] = stats['mase_failure_reasons'].get('nan', 0) + 1
    
    metrics = {
        'mse': np.mean(all_mse) if all_mse else np.nan,
        'mae': np.mean(all_mae) if all_mae else np.nan
    }
    
    if all_mase:
        metrics['mase'] = np.mean(all_mase)
    
    if return_stats:
        return metrics, stats
    
    return metrics
