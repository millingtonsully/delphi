"""
Evaluation metrics for DELPHI: MASE, OWA, PDA, Calibration.
"""

import numpy as np
from typing import Dict, Optional, Tuple


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    seasonal_period: int = 52
) -> float:
    """
    Mean Absolute Scaled Error (MASE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_train: Training data for scaling
        seasonal_period: Seasonal period
    
    Returns:
        MASE score
    """
    mae_val = np.mean(np.abs(y_true - y_pred))
    
    if y_train is not None:
        # Use seasonal naive error as scale
        if len(y_train) >= seasonal_period:
            seasonal_naive_errors = np.abs(
                y_train[seasonal_period:] - y_train[:-seasonal_period]
            )
            scale = np.mean(seasonal_naive_errors)
        else:
            scale = np.mean(np.abs(np.diff(y_train)))
    else:
        # Fallback: use mean of true values
        scale = np.mean(np.abs(y_true))
    
    if scale < 1e-8:
        return np.inf
    
    return mae_val / scale


def owa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_naive: Optional[np.ndarray] = None
) -> float:
    """
    Overall Weighted Average (OWA).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_naive: Naive forecast for normalization
    
    Returns:
        OWA score
    """
    mae_val = np.mean(np.abs(y_true - y_pred))
    
    if y_naive is not None:
        mae_naive = np.mean(np.abs(y_true - y_naive))
        if mae_naive < 1e-8:
            return np.inf
        return mae_val / mae_naive
    else:
        # Use mean as naive baseline
        mae_naive = np.mean(np.abs(y_true - np.mean(y_true)))
        if mae_naive < 1e-8:
            return np.inf
        return mae_val / mae_naive


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def pda(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_history: Optional[np.ndarray] = None
) -> float:
    """
    Prediction Direction Accuracy (PDA).
    
    Measures how often the predicted direction of change matches the true direction.
    Direction is computed relative to the last known value (from history) or the
    first true value if history is not provided.
    
    Args:
        y_true: True values (forecast horizon)
        y_pred: Predicted values (forecast horizon)
        y_history: Historical data (training/context), last value used as reference
    
    Returns:
        PDA score between 0 and 1 (higher is better)
    """
    # Get the reference point (last known value before forecast)
    if y_history is not None and len(y_history) > 0:
        last_known = y_history[-1]
    else:
        # Fallback: use first true value as reference
        last_known = y_true[0]
    
    # Compute direction of change from the last known value
    true_direction = np.sign(y_true - last_known)
    pred_direction = np.sign(y_pred - last_known)
    
    # Calculate accuracy (proportion of correct direction predictions)
    correct_directions = (true_direction == pred_direction)
    
    return np.mean(correct_directions)


def calibration_score(
    y_true: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Compute calibration score for prediction intervals.
    
    Args:
        y_true: True values
        lower_bound: Lower bound of prediction interval
        upper_bound: Upper bound of prediction interval
        confidence_level: Expected coverage (default 0.95 for 95% CI)
    
    Returns:
        Tuple of (coverage, calibration_error):
            - coverage: Actual proportion of true values within bounds
            - calibration_error: |coverage - expected_coverage|
    """
    in_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
    coverage = np.mean(in_interval)
    calibration_error = abs(coverage - confidence_level)
    
    return coverage, calibration_error


def compute_all_metrics(
    y_true: Dict[str, np.ndarray],
    y_pred: Dict[str, np.ndarray],
    y_train: Optional[Dict[str, np.ndarray]] = None,
    y_lower: Optional[Dict[str, np.ndarray]] = None,
    y_upper: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: Dictionary of true values
        y_pred: Dictionary of predicted values
        y_train: Dictionary of training data (for MASE)
        y_lower: Dictionary of lower 95% confidence bounds (for calibration)
        y_upper: Dictionary of upper 95% confidence bounds (for calibration)
    
    Returns:
        Dictionary of metric scores
    """
    all_mase = []
    all_owa = []
    all_mae = []
    all_rmse = []
    all_pda = []
    all_coverage = []
    
    for series_id in y_true.keys():
        if series_id not in y_pred:
            continue
        
        true_vals = y_true[series_id]
        pred_vals = y_pred[series_id]
        
        if len(true_vals) != len(pred_vals):
            continue
        
        # Basic metrics
        all_mae.append(mae(true_vals, pred_vals))
        all_rmse.append(rmse(true_vals, pred_vals))
        
        # MASE
        train_vals = y_train.get(series_id) if y_train else None
        all_mase.append(mase(true_vals, pred_vals, train_vals))
        
        # OWA
        all_owa.append(owa(true_vals, pred_vals))
        
        # PDA (Prediction Direction Accuracy)
        all_pda.append(pda(true_vals, pred_vals, train_vals))
        
        # Calibration (if uncertainty bounds provided)
        if y_lower is not None and y_upper is not None:
            if series_id in y_lower and series_id in y_upper:
                coverage, _ = calibration_score(
                    true_vals, y_lower[series_id], y_upper[series_id]
                )
                all_coverage.append(coverage)
    
    metrics = {
        'mase': np.mean(all_mase) if all_mase else np.nan,
        'owa': np.mean(all_owa) if all_owa else np.nan,
        'mae': np.mean(all_mae) if all_mae else np.nan,
        'rmse': np.mean(all_rmse) if all_rmse else np.nan,
        'pda': np.mean(all_pda) if all_pda else np.nan
    }
    
    # Add calibration if uncertainty was provided
    if all_coverage:
        metrics['coverage_95'] = np.mean(all_coverage)
    
    return metrics
