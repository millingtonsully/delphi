"""
Result Loader: Utilities to read evaluation results and identify series to explain.

Reads evaluation_results/predictions.csv to extract series IDs and provides
utilities to map results back to model inputs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Set
import warnings

warnings.filterwarnings('ignore')


def load_evaluation_results(results_dir: str = "evaluation_results") -> pd.DataFrame:
    """
    Load predictions from evaluation results.
    
    Args:
        results_dir: Directory containing evaluation results
    
    Returns:
        DataFrame with predictions
    """
    results_path = Path(results_dir) / "predictions.csv"
    
    if not results_path.exists():
        raise FileNotFoundError(f"Evaluation results not found: {results_path}")
    
    df = pd.read_csv(results_path)
    return df


def get_series_ids(results_dir: str = "evaluation_results") -> List[str]:
    """
    Extract unique series IDs from evaluation results.
    
    Args:
        results_dir: Directory containing evaluation results
    
    Returns:
        List of unique series IDs
    """
    df = load_evaluation_results(results_dir)
    
    if 'series_id' in df.columns:
        series_ids = df['series_id'].unique().tolist()
    else:
        raise ValueError("'series_id' column not found in predictions.csv")
    
    return series_ids


def get_series_horizons(results_dir: str = "evaluation_results") -> Dict[str, int]:
    """
    Get forecast horizon for each series.
    
    Args:
        results_dir: Directory containing evaluation results
    
    Returns:
        Dictionary mapping series_id to forecast horizon
    """
    df = load_evaluation_results(results_dir)
    
    if 'series_id' not in df.columns or 'horizon' not in df.columns:
        raise ValueError("Required columns 'series_id' and 'horizon' not found")
    
    horizons = df.groupby('series_id')['horizon'].max().to_dict()
    return horizons


def filter_series_by_criteria(
    results_dir: str = "evaluation_results",
    min_horizon: Optional[int] = None,
    max_series: Optional[int] = None
) -> List[str]:
    """
    Filter series based on criteria.
    
    Args:
        results_dir: Directory containing evaluation results
        min_horizon: Minimum forecast horizon required
        max_series: Maximum number of series to return (None for all)
    
    Returns:
        Filtered list of series IDs
    """
    series_ids = get_series_ids(results_dir)
    
    if min_horizon is not None:
        horizons = get_series_horizons(results_dir)
        series_ids = [sid for sid in series_ids if horizons.get(sid, 0) >= min_horizon]
    
    if max_series is not None:
        series_ids = series_ids[:max_series]
    
    return series_ids


def get_evaluation_metadata(results_dir: str = "evaluation_results") -> Dict:
    """
    Get metadata from evaluation results (forecast horizon, context length, etc.).
    
    Args:
        results_dir: Directory containing evaluation results
    
    Returns:
        Dictionary with metadata
    """
    metrics_path = Path(results_dir) / "evaluation_metrics.txt"
    
    metadata = {}
    
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Forecast horizon' in line:
                    try:
                        metadata['forecast_horizon'] = int(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Context length' in line:
                    try:
                        metadata['context_length'] = int(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Series evaluated' in line:
                    try:
                        metadata['num_series'] = int(line.split(':')[1].strip())
                    except:
                        pass
    
    # Also get from predictions if available
    df = load_evaluation_results(results_dir)
    if 'horizon' in df.columns:
        metadata['max_horizon'] = int(df['horizon'].max())
        metadata['min_horizon'] = int(df['horizon'].min())
    
    return metadata

