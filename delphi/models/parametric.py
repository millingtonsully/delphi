"""
Parametric baseline models for DELPHI: TBATS implementation.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
import warnings
import time
import os
import pickle
import multiprocessing
from tbats import TBATS

warnings.filterwarnings('ignore')


def _fit_single_series(args):
    """
    Helper function to fit TBATS model for a single series.
    This function is designed to be picklable for multiprocessing.
    
    Args:
        args: Tuple of (series_id, ts, use_box_cox, use_trend, use_arma_errors, 
                        seasonal_periods, n_jobs, min_seasonal_length)
    
    Returns:
        Tuple of (series_id, result_dict) where result_dict contains:
            - 'model': fitted TBATS model or None if failed
            - 'offset': offset applied to make values positive (0.0 if none needed)
            - 'success': bool indicating success
            - 'error': error message if failed
    """
    series_id, ts, use_box_cox, use_trend, use_arma_errors, seasonal_periods, n_jobs, min_seasonal_length = args
    
    # Check if series is too short
    if len(ts) < min_seasonal_length:
        return (series_id, {
            'model': None,
            'offset': 0.0,
            'success': False,
            'error': f'Series too short (length {len(ts)} < {min_seasonal_length})'
        })
    
    try:
        # Handle Box-Cox transformation: use consistent offset-based approach
        # Add offset to make all values strictly positive (statistically sound approach)
        offset = 0.0
        ts_positive = ts.copy()
        
        if use_box_cox:
            min_val = np.min(ts)
            if min_val <= 0:
                # Add offset to make all values positive
                offset = abs(min_val) + 0.001
                ts_positive = ts + offset
            else:
                # Ensure minimum is not too close to zero for numerical stability
                if min_val < 0.01:
                    offset = 0.01 - min_val
                    ts_positive = ts + offset
                else:
                    # Values are already positive and sufficiently large
                    ts_positive = np.maximum(ts, 1e-6)
        else:
            # Even without Box-Cox, ensure positive values for numerical stability
            ts_positive = np.maximum(ts, 1e-6)
        
        # Fit TBATS model
        estimator = TBATS(
            use_box_cox=use_box_cox,
            use_trend=use_trend,
            use_arma_errors=use_arma_errors,
            seasonal_periods=seasonal_periods,
            n_jobs=n_jobs
        )
        
        fitted_model = estimator.fit(ts_positive)
        
        return (series_id, {
            'model': fitted_model,
            'offset': offset,
            'success': True,
            'error': None
        })
        
    except Exception as e:
        return (series_id, {
            'model': None,
            'offset': 0.0,
            'success': False,
            'error': str(e)
        })


class TBATSBaseline:
    """
    TBATS (Exponential smoothing state space model with Box-Cox transformation,
    ARMA errors, Trend, and Seasonal components) baseline for initial predictions.
    """
    
    def __init__(
        self,
        use_box_cox: bool = True,
        use_trend: bool = True,
        use_arma_errors: bool = True,
        seasonal_periods: Optional[List[int]] = None,
        n_jobs: int = 1,
        n_parallel_workers: int = 8
    ):
        """
        Initialize TBATS baseline model.
        
        Args:
            use_box_cox: Whether to use Box-Cox transformation
            use_trend: Whether to include trend component
            use_arma_errors: Whether to use ARMA errors
            seasonal_periods: List of seasonal periods (e.g., [52] for annual)
            n_jobs: Number of parallel jobs for TBATS internal fitting (per series)
            n_parallel_workers: Number of parallel workers for fitting multiple series (default: 8)
        """
        self.use_box_cox = use_box_cox
        self.use_trend = use_trend
        self.use_arma_errors = use_arma_errors
        self.seasonal_periods = seasonal_periods or [52]
        self.n_jobs = n_jobs
        # Limit to available CPU cores
        available_cores = os.cpu_count() or 1
        self.n_parallel_workers = min(n_parallel_workers, available_cores)
        
        self.fitted_models: Dict[str, any] = {}
        self.forecasts: Dict[str, np.ndarray] = {}
        self.residuals: Dict[str, np.ndarray] = {}
        # Track offsets for each series (to reverse Box-Cox offset)
        self.offset_factors: Dict[str, float] = {}
        # Diagnostics: track fitting success
        self.fitting_stats = {
            'successful_fits': 0,
            'failed_fits': 0,
            'fallback_to_mean': 0,
            'offset_applied': 0
        }
    
    def fit(
        self,
        time_series: Dict[str, np.ndarray],
        forecast_horizon: int = 26,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Fit TBATS model to each time series using multiprocessing.
        
        Args:
            time_series: Dictionary of series_id -> time series array
            forecast_horizon: Number of steps ahead to forecast
            verbose: Whether to print progress indicators
            
        Returns:
            Dictionary of fitted models
        """
        self.fitted_models = {}
        self.offset_factors = {}
        self.fitting_stats = {
            'successful_fits': 0,
            'failed_fits': 0,
            'fallback_to_mean': 0,
            'offset_applied': 0
        }
        
        total_series = len(time_series)
        if total_series == 0:
            return self.fitted_models
        
        start_time = time.time()
        min_seasonal_length = max(self.seasonal_periods) * 2
        
        # Prepare arguments for multiprocessing
        # When using multiprocessing, disable internal TBATS parallelism (n_jobs=1)
        # to avoid "daemonic processes" error on Windows
        # Internal parallelism is only used when running sequentially
        use_multiprocessing = self.n_parallel_workers > 1 and total_series > 1
        effective_n_jobs = 1 if use_multiprocessing else self.n_jobs
        
        fit_args = [
            (series_id, ts, self.use_box_cox, self.use_trend, self.use_arma_errors,
             self.seasonal_periods, effective_n_jobs, min_seasonal_length)
            for series_id, ts in time_series.items()
        ]
        
        # Use multiprocessing if we have multiple workers and series
        if use_multiprocessing:
            if verbose:
                print(f"Fitting TBATS models for {total_series} series using {self.n_parallel_workers} parallel workers "
                      f"(internal threading disabled to avoid nested multiprocessing issues)...")
            
            # Process with multiprocessing using imap_unordered for better progress tracking
            with multiprocessing.Pool(processes=self.n_parallel_workers) as pool:
                completed = 0
                results_dict = {}
                
                # Use imap_unordered to get results as they complete
                for result in pool.imap_unordered(_fit_single_series, fit_args):
                    series_id, result_dict = result
                    results_dict[series_id] = result_dict
                    completed += 1
                    
                    # Progress indicator every 100 series or at completion
                    if verbose and (completed % 100 == 0 or completed == total_series):
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        remaining = (total_series - completed) / rate if rate > 0 else 0
                        pct = (completed / total_series) * 100
                        print(f"Fitting TBATS models... {completed}/{total_series} ({pct:.1f}%) "
                              f"| Elapsed: {elapsed:.1f}s | Rate: {rate:.2f} series/s | "
                              f"Est. remaining: {remaining:.1f}s")
                
                # Process all results
                for series_id, result_dict in results_dict.items():
                    if result_dict['success']:
                        self.fitted_models[series_id] = result_dict['model']
                        self.offset_factors[series_id] = result_dict['offset']
                        self.fitting_stats['successful_fits'] += 1
                        if result_dict['offset'] != 0.0:
                            self.fitting_stats['offset_applied'] += 1
                    else:
                        if verbose and result_dict['error'] and 'too short' not in result_dict['error'].lower():
                            print(f"Warning: Series {series_id} failed: {result_dict['error']}")
                        self.fitted_models[series_id] = None
                        self.offset_factors[series_id] = 0.0
                        self.fitting_stats['failed_fits'] += 1
        else:
            # Sequential processing (fallback or single worker)
            if verbose:
                print(f"Fitting TBATS models for {total_series} series sequentially...")
            
            # Sequential mode: can use internal parallelism (n_jobs from config)
            for idx, (series_id, ts) in enumerate(time_series.items(), 1):
                args = (series_id, ts, self.use_box_cox, self.use_trend, self.use_arma_errors,
                        self.seasonal_periods, self.n_jobs, min_seasonal_length)
                series_id_result, result_dict = _fit_single_series(args)
                
                if result_dict['success']:
                    self.fitted_models[series_id_result] = result_dict['model']
                    self.offset_factors[series_id_result] = result_dict['offset']
                    self.fitting_stats['successful_fits'] += 1
                    if result_dict['offset'] != 0.0:
                        self.fitting_stats['offset_applied'] += 1
                else:
                    if verbose and result_dict['error']:
                        print(f"Warning: Series {series_id_result} failed: {result_dict['error']}")
                    self.fitted_models[series_id_result] = None
                    self.offset_factors[series_id_result] = 0.0
                    self.fitting_stats['failed_fits'] += 1
                
                # Progress indicator every 100 series
                if verbose and idx % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = idx / elapsed if elapsed > 0 else 0
                    remaining = (total_series - idx) / rate if rate > 0 else 0
                    pct = (idx / total_series) * 100
                    print(f"Fitting TBATS models... {idx}/{total_series} ({pct:.1f}%) "
                          f"| Elapsed: {elapsed:.1f}s | Rate: {rate:.2f} series/s | "
                          f"Est. remaining: {remaining:.1f}s")
        
        if verbose and total_series > 0:
            elapsed = time.time() - start_time
            successful = self.fitting_stats['successful_fits']
            print(f"\nTBATS fitting complete. Fitted {successful}/{total_series} models "
                  f"in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
            if self.fitting_stats['offset_applied'] > 0:
                print(f"  Applied offset to {self.fitting_stats['offset_applied']} series for Box-Cox")
            if self.fitting_stats['failed_fits'] > 0:
                print(f"  Failed fits: {self.fitting_stats['failed_fits']}")
        
        return self.fitted_models
    
    def forecast(
        self,
        time_series: Dict[str, np.ndarray],
        forecast_horizon: int = 26,
        return_residuals: bool = True,
        verbose: bool = True
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Generate forecasts and residuals for each time series.
        
        Args:
            time_series: Dictionary of series_id -> time series array
            forecast_horizon: Number of steps ahead to forecast
            return_residuals: Whether to compute residuals
            verbose: Whether to print progress indicators
            
        Returns:
            Tuple of (forecasts_dict, residuals_dict)
        """
        forecasts = {}
        residuals = {}
        total_series = len(time_series)
        start_time = time.time()
        
        for idx, (series_id, ts) in enumerate(time_series.items(), 1):
            if series_id not in self.fitted_models:
                # Fit if not already fitted
                self.fit({series_id: ts}, forecast_horizon, verbose=False)
            
            fitted_model = self.fitted_models.get(series_id)
            
            if fitted_model is None:
                # Fallback: use mean forecast
                mean_val = np.mean(ts)
                forecasts[series_id] = np.full(forecast_horizon, mean_val)
                if return_residuals:
                    residuals[series_id] = ts - mean_val
                # Ensure offset factor exists for consistency
                if series_id not in self.offset_factors:
                    self.offset_factors[series_id] = 0.0
                self.fitting_stats['fallback_to_mean'] += 1
                continue
            
            try:
                # Generate forecast
                forecast_result = fitted_model.forecast(steps=forecast_horizon)
                forecast_array = np.array(forecast_result)
                
                # Reverse offset to return forecast to original scale
                offset = self.offset_factors.get(series_id, 0.0)
                if offset != 0.0:
                    forecast_array = forecast_array - offset
                
                forecasts[series_id] = forecast_array
                
                # Compute in-sample fitted values and residuals
                if return_residuals:
                    # Try to get fitted values - TBATS model may have different attributes
                    fitted_values = None
                    if hasattr(fitted_model, 'fittedvalues'):
                        fitted_values = fitted_model.fittedvalues
                    elif hasattr(fitted_model, 'y_hat'):
                        fitted_values = fitted_model.y_hat
                    elif hasattr(fitted_model, 'fitted'):
                        fitted_values = fitted_model.fitted
                    
                    if fitted_values is not None and len(fitted_values) == len(ts):
                        # Reverse offset from fitted values to return to original scale
                        if offset != 0.0:
                            fitted_values = fitted_values - offset
                        residuals[series_id] = ts - fitted_values
                    else:
                        # Fallback: compute residuals using mean as approximation
                        residuals[series_id] = ts - np.mean(ts)
                
            except Exception as e:
                if verbose:
                    print(f"Error forecasting for {series_id}: {e}")
                # Fallback: use mean
                mean_val = np.mean(ts)
                forecasts[series_id] = np.full(forecast_horizon, mean_val)
                if return_residuals:
                    residuals[series_id] = ts - mean_val
            
            # Progress indicator every 1000 series
            if verbose and idx % 1000 == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed if elapsed > 0 else 0
                remaining = (total_series - idx) / rate if rate > 0 else 0
                print(f"Generating TBATS forecasts... {idx}/{total_series} series processed "
                      f"(elapsed: {elapsed:.1f}s, est. remaining: {remaining:.1f}s)")
        
        if verbose and total_series > 0:
            elapsed = time.time() - start_time
            print(f"TBATS forecasting complete. Generated forecasts for {len(forecasts)}/{total_series} series "
                  f"in {elapsed:.1f}s")
        
        self.forecasts = forecasts
        self.residuals = residuals
        
        return forecasts, residuals
    
    def fit_and_forecast(
        self,
        time_series: Dict[str, np.ndarray],
        forecast_horizon: int = 26,
        verbose: bool = True
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Fit model and generate forecasts in one step.
        
        Args:
            time_series: Dictionary of series_id -> time series array
            forecast_horizon: Number of steps ahead to forecast
            verbose: Whether to print progress indicators
            
        Returns:
            Tuple of (forecasts_dict, residuals_dict)
        """
        self.fit(time_series, forecast_horizon, verbose=verbose)
        return self.forecast(time_series, forecast_horizon, return_residuals=True, verbose=verbose)
    
    def get_forecast_for_series(
        self,
        series_id: str,
        forecast_horizon: int = 26
    ) -> Optional[np.ndarray]:
        """Get forecast for a specific series."""
        if series_id in self.forecasts:
            return self.forecasts[series_id]
        return None
    
    def get_residuals_for_series(self, series_id: str) -> Optional[np.ndarray]:
        """Get residuals for a specific series."""
        if series_id in self.residuals:
            return self.residuals[series_id]
        return None
    
    def get_fitting_stats(self) -> Dict[str, int]:
        """Get statistics about TBATS fitting."""
        return self.fitting_stats.copy()
    
    def save_checkpoint(self, filepath: str) -> None:
        """
        Save TBATS checkpoint to disk for resuming training.
        
        Args:
            filepath: Path to save the checkpoint file
        """
        checkpoint = {
            'fitted_models': self.fitted_models,
            'forecasts': self.forecasts,
            'residuals': self.residuals,
            'offset_factors': self.offset_factors,
            'fitting_stats': self.fitting_stats,
            # Save config for validation on load
            'config': {
                'use_box_cox': self.use_box_cox,
                'use_trend': self.use_trend,
                'use_arma_errors': self.use_arma_errors,
                'seasonal_periods': self.seasonal_periods,
                'n_jobs': self.n_jobs,
                'n_parallel_workers': self.n_parallel_workers
            }
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"TBATS checkpoint saved to {filepath}")
        print(f"  Models: {len(self.fitted_models)}, Forecasts: {len(self.forecasts)}")
    
    @classmethod
    def load_checkpoint(cls, filepath: str) -> 'TBATSBaseline':
        """
        Load TBATS checkpoint from disk.
        
        Args:
            filepath: Path to the checkpoint file
            
        Returns:
            TBATSBaseline instance with restored state
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"TBATS checkpoint not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Create instance with saved config
        config = checkpoint.get('config', {})
        instance = cls(
            use_box_cox=config.get('use_box_cox', True),
            use_trend=config.get('use_trend', True),
            use_arma_errors=config.get('use_arma_errors', True),
            seasonal_periods=config.get('seasonal_periods', [52]),
            n_jobs=config.get('n_jobs', 1),
            n_parallel_workers=config.get('n_parallel_workers', 8)
        )
        
        # Restore state
        instance.fitted_models = checkpoint.get('fitted_models', {})
        instance.forecasts = checkpoint.get('forecasts', {})
        instance.residuals = checkpoint.get('residuals', {})
        instance.offset_factors = checkpoint.get('offset_factors', {})
        instance.fitting_stats = checkpoint.get('fitting_stats', {
            'successful_fits': 0,
            'failed_fits': 0,
            'fallback_to_mean': 0,
            'offset_applied': 0
        })
        
        print(f"TBATS checkpoint loaded from {filepath}")
        print(f"  Models: {len(instance.fitted_models)}, Forecasts: {len(instance.forecasts)}")
        
        return instance


def fit_parametric_baseline(
    time_series: Dict[str, np.ndarray],
    model_type: str = "tbats",
    forecast_horizon: int = 26,
    **kwargs
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Convenience function to fit parametric baseline and get forecasts/residuals.
    
    Args:
        time_series: Dictionary of series_id -> time series array
        model_type: Type of parametric model ("tbats")
        forecast_horizon: Number of steps ahead to forecast
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Tuple of (forecasts_dict, residuals_dict)
    """
    if model_type.lower() == "tbats":
        model = TBATSBaseline(**kwargs)
        return model.fit_and_forecast(time_series, forecast_horizon)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

