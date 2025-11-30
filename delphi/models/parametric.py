"""
Parametric baseline models for DELPHI: TBATS implementation.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
import warnings
import time
from tbats import TBATS

warnings.filterwarnings('ignore')


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
        n_jobs: int = 1
    ):
        """
        Initialize TBATS baseline model.
        
        Args:
            use_box_cox: Whether to use Box-Cox transformation
            use_trend: Whether to include trend component
            use_arma_errors: Whether to use ARMA errors
            seasonal_periods: List of seasonal periods (e.g., [52] for annual)
            n_jobs: Number of parallel jobs for fitting
        """
        self.use_box_cox = use_box_cox
        self.use_trend = use_trend
        self.use_arma_errors = use_arma_errors
        self.seasonal_periods = seasonal_periods or [52]
        self.n_jobs = n_jobs
        
        self.fitted_models: Dict[str, any] = {}
        self.forecasts: Dict[str, np.ndarray] = {}
        self.residuals: Dict[str, np.ndarray] = {}
    
    def fit(
        self,
        time_series: Dict[str, np.ndarray],
        forecast_horizon: int = 26,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Fit TBATS model to each time series.
        
        Args:
            time_series: Dictionary of series_id -> time series array
            forecast_horizon: Number of steps ahead to forecast
            verbose: Whether to print progress indicators
            
        Returns:
            Dictionary of fitted models
        """
        self.fitted_models = {}
        total_series = len(time_series)
        start_time = time.time()
        
        for idx, (series_id, ts) in enumerate(time_series.items(), 1):
            if len(ts) < max(self.seasonal_periods) * 2:
                # Skip series that are too short
                if verbose:
                    print(f"Warning: Series {series_id} too short for TBATS, skipping")
                continue
            
            try:
                # Ensure non-negative values for Box-Cox
                ts_positive = np.maximum(ts, 0.1)
                
                # Fit TBATS model
                estimator = TBATS(
                    use_box_cox=self.use_box_cox,
                    use_trend=self.use_trend,
                    use_arma_errors=self.use_arma_errors,
                    seasonal_periods=self.seasonal_periods,
                    n_jobs=self.n_jobs
                )
                
                fitted_model = estimator.fit(ts_positive)
                self.fitted_models[series_id] = fitted_model
                
            except Exception as e:
                if verbose:
                    print(f"Error fitting TBATS for {series_id}: {e}")
                # Fallback: use simple mean
                self.fitted_models[series_id] = None
            
            # Progress indicator every 1000 series
            if verbose and idx % 1000 == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed if elapsed > 0 else 0
                remaining = (total_series - idx) / rate if rate > 0 else 0
                print(f"Fitting TBATS models... {idx}/{total_series} series processed "
                      f"(elapsed: {elapsed:.1f}s, est. remaining: {remaining:.1f}s)")
        
        if verbose and total_series > 0:
            elapsed = time.time() - start_time
            print(f"TBATS fitting complete. Fitted {len(self.fitted_models)}/{total_series} models "
                  f"in {elapsed:.1f}s")
        
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
                continue
            
            try:
                # Generate forecast
                forecast_result = fitted_model.forecast(steps=forecast_horizon)
                forecasts[series_id] = np.array(forecast_result)
                
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

