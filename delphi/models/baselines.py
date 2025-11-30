"""
Classic baseline models: ETS and SNAIVE for meta-ensemble.
"""

import numpy as np
from typing import Dict, Optional
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings('ignore')


class ETSModel:
    """
    Exponential Smoothing (ETS) model for meta-ensemble.
    """
    
    def __init__(
        self,
        seasonal: str = 'add',
        seasonal_periods: int = 52
    ):
        """
        Initialize ETS model.
        
        Args:
            seasonal: Seasonal component type ('add' or 'mul')
            seasonal_periods: Seasonal period
        """
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.fitted_models: Dict[str, any] = {}
    
    def fit(self, time_series: Dict[str, np.ndarray]) -> Dict[str, any]:
        """
        Fit ETS model to each time series.
        
        Args:
            time_series: Dictionary of series_id -> time series array
        
        Returns:
            Dictionary of fitted models
        """
        self.fitted_models = {}
        
        for series_id, ts in time_series.items():
            if len(ts) < self.seasonal_periods * 2:
                # Too short for seasonal model
                self.fitted_models[series_id] = None
                continue
            
            try:
                # Ensure non-negative
                ts_positive = np.maximum(ts, 0.1)
                
                model = ExponentialSmoothing(
                    ts_positive,
                    seasonal=self.seasonal,
                    seasonal_periods=self.seasonal_periods
                )
                fitted = model.fit()
                self.fitted_models[series_id] = fitted
            except Exception as e:
                print(f"Error fitting ETS for {series_id}: {e}")
                self.fitted_models[series_id] = None
        
        return self.fitted_models
    
    def forecast(
        self,
        time_series: Dict[str, np.ndarray],
        forecast_horizon: int = 26
    ) -> Dict[str, np.ndarray]:
        """
        Generate forecasts.
        
        Args:
            time_series: Dictionary of series_id -> time series array
            forecast_horizon: Number of steps ahead
        
        Returns:
            Dictionary of forecasts
        """
        forecasts = {}
        
        for series_id, ts in time_series.items():
            if series_id not in self.fitted_models:
                self.fit({series_id: ts})
            
            fitted = self.fitted_models.get(series_id)
            
            if fitted is None:
                # Fallback: use mean
                forecasts[series_id] = np.full(forecast_horizon, np.mean(ts))
                continue
            
            try:
                forecast = fitted.forecast(steps=forecast_horizon)
                forecasts[series_id] = np.array(forecast)
            except Exception as e:
                print(f"Error forecasting ETS for {series_id}: {e}")
                forecasts[series_id] = np.full(forecast_horizon, np.mean(ts))
        
        return forecasts


class SNAIVEModel:
    """
    Seasonal Naive (SNAIVE) model for meta-ensemble.
    """
    
    def __init__(self, seasonal_period: int = 52):
        """
        Initialize SNAIVE model.
        
        Args:
            seasonal_period: Seasonal period
        """
        self.seasonal_period = seasonal_period
    
    def forecast(
        self,
        time_series: Dict[str, np.ndarray],
        forecast_horizon: int = 26
    ) -> Dict[str, np.ndarray]:
        """
        Generate seasonal naive forecasts.
        
        Args:
            time_series: Dictionary of series_id -> time series array
            forecast_horizon: Number of steps ahead
        
        Returns:
            Dictionary of forecasts
        """
        forecasts = {}
        
        for series_id, ts in time_series.items():
            if len(ts) < self.seasonal_period:
                # Too short, use last value
                forecasts[series_id] = np.full(forecast_horizon, ts[-1] if len(ts) > 0 else 0.0)
                continue
            
            # Seasonal naive: repeat last seasonal_period values
            last_season = ts[-self.seasonal_period:]
            # Tile to cover forecast horizon
            n_repeats = (forecast_horizon // self.seasonal_period) + 1
            forecast = np.tile(last_season, n_repeats)[:forecast_horizon]
            forecasts[series_id] = forecast
        
        return forecasts


