"""
Meta-Ensemble Integration: Combines DELPHI core with external models.
Uses XGBoost meta-learner for weighting.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import PatchTST, NBEATS, NHITS
    NEURALFORECAST_AVAILABLE = True
except ImportError:
    NEURALFORECAST_AVAILABLE = False
    warnings.warn("neuralforecast not available. PatchTST, N-BEATS, N-HiTS will be disabled.")

import pandas as pd

try:
    from chronos import ChronosPipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import path
        from chronos_forecast import ChronosPipeline
        CHRONOS_AVAILABLE = True
    except ImportError:
        CHRONOS_AVAILABLE = False
        warnings.warn("chronos-forecast not available. Chronos-2 will be disabled.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("xgboost not available. Meta-learner will be disabled.")

from .delphi_core import DELPHICore
from .baselines import ETSModel, SNAIVEModel

warnings.filterwarnings('ignore')


class MetaEnsemble:
    """
    Meta-ensemble combining multiple forecasting models with XGBoost weighting.
    """
    
    def __init__(
        self,
        models: List[str],
        forecast_horizon: int = 26,
        meta_learner_config: Optional[Dict] = None,
        model_configs: Optional[Dict] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize meta-ensemble.
        
        Args:
            models: List of model names to include
            forecast_horizon: Forecast horizon
            meta_learner_config: Configuration for XGBoost meta-learner
            model_configs: Configuration dictionary for individual models
            device: Device for neural models
        """
        self.models = models
        self.forecast_horizon = forecast_horizon
        self.device = device
        self.model_instances = {}
        self.model_configs = model_configs or {}
        
        # Store configs for later use
        self.chronos_config = self.model_configs.get('chronos2', {})
        self.xlstm_config = self.model_configs.get('xlstmtime', {})
        self.neuralforecast_instance = None
        
        # Initialize models
        self._initialize_models()
        
        # Meta-learner
        meta_config = meta_learner_config or {}
        if XGBOOST_AVAILABLE:
            self.meta_learner = xgb.XGBRegressor(
                n_estimators=meta_config.get('n_estimators', 100),
                max_depth=meta_config.get('max_depth', 6),
                learning_rate=meta_config.get('learning_rate', 0.1),
                subsample=meta_config.get('subsample', 0.8),
                colsample_bytree=meta_config.get('colsample_bytree', 0.8),
                random_state=42
            )
        else:
            self.meta_learner = None
    
    def _initialize_models(self):
        """Initialize all model instances."""
        for model_name in self.models:
            if model_name == 'delphi_core':
                # DELPHI core will be set separately
                continue
            elif model_name == 'ets':
                self.model_instances[model_name] = ETSModel()
            elif model_name == 'snaive':
                self.model_instances[model_name] = SNAIVEModel()
            elif model_name == 'patchtst' and NEURALFORECAST_AVAILABLE:
                # PatchTST will be initialized with data
                self.model_instances[model_name] = None
            elif model_name == 'nbeats' and NEURALFORECAST_AVAILABLE:
                # N-BEATS will be initialized with data
                self.model_instances[model_name] = None
            elif model_name == 'nhits' and NEURALFORECAST_AVAILABLE:
                # N-HiTS will be initialized with data
                self.model_instances[model_name] = None
            elif model_name == 'chronos2' and CHRONOS_AVAILABLE:
                # Chronos-2 will be initialized with data
                try:
                    chronos_config = getattr(self, 'chronos_config', {})
                    model_size = chronos_config.get('model_size', 'small')
                    self.model_instances[model_name] = ChronosPipeline.from_pretrained(
                        model_name=f"amazon/chronos-{model_size}-base"
                    )
                except Exception as e:
                    warnings.warn(f"Failed to initialize Chronos-2: {e}")
                    self.model_instances[model_name] = None
            elif model_name == 'xlstmtime':
                # xLSTMTime integration
                try:
                    from .xlstm_time import xLSTMTimeModel
                    xlstm_config = getattr(self, 'xlstm_config', {})
                    self.model_instances[model_name] = xLSTMTimeModel(
                        input_size=xlstm_config.get('context_length', 52),
                        forecast_horizon=self.forecast_horizon,
                        hidden_size=xlstm_config.get('hidden_size', 128),
                        num_layers=xlstm_config.get('num_layers', 2),
                        device=self.device
                    )
                except ImportError:
                    # Fallback: use simplified xLSTM implementation
                    from .ensemble_correctors import TrendCorrectorRNN
                    warnings.warn("xLSTMTime package not available, using simplified xLSTM")
                    self.model_instances[model_name] = TrendCorrectorRNN(
                        input_dim=1,
                        hidden_size=128,
                        num_layers=2,
                        output_dim=self.forecast_horizon,
                        use_xlstm=True
                    )
                except Exception as e:
                    warnings.warn(f"Failed to initialize xLSTMTime: {e}")
                    self.model_instances[model_name] = None
            else:
                warnings.warn(f"Model {model_name} not available or not implemented")
    
    def set_delphi_core(self, delphi_model: DELPHICore):
        """Set DELPHI core model."""
        if 'delphi_core' in self.models:
            self.model_instances['delphi_core'] = delphi_model
    
    def _prepare_neuralforecast_data(
        self,
        data: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Convert data dictionary to neuralforecast format.
        
        neuralforecast expects DataFrame with columns: ['ds', 'y', 'unique_id']
        """
        records = []
        for series_id, ts in data.items():
            for i, value in enumerate(ts):
                records.append({
                    'unique_id': series_id,
                    'ds': i,  # Time index
                    'y': float(value)
                })
        return pd.DataFrame(records)
    
    def fit_neural_models(
        self,
        data: Dict[str, np.ndarray],
        context_length: int = 52,
        config: Optional[Dict] = None
    ):
        """
        Fit neural forecasting models (PatchTST, N-BEATS, N-HiTS).
        
        Args:
            data: Dictionary of series_id -> time series array
            context_length: Context length for neural models
            config: Optional configuration dictionary
        """
        if not NEURALFORECAST_AVAILABLE:
            return
        
        # Prepare data for neuralforecast
        df = self._prepare_neuralforecast_data(data)
        
        # Get model configs
        model_configs = config or {}
        patchtst_config = model_configs.get('patchtst', {})
        nbeats_config = model_configs.get('nbeats', {})
        nhits_config = model_configs.get('nhits', {})
        
        models_list = []
        
        # Initialize and fit PatchTST
        if 'patchtst' in self.models:
            try:
                patchtst = PatchTST(
                    h=self.forecast_horizon,
                    input_size=patchtst_config.get('context_length', context_length),
                    patch_len=patchtst_config.get('patch_len', 16),
                    stride=patchtst_config.get('stride', 8),
                    d_model=patchtst_config.get('d_model', 128),
                    n_heads=patchtst_config.get('n_heads', 8),
                    n_layers=patchtst_config.get('n_layers', 3),
                )
                models_list.append(patchtst)
                self.model_instances['patchtst'] = patchtst
            except Exception as e:
                warnings.warn(f"Failed to initialize PatchTST: {e}")
        
        # Initialize and fit N-BEATS
        if 'nbeats' in self.models:
            try:
                nbeats = NBEATS(
                    h=self.forecast_horizon,
                    input_size=nbeats_config.get('context_length', context_length),
                    n_blocks=nbeats_config.get('n_blocks', 3),
                    n_layers=nbeats_config.get('n_layers', 4),
                    hidden_size=nbeats_config.get('hidden_size', 256),
                )
                models_list.append(nbeats)
                self.model_instances['nbeats'] = nbeats
            except Exception as e:
                warnings.warn(f"Failed to initialize N-BEATS: {e}")
        
        # Initialize and fit N-HiTS
        if 'nhits' in self.models:
            try:
                nhits = NHITS(
                    h=self.forecast_horizon,
                    input_size=nhits_config.get('context_length', context_length),
                    n_blocks=nhits_config.get('n_blocks', 3),
                    n_layers=nhits_config.get('n_layers', 4),
                    hidden_size=nhits_config.get('hidden_size', 256),
                )
                models_list.append(nhits)
                self.model_instances['nhits'] = nhits
            except Exception as e:
                warnings.warn(f"Failed to initialize N-HiTS: {e}")
        
        # Fit all neural models
        if len(models_list) > 0:
            try:
                nf = NeuralForecast(models=models_list, freq='W')  # Weekly frequency
                nf.fit(df=df)
                # Store fitted NeuralForecast instance
                self.neuralforecast_instance = nf
            except Exception as e:
                warnings.warn(f"Failed to fit neural models: {e}")
                self.neuralforecast_instance = None
        else:
            self.neuralforecast_instance = None
    
    def predict_all_models(
        self,
        data: Dict[str, np.ndarray],
        delphi_forecast: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get predictions from all models.
        
        Args:
            data: Dictionary of series_id -> time series array
            delphi_forecast: DELPHI core forecasts (if available)
        
        Returns:
            Dictionary of model_name -> {series_id -> forecast}
        """
        all_forecasts = {}
        
        for model_name in self.models:
            if model_name == 'delphi_core':
                if delphi_forecast is not None:
                    all_forecasts[model_name] = delphi_forecast
                continue
            
            model = self.model_instances.get(model_name)
            if model is None:
                continue
            
            try:
                if model_name in ['ets', 'snaive']:
                    forecasts = model.forecast(data, self.forecast_horizon)
                    all_forecasts[model_name] = forecasts
                elif model_name in ['patchtst', 'nbeats', 'nhits']:
                    # NeuralForecast models
                    if hasattr(self, 'neuralforecast_instance') and self.neuralforecast_instance is not None:
                        try:
                            # Prepare data for prediction
                            df = self._prepare_neuralforecast_data(data)
                            # Get forecasts
                            forecasts_df = self.neuralforecast_instance.predict(df=df)
                            # Convert to dictionary format
                            forecasts_dict = {}
                            
                            # NeuralForecast column names mapping (model_name -> possible column names)
                            column_name_options = {
                                'patchtst': ['PatchTST', 'PATCHTST', 'patchtst'],
                                'nbeats': ['NBEATS', 'nbeats', 'NBeats'],
                                'nhits': ['NHITS', 'nhits', 'NHiTS']
                            }
                            
                            # Find the correct column name
                            col_name = None
                            for option in column_name_options.get(model_name, [model_name.upper()]):
                                if option in forecasts_df.columns:
                                    col_name = option
                                    break
                            
                            if col_name is None:
                                print(f"Warning: Could not find column for {model_name} in {list(forecasts_df.columns)}")
                                all_forecasts[model_name] = {}
                                continue
                            
                            for series_id in data.keys():
                                series_mask = forecasts_df['unique_id'] == series_id
                                if series_mask.any():
                                    series_forecasts = forecasts_df.loc[series_mask, col_name].values
                                    if len(series_forecasts) > 0:
                                        forecasts_dict[series_id] = series_forecasts[:self.forecast_horizon]
                            all_forecasts[model_name] = forecasts_dict
                        except Exception as e:
                            print(f"Error predicting with {model_name}: {e}")
                            all_forecasts[model_name] = {}
                    else:
                        all_forecasts[model_name] = {}
                
                elif model_name == 'chronos2':
                    # Chronos-2 model
                    try:
                        chronos_model = self.model_instances.get('chronos2')
                        if chronos_model is not None:
                            forecasts_dict = {}
                            for series_id, ts in data.items():
                                # Chronos expects numpy array
                                ts_array = np.array(ts).reshape(1, -1)
                                # Try different parameter names for compatibility
                                try:
                                    # Chronos v2 API uses prediction_length
                                    forecast = chronos_model.predict(
                                        context=ts_array,
                                        prediction_length=self.forecast_horizon
                                    )
                                except TypeError:
                                    # Fallback for older API
                                    forecast = chronos_model.predict(
                                        context=ts_array,
                                        prediction_horizon=self.forecast_horizon
                                    )
                                # Extract forecast values
                                if hasattr(forecast, 'mean'):
                                    forecasts_dict[series_id] = forecast.mean.cpu().numpy().flatten()
                                elif isinstance(forecast, np.ndarray):
                                    forecasts_dict[series_id] = forecast.flatten()[:self.forecast_horizon]
                                else:
                                    forecasts_dict[series_id] = np.array(forecast).flatten()[:self.forecast_horizon]
                            all_forecasts[model_name] = forecasts_dict
                        else:
                            all_forecasts[model_name] = {}
                    except Exception as e:
                        print(f"Error predicting with Chronos-2: {e}")
                        all_forecasts[model_name] = {}
                
                elif model_name == 'xlstmtime':
                    # xLSTMTime model
                    try:
                        xlstm_model = self.model_instances.get('xlstmtime')
                        if xlstm_model is not None:
                            forecasts_dict = {}
                            for series_id, ts in data.items():
                                # Convert to tensor
                                ts_tensor = torch.FloatTensor(ts[-52:]).unsqueeze(0).to(self.device)
                                # Get forecast
                                with torch.no_grad():
                                    forecast = xlstm_model(ts_tensor)
                                forecasts_dict[series_id] = forecast.cpu().numpy().flatten()[:self.forecast_horizon]
                            all_forecasts[model_name] = forecasts_dict
                        else:
                            all_forecasts[model_name] = {}
                    except Exception as e:
                        print(f"Error predicting with xLSTMTime: {e}")
                        all_forecasts[model_name] = {}
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                all_forecasts[model_name] = {}
        
        return all_forecasts
    
    def combine_forecasts(
        self,
        all_forecasts: Dict[str, Dict[str, np.ndarray]],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Combine forecasts from all models.
        
        Args:
            all_forecasts: Dictionary of model forecasts
            weights: Optional model weights (if None, use equal weights or meta-learner)
        
        Returns:
            Combined forecasts
        """
        # Get all series IDs
        series_ids = set()
        for model_forecasts in all_forecasts.values():
            series_ids.update(model_forecasts.keys())
        
        combined = {}
        
        for series_id in series_ids:
            forecasts_list = []
            model_names = []
            
            for model_name, model_forecasts in all_forecasts.items():
                if series_id in model_forecasts:
                    forecast = model_forecasts[series_id]
                    if len(forecast) == self.forecast_horizon:
                        forecasts_list.append(forecast)
                        model_names.append(model_name)
            
            if len(forecasts_list) == 0:
                continue
            
            # Combine forecasts
            if weights is None:
                # Equal weights
                combined[series_id] = np.mean(forecasts_list, axis=0)
            else:
                # Weighted combination
                weights_array = np.array([weights.get(name, 1.0) for name in model_names])
                weights_array = weights_array / weights_array.sum()  # Normalize
                combined[series_id] = np.average(forecasts_list, axis=0, weights=weights_array)
        
        return combined
    
    def fit_meta_learner(
        self,
        X: np.ndarray,
        y: np.ndarray
    ):
        """
        Fit XGBoost meta-learner on validation data.
        
        Args:
            X: Features (e.g., model predictions, time series features)
            y: True values
        """
        if self.meta_learner is not None:
            self.meta_learner.fit(X, y)
    
    def predict_with_meta_learner(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Predict using meta-learner.
        
        Args:
            X: Features
        
        Returns:
            Weighted predictions
        """
        if self.meta_learner is not None:
            return self.meta_learner.predict(X)
        else:
            # Fallback: return mean
            return np.mean(X, axis=1) if len(X.shape) > 1 else X

