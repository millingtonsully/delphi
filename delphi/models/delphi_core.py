"""
Core DELPHI Model: Integrates Variational HMM Gating with Deep Ensemble Correctors.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
import numpy as np

from .hmm_gating import VariationalHMMGating
from .ensemble_correctors import DeepEnsembleCorrectors


class DELPHICore(nn.Module):
    """
    Core DELPHI model combining:
    - Variational HMM gating for regime detection (xLSTM-based)
    - Deep ensemble of specialized xLSTM correctors
    - Probabilistic uncertainty quantification
    
    All components use xLSTM with exponential gating for enhanced long-term memory.
    """
    
    def __init__(
        self,
        input_dim: int = 3,  # [z_t, w_t, ŷ_pred_t]
        n_states: int = 4,
        hmm_hidden_size: int = 64,
        hmm_num_layers: int = 2,
        ensemble_hidden_size: int = 64,
        ensemble_num_layers: int = 2,
        output_dim: int = 26,  # Forecast horizon
        n_ensemble_members: int = 5,
        dropout: float = 0.2
    ):
        """
        Initialize DELPHI core model.
        
        Args:
            input_dim: Input feature dimension
            n_states: Number of HMM latent states
            hmm_hidden_size: HMM xLSTM hidden size
            hmm_num_layers: HMM xLSTM layers
            ensemble_hidden_size: Ensemble xLSTM hidden size
            ensemble_num_layers: Ensemble xLSTM layers
            output_dim: Forecast horizon
            n_ensemble_members: Number of ensemble members (M=5)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.n_states = n_states
        self.output_dim = output_dim
        
        # Variational HMM Gating (xLSTM-based)
        self.hmm_gating = VariationalHMMGating(
            input_dim=input_dim,
            n_states=n_states,
            hidden_size=hmm_hidden_size,
            num_layers=hmm_num_layers,
            dropout=dropout
        )
        
        # Deep Ensemble Correctors (xLSTM-based)
        self.ensemble = DeepEnsembleCorrectors(
            input_dim=input_dim,
            hidden_size=ensemble_hidden_size,
            num_layers=ensemble_num_layers,
            output_dim=output_dim,
            dropout=dropout,
            n_members=n_ensemble_members
        )
    
    def forward(
        self,
        x: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor] = None,
        return_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DELPHI model.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            parametric_forecast: Parametric baseline forecast (batch, output_dim)
            return_states: Whether to return HMM states
        
        Returns:
            Dictionary with:
                - 'correction': Correction from ensemble (batch, output_dim)
                - 'forecast': Final forecast (batch, output_dim)
                - 'states': HMM states if return_states=True
                - 'state_probs': HMM state probabilities
        """
        batch_size = x.shape[0]
        
        # Get HMM state probabilities (posterior for training)
        state_probs = self.hmm_gating(x, mode='posterior')
        
        if self.training:
            # Training: sample single state from posterior
            states = torch.multinomial(state_probs, 1).squeeze(-1)  # (batch,)
            states_expanded = states.unsqueeze(1).expand(-1, x.shape[1])  # (batch, seq_len)
        else:
            # Inference: use per-timestep trajectory sampling via Markov chain
            states_expanded = self.hmm_gating(x, mode='prior')  # (batch, seq_len)
            states = states_expanded[:, -1]  # Last timestep state for return
        
        # Get correction from ensemble with state-based routing
        correction = self.ensemble(x, states=states_expanded)
        
        # Combine with parametric forecast
        if parametric_forecast is not None:
            forecast = parametric_forecast + correction
        else:
            # If no parametric forecast, use correction only
            forecast = correction
        
        result = {
            'correction': correction,
            'forecast': forecast,
            'state_probs': state_probs
        }
        
        if return_states:
            result['states'] = states
        
        return result
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor] = None,
        num_samples: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions with uncertainty via HMM trajectory sampling.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            parametric_forecast: Parametric baseline forecast (batch, output_dim)
            num_samples: Number of trajectory samples for uncertainty estimation
        
        Returns:
            Dictionary with:
                - 'mean': Mean forecast (batch, output_dim)
                - 'std': Standard deviation (batch, output_dim)
                - 'lower_95': Lower 95% confidence bound (batch, output_dim)
                - 'upper_95': Upper 95% confidence bound (batch, output_dim)
                - 'samples': All samples (num_samples, batch, output_dim)
        """
        self.eval()
        preds = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Sample per-timestep trajectory via Markov chain
                states = self.hmm_gating(x, mode='prior')  # (batch, seq_len)
                correction = self.ensemble(x, states=states)
                
                if parametric_forecast is not None:
                    pred = parametric_forecast + correction
                else:
                    pred = correction
                preds.append(pred)
        
        # Stack samples: (num_samples, batch, output_dim)
        preds = torch.stack(preds, dim=0)
        
        # Compute statistics
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        
        return {
            'mean': mean,
            'std': std,
            'lower_95': mean - 1.96 * std,
            'upper_95': mean + 1.96 * std,
            'samples': preds
        }
    
    def get_regime_explanation(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get interpretable regime explanation from HMM states.
        
        Args:
            x: Input tensor
        
        Returns:
            Dictionary with regime information
        """
        state_probs = self.hmm_gating.get_state_probs(x)
        transition_matrix = self.hmm_gating.get_transition_matrix(x)
        
        return {
            'state_probs': state_probs,
            'transition_matrix': transition_matrix,
            'dominant_state': torch.argmax(state_probs, dim=-1)
        }


