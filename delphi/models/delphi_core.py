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
            dropout=dropout,
            horizon=output_dim  # Forecast horizon for per-timestep probabilities
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
        future_observations: Optional[torch.Tensor] = None,
        return_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DELPHI model.
        
        Process:
        1. Compute all emission law predictions (mu, sigma)
        2. Use mu for prior computation (or future observations for posterior during training)
        3. Sample/use states per timestep
        4. Select emissions based on per-timestep states
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim) - past observations
            parametric_forecast: Parametric baseline forecast (batch, output_dim)
            future_observations: Future observations (batch, output_dim, 1) for posterior during training
            return_states: Whether to return HMM states
        
        Returns:
            Dictionary with:
                - 'correction': Correction from ensemble (batch, output_dim)
                - 'forecast': Final forecast (batch, output_dim)
                - 'states': HMM states if return_states=True (batch, output_dim)
                - 'state_probs': HMM state probabilities (batch, output_dim, n_states)
        """
        batch_size = x.shape[0]
        
        # Step 1: Compute all emission law predictions (mu, sigma from all correctors)
        # Get mu and sigma from all ensemble members
        all_mus = []
        all_sigmas = []
        for corrector in self.ensemble.correctors:
            mu, sigma = corrector(x)  # (batch, output_dim) each
            all_mus.append(mu)
            all_sigmas.append(sigma)
        
        # Stack: (n_states, batch, output_dim)
        emission_mu = torch.stack(all_mus, dim=0)
        emission_sigma = torch.stack(all_sigmas, dim=0)
        
        # Reshape for prior: (batch, output_dim, n_states)
        emission_mu_for_prior = emission_mu.permute(1, 2, 0)
        
        # Step 2: Compute state probabilities
        if self.training and future_observations is not None:
            # Training: use posterior with future observations
            # future_observations: (batch, output_dim, 1)
            state_probs = self.hmm_gating(
                x_past=x,
                x_future=future_observations,
                mode='posterior'
            )  # (batch, output_dim, n_states)
            
            # Sample states from posterior per timestep
            # For each timestep, sample from categorical distribution
            states = torch.zeros(batch_size, self.output_dim, dtype=torch.long, device=x.device)
            for t in range(self.output_dim):
                probs_t = state_probs[:, t, :]  # (batch, n_states)
                states[:, t] = torch.multinomial(probs_t, 1).squeeze(-1)
        else:
            # Inference: use prior with emission predictions
            # Sample states from prior using Markov chain
            states = self.hmm_gating.sample_states_from_prior(
                x_past=x,
                x_future=emission_mu_for_prior
            )  # (batch, output_dim)
            
            # Get state probabilities from prior (for return value)
            init_probs, trans_matrices = self.hmm_gating.get_prior_components(
                x_past=x,
                x_future=emission_mu_for_prior
            )
            # Compute per-timestep probabilities from prior
            state_probs = torch.zeros(batch_size, self.output_dim, self.n_states, device=x.device)
            state_probs[:, 0, :] = init_probs
            for t in range(1, self.output_dim):
                prev_probs = state_probs[:, t-1, :].unsqueeze(1)  # (batch, 1, n_states)
                trans_matrix = trans_matrices[:, t-1, :, :]  # (batch, n_states, n_states)
                state_probs[:, t, :] = torch.bmm(prev_probs, trans_matrix).squeeze(1)
        
        # Step 3: Get correction from ensemble with per-timestep state routing
        correction_mu, correction_sigma = self.ensemble(x, states=states)  # (batch, output_dim) each
        
        # Combine with parametric forecast (add to mu, keep sigma from emissions)
        if parametric_forecast is not None:
            mu_final = parametric_forecast + correction_mu
            sigma_final = correction_sigma
            forecast = mu_final  # Forecast is the mean
        else:
            # If no parametric forecast, use correction only
            mu_final = correction_mu
            sigma_final = correction_sigma
            forecast = mu_final

        # Clamp final mean to prevent extreme outputs (stabilizes ELBO)
        mu_final = torch.clamp(mu_final, -5.0, 5.0)
        forecast = mu_final
        
        result = {
            'correction': correction_mu,  # Keep for backward compatibility
            'forecast': forecast,
            'mu': mu_final,
            'sigma': sigma_final,
            'emission_mu': emission_mu,  # All emission mus for loss computation
            'emission_sigma': emission_sigma,  # All emission sigmas for loss computation
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
            # Compute emission predictions for prior
            all_mus = []
            for corrector in self.ensemble.correctors:
                mu, _ = corrector(x)
                all_mus.append(mu)
            emission_mu = torch.stack(all_mus, dim=0)  # (n_states, batch, output_dim)
            emission_mu_for_prior = emission_mu.permute(1, 2, 0)  # (batch, output_dim, n_states)
            
            for _ in range(num_samples):
                # Sample per-timestep trajectory via Markov chain from prior
                states = self.hmm_gating.sample_states_from_prior(
                    x_past=x,
                    x_future=emission_mu_for_prior
                )  # (batch, output_dim)
                
                correction_mu, _ = self.ensemble(x, states=states)
                
                if parametric_forecast is not None:
                    pred = parametric_forecast + correction_mu
                else:
                    pred = correction_mu
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
        x: torch.Tensor,
        future_observations: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get interpretable regime explanation from HMM states.
        
        Args:
            x: Input tensor (past observations)
            future_observations: Future observations (batch, output_dim, 1) for posterior
        
        Returns:
            Dictionary with regime information
        """
        # Compute emission predictions for prior
        all_mus = []
        for corrector in self.ensemble.correctors:
            mu, _ = corrector(x)
            all_mus.append(mu)
        emission_mu = torch.stack(all_mus, dim=0)  # (n_states, batch, output_dim)
        emission_mu_for_prior = emission_mu.permute(1, 2, 0)  # (batch, output_dim, n_states)
        
        if future_observations is not None:
            state_probs = self.hmm_gating.get_state_probs(x, future_observations)
        else:
            # Use prior probabilities
            init_probs, trans_matrices = self.hmm_gating.get_prior_components(
                x_past=x,
                x_future=emission_mu_for_prior
            )
            batch_size = x.shape[0]
            state_probs = torch.zeros(batch_size, self.output_dim, self.n_states, device=x.device)
            state_probs[:, 0, :] = init_probs
            for t in range(1, self.output_dim):
                prev_probs = state_probs[:, t-1, :].unsqueeze(1)
                trans_matrix = trans_matrices[:, t-1, :, :]
                state_probs[:, t, :] = torch.bmm(prev_probs, trans_matrix).squeeze(1)
        
        return {
            'state_probs': state_probs,  # (batch, horizon, n_states)
            'transition_matrices': trans_matrices if future_observations is None else None,
            'dominant_states': torch.argmax(state_probs, dim=-1)  # (batch, horizon)
        }


