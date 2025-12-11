"""
Variational Explainer: Leverages ELBO variational posteriors to explain regime shifts.

Uses the variational HMM structure to provide interpretable explanations of regime
transitions and state attributions with uncertainty quantification.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class VariationalExplainer:
    """
    Explainer that leverages variational inference structure for regime shift explanations.
    
    Uses ELBO variational posteriors to:
    - Compute regime shift probabilities
    - Attribute forecast variance to different HMM states
    - Decompose ELBO contributions for interpretability
    """
    
    def __init__(self, n_states: int = 4, state_names: Optional[list] = None):
        """
        Initialize variational explainer.
        
        Args:
            n_states: Number of HMM states (default 4)
            state_names: Optional names for states (default: Trend, Seasonality, Volatility, External)
        """
        self.n_states = n_states
        if state_names is None:
            self.state_names = ['Trend', 'Seasonality', 'Volatility', 'External Signal']
        else:
            self.state_names = state_names
        
        if len(self.state_names) != n_states:
            raise ValueError(f"Number of state names ({len(self.state_names)}) must match n_states ({n_states})")
    
    def explain_regime_shifts(
        self,
        model,
        x: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Explain regime shifts using HMM state probabilities.
        
        Args:
            model: DELPHICore model instance
            x: Input tensor (batch, seq_len, input_dim)
            parametric_forecast: Parametric baseline forecast (batch, output_dim)
        
        Returns:
            Dictionary with regime shift explanations:
                - state_probs: Per-timestep state probabilities (batch, horizon, n_states)
                - dominant_states: Most likely state per timestep (batch, horizon)
                - regime_transitions: Transition probabilities between states
                - shift_probabilities: Probability of regime shifts at each timestep
        """
        model.eval()
        with torch.no_grad():
            # Get model forward pass to extract state probabilities
            results = model(x, parametric_forecast=parametric_forecast, return_states=True)
            state_probs = results['state_probs']  # (batch, horizon, n_states)
            states = results.get('states')  # (batch, horizon) if available
            
            batch_size, horizon, n_states = state_probs.shape
            
            # Get dominant state per timestep
            dominant_states = torch.argmax(state_probs, dim=-1)  # (batch, horizon)
            
            # Compute regime shift probabilities
            # Shift occurs when dominant state changes between consecutive timesteps
            shift_probabilities = torch.zeros(batch_size, horizon, device=x.device)
            for t in range(1, horizon):
                prev_state = dominant_states[:, t-1]
                curr_state = dominant_states[:, t]
                shift_probabilities[:, t] = (prev_state != curr_state).float()
            
            # Compute transition probabilities from state_probs
            # Average transition probability across batch
            transition_probs = torch.zeros(horizon - 1, n_states, n_states, device=x.device)
            for t in range(horizon - 1):
                prev_probs = state_probs[:, t, :]  # (batch, n_states)
                curr_probs = state_probs[:, t+1, :]  # (batch, n_states)
                # Outer product to get transition probabilities
                for i in range(n_states):
                    for j in range(n_states):
                        transition_probs[t, i, j] = (prev_probs[:, i] * curr_probs[:, j]).mean()
            
            # Get regime explanation from model if available
            regime_info = model.get_regime_explanation(x)
            transition_matrices = regime_info.get('transition_matrices')
            
            # Safely convert transition matrices to numpy if available
            transition_matrices_np = None
            if transition_matrices is not None:
                transition_matrices_np = transition_matrices.cpu().numpy()
            
            return {
                'state_probs': state_probs.cpu().numpy(),
                'dominant_states': dominant_states.cpu().numpy(),
                'regime_transitions': transition_probs.cpu().numpy(),
                'shift_probabilities': shift_probabilities.cpu().numpy(),
                'transition_matrices': transition_matrices_np,
                'state_names': self.state_names,
                'horizon': horizon
            }
    
    def compute_state_attribution(
        self,
        state_probs: torch.Tensor,
        emission_mu: torch.Tensor,
        emission_sigma: torch.Tensor,
        forecast: torch.Tensor
    ) -> Dict:
        """
        Attribute forecast variance to different HMM states using variational posteriors.
        
        Args:
            state_probs: State probabilities (batch, horizon, n_states)
            emission_mu: Emission means from all states (n_states, batch, horizon)
            emission_sigma: Emission standard deviations (n_states, batch, horizon)
            forecast: Final forecast (batch, horizon)
        
        Returns:
            Dictionary with state attributions:
                - state_contributions: Contribution of each state to forecast (batch, horizon, n_states)
                - variance_attribution: Variance attributed to each state
                - importance_scores: Importance of each state across horizon
        """
        batch_size, horizon, n_states = state_probs.shape
        
        # Compute weighted contribution of each state
        # state_probs: (batch, horizon, n_states)
        # emission_mu: (n_states, batch, horizon)
        state_contributions = torch.zeros(batch_size, horizon, n_states, device=state_probs.device)
        
        for k in range(n_states):
            # Weight emission by state probability
            mu_k = emission_mu[k]  # (batch, horizon)
            prob_k = state_probs[:, :, k]  # (batch, horizon)
            state_contributions[:, :, k] = mu_k * prob_k
        
        # Compute variance attribution
        # Variance from each state's contribution
        variance_attribution = torch.zeros(batch_size, horizon, n_states, device=state_probs.device)
        for k in range(n_states):
            sigma_k = emission_sigma[k]  # (batch, horizon)
            prob_k = state_probs[:, :, k]  # (batch, horizon)
            # Weighted variance contribution
            variance_attribution[:, :, k] = (sigma_k ** 2) * prob_k
        
        # Compute importance scores (average contribution magnitude)
        importance_scores = torch.abs(state_contributions).mean(dim=(0, 1))  # (n_states,)
        importance_scores = importance_scores / (importance_scores.sum() + 1e-8)  # Normalize
        
        return {
            'state_contributions': state_contributions.cpu().numpy(),
            'variance_attribution': variance_attribution.cpu().numpy(),
            'importance_scores': importance_scores.cpu().numpy(),
            'state_names': self.state_names
        }
    
    def decompose_elbo_contributions(
        self,
        model,
        x: torch.Tensor,
        target: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Decompose ELBO loss into per-state contributions for interpretability.
        
        Args:
            model: DELPHICore model instance
            x: Input tensor (batch, seq_len, input_dim)
            target: Target values (batch, horizon)
            parametric_forecast: Parametric baseline forecast (batch, output_dim)
        
        Returns:
            Dictionary with ELBO decomposition:
                - per_state_log_likelihood: Log-likelihood contribution per state
                - per_state_kl: KL divergence per state (if available)
                - total_elbo: Total ELBO value
                - state_contributions: Breakdown by state
        """
        model.eval()
        with torch.no_grad():
            # Get model outputs
            results = model(x, parametric_forecast=parametric_forecast)
            state_probs = results['state_probs']  # (batch, horizon, n_states)
            emission_mu = results['emission_mu']  # (n_states, batch, horizon)
            emission_sigma = results['emission_sigma']  # (n_states, batch, horizon)
            
            batch_size, horizon, n_states = state_probs.shape
            
            # Compute log-likelihood for each state
            per_state_log_likelihood = torch.zeros(n_states, device=x.device)
            
            for k in range(n_states):
                mu_k = emission_mu[k]  # (batch, horizon)
                sigma_k = emission_sigma[k]  # (batch, horizon)
                prob_k = state_probs[:, :, k]  # (batch, horizon)
                
                # Gaussian log-likelihood: log N(target | mu_k, sigma_k^2)
                # log N(x|mu,sigma^2) = -0.5 * log(2*pi*sigma^2) - 0.5 * ((x-mu)/sigma)^2
                log_2pi_sigma2 = torch.log(2 * np.pi * sigma_k ** 2 + 1e-8)
                squared_error = ((target - mu_k) / (sigma_k + 1e-8)) ** 2
                log_likelihood_k = -0.5 * (log_2pi_sigma2 + squared_error)  # (batch, horizon)
                
                # Weight by state probability and average
                weighted_log_likelihood = (log_likelihood_k * prob_k).mean()
                per_state_log_likelihood[k] = weighted_log_likelihood
            
            # Total ELBO approximation (reconstruction term)
            total_elbo = per_state_log_likelihood.sum()
            
            # State contributions (normalized)
            state_contributions = per_state_log_likelihood / (total_elbo.abs() + 1e-8)
            
            return {
                'per_state_log_likelihood': per_state_log_likelihood.cpu().numpy(),
                'total_elbo': total_elbo.item(),
                'state_contributions': state_contributions.cpu().numpy(),
                'state_names': self.state_names
            }

