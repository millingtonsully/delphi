"""
Uncertainty Quantifier: Enhanced UQ with aleatoric/epistemic decomposition.

Builds on existing predict_with_uncertainty to provide:
- Aleatoric vs epistemic uncertainty separation
- Regime uncertainty quantification
- Enhanced confidence intervals
"""

import torch
import numpy as np
from typing import Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class UncertaintyQuantifier:
    """
    Enhanced uncertainty quantification for DELPHI forecasts.
    
    Decomposes uncertainty into:
    - Aleatoric (data) uncertainty
    - Epistemic (model) uncertainty
    - Regime uncertainty
    """
    
    def __init__(self, num_samples: int = 100):
        """
        Initialize uncertainty quantifier.
        
        Args:
            num_samples: Number of samples for uncertainty estimation
        """
        self.num_samples = num_samples
    
    def decompose_uncertainty(
        self,
        model,
        x: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None
    ) -> Dict:
        """
        Decompose uncertainty into aleatoric and epistemic components.
        
        Args:
            model: DELPHICore model instance
            x: Input tensor (batch, seq_len, input_dim)
            parametric_forecast: Parametric baseline forecast (batch, output_dim)
            num_samples: Number of samples (overrides default)
        
        Returns:
            Dictionary with uncertainty decomposition
        """
        model.eval()
        num_samples = num_samples or self.num_samples
        
        batch_size = x.shape[0]
        horizon = model.output_dim
        
        # Get emission predictions from all correctors
        all_mus = []
        all_sigmas = []
        with torch.no_grad():
            for corrector in model.ensemble.correctors:
                mu, sigma = corrector(x)
                all_mus.append(mu)
                all_sigmas.append(sigma)
        
        emission_mu = torch.stack(all_mus, dim=0)  # (n_states, batch, horizon)
        emission_sigma = torch.stack(all_sigmas, dim=0)  # (n_states, batch, horizon)
        
        # Sample trajectories
        samples = []
        state_samples = []
        
        for _ in range(num_samples):
            # Sample states from prior
            emission_mu_for_prior = emission_mu.permute(1, 2, 0)  # (batch, horizon, n_states)
            states = model.hmm_gating.sample_states_from_prior(
                x_past=x,
                x_future=emission_mu_for_prior
            )  # (batch, horizon)
            
            # Get correction for sampled states
            correction_mu, correction_sigma = model.ensemble(x, states=states)
            
            # Sample from emission distribution (aleatoric uncertainty)
            eps = torch.randn_like(correction_mu)
            correction_sample = correction_mu + correction_sigma * eps
            
            # Combine with parametric forecast
            if parametric_forecast is not None:
                forecast_sample = parametric_forecast + correction_sample
            else:
                forecast_sample = correction_sample
            
            samples.append(forecast_sample)
            state_samples.append(states)
        
        samples = torch.stack(samples, dim=0)  # (num_samples, batch, horizon)
        state_samples = torch.stack(state_samples, dim=0)  # (num_samples, batch, horizon)
        
        # Compute statistics
        mean_forecast = samples.mean(dim=0)  # (batch, horizon)
        std_forecast = samples.std(dim=0)  # (batch, horizon)
        
        # Decompose uncertainty using proper UQ theory:
        # - Epistemic: Var(E[y | z_1:T]) - variance of mean predictions across state trajectories
        # - Aleatoric: E[Var(y | z_1:T)] - expected variance given states
        
        # Get state probabilities for weighted averaging
        with torch.no_grad():
            results = model(x, parametric_forecast=parametric_forecast)
            state_probs = results['state_probs']  # (batch, horizon, n_states)
        
        # Compute mean predictions for each trajectory (without aleatoric noise)
        # This requires recomputing forecasts without sampling noise
        trajectory_means = []
        for i in range(num_samples):
            states = state_samples[i]  # (batch, horizon)
            # Get correction means (without noise) for this trajectory
            correction_mu, _ = model.ensemble(x, states=states)
            if parametric_forecast is not None:
                trajectory_mean = parametric_forecast + correction_mu
            else:
                trajectory_mean = correction_mu
            trajectory_means.append(trajectory_mean)
        
        trajectory_means = torch.stack(trajectory_means, dim=0)  # (num_samples, batch, horizon)
        
        # Epistemic uncertainty: variance of mean predictions across trajectories
        # Var(E[y | z_1:T]) = variance across trajectories
        epistemic_uncertainty = trajectory_means.std(dim=0)  # (batch, horizon)
        
        # Aleatoric uncertainty: expected variance given states
        # E[Var(y | z_1:T)] = weighted average of emission variances
        aleatoric_variance = torch.zeros(batch_size, horizon, device=x.device)
        for k in range(model.n_states):
            prob_k = state_probs[:, :, k]  # (batch, horizon)
            sigma_k = emission_sigma[k]  # (batch, horizon)
            aleatoric_variance += prob_k * (sigma_k ** 2)
        aleatoric_uncertainty = torch.sqrt(aleatoric_variance)  # (batch, horizon)
        
        # Verify: total uncertainty should approximately equal sqrt(epistemic^2 + aleatoric^2)
        # Note: This is approximate due to sampling, but should be close
        total_uncertainty_approx = torch.sqrt(epistemic_uncertainty ** 2 + aleatoric_uncertainty ** 2)
        
        # Ensure epistemic is non-negative (should already be, but safety check)
        epistemic_uncertainty = torch.clamp(epistemic_uncertainty, min=0.0)
        
        return {
            'mean_forecast': mean_forecast.cpu().numpy(),
            'total_uncertainty': std_forecast.cpu().numpy(),
            'aleatoric_uncertainty': aleatoric_uncertainty.cpu().numpy(),
            'epistemic_uncertainty': epistemic_uncertainty.cpu().numpy(),
            'samples': samples.cpu().numpy(),
            'state_samples': state_samples.cpu().numpy(),
            'num_samples': num_samples
        }
    
    def regime_uncertainty(
        self,
        state_probs: torch.Tensor
    ) -> Dict:
        """
        Quantify uncertainty in regime detection itself.
        
        Args:
            state_probs: State probabilities (batch, horizon, n_states)
        
        Returns:
            Dictionary with regime uncertainty metrics
        """
        batch_size, horizon, n_states = state_probs.shape
        
        # Compute entropy (uncertainty measure)
        # Higher entropy = more uncertain about which state
        entropy = -torch.sum(
            state_probs * torch.log(state_probs + 1e-8),
            dim=-1
        )  # (batch, horizon)
        
        # Maximum entropy (uniform distribution)
        max_entropy = np.log(n_states)
        
        # Normalized uncertainty (0 = certain, 1 = completely uncertain)
        normalized_uncertainty = entropy / max_entropy
        
        # Compute confidence (inverse of uncertainty)
        confidence = 1.0 - normalized_uncertainty
        
        # Find timesteps with high uncertainty
        high_uncertainty_threshold = 0.5
        high_uncertainty_mask = normalized_uncertainty > high_uncertainty_threshold
        
        return {
            'entropy': entropy.cpu().numpy(),
            'normalized_uncertainty': normalized_uncertainty.cpu().numpy(),
            'confidence': confidence.cpu().numpy(),
            'mean_uncertainty': float(normalized_uncertainty.mean().item()),
            'max_uncertainty': float(normalized_uncertainty.max().item()),
            'high_uncertainty_timesteps': high_uncertainty_mask.cpu().numpy(),
            'max_entropy': max_entropy
        }
    
    def enhanced_confidence_intervals(
        self,
        model,
        x: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor] = None,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Compute enhanced confidence intervals with regime-aware adjustments.
        
        Args:
            model: DELPHICore model instance
            x: Input tensor (batch, seq_len, input_dim)
            parametric_forecast: Parametric baseline forecast (batch, output_dim)
            confidence_level: Confidence level (default 0.95)
        
        Returns:
            Dictionary with confidence intervals
        """
        # Get uncertainty decomposition (keep in torch tensors)
        uncertainty_results = self.decompose_uncertainty(model, x, parametric_forecast)
        
        # Convert numpy results back to torch tensors (matching device)
        mean_forecast = torch.tensor(uncertainty_results['mean_forecast'], device=x.device)
        total_uncertainty = torch.tensor(uncertainty_results['total_uncertainty'], device=x.device)
        
        # Get regime uncertainty (keep in torch tensors)
        with torch.no_grad():
            results = model(x, parametric_forecast=parametric_forecast)
            state_probs = results['state_probs']  # Already a torch tensor
        
        regime_uncertainty_results = self.regime_uncertainty(state_probs)
        regime_uncertainty = torch.tensor(
            regime_uncertainty_results['normalized_uncertainty'], 
            device=x.device
        )
        
        # Adjust confidence intervals based on regime uncertainty
        # Higher regime uncertainty -> wider intervals
        adjustment_factor = 1.0 + regime_uncertainty  # Scale by regime uncertainty
        adjusted_uncertainty = total_uncertainty * adjustment_factor
        
        # Compute z-score for confidence level
        # Use standard normal approximation: 95% = 1.96, 90% = 1.645, 99% = 2.576
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z_score = z_scores.get(confidence_level, 1.96)  # Default to 95%
        
        # Confidence intervals
        lower_bound = mean_forecast - z_score * adjusted_uncertainty
        upper_bound = mean_forecast + z_score * adjusted_uncertainty
        
        return {
            'mean_forecast': mean_forecast.numpy(),
            'lower_bound': lower_bound.numpy(),
            'upper_bound': upper_bound.numpy(),
            'confidence_level': confidence_level,
            'adjusted_uncertainty': adjusted_uncertainty.numpy(),
            'regime_uncertainty': regime_uncertainty.numpy(),
            'adjustment_factor': adjustment_factor.numpy()
        }

