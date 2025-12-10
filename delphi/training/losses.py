"""
Loss functions for DELPHI training: ELBO with log-likelihood, KL divergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

EPS = 1e-6


def gaussian_log_likelihood(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor
) -> torch.Tensor:
    """
    Compute Gaussian log-likelihood: log N(x | mu, sigma^2).
    
    Formula: -0.5 * log(2*pi*sigma^2) - 0.5 * ((x - mu) / sigma)^2
    
    Args:
        x: Observed values (batch, horizon) or (batch,)
        mu: Mean values (batch, horizon) or (batch,)
        sigma: Standard deviation values (batch, horizon) or (batch,)
    
    Returns:
        Log-likelihood values of same shape as x
    """
    # Clamp mean and sigma for numerical stability / to prevent blow-ups
    mu = torch.clamp(mu, -5.0, 5.0)
    # Ensure sigma has reasonable bounds for numerical stability
    sigma = torch.clamp(sigma, min=EPS, max=10.0)
    
    # log N(x | mu, sigma^2) = -0.5 * log(2*pi*sigma^2) - 0.5 * ((x - mu) / sigma)^2
    log_2pi_sigma2 = torch.log(2 * torch.pi * sigma**2 + EPS)
    squared_error = ((x - mu) / sigma)**2
    log_likelihood = -0.5 * (log_2pi_sigma2 + squared_error)
    
    return log_likelihood


class ELBOLoss(nn.Module):
    """
    Evidence Lower BOund (ELBO) loss for variational HMM training.
    
    ELBO = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))
    """
    
    def __init__(
        self,
        kl_weight: float = 0.1,
        entropy_weight: float = 0.01
    ):
        """
        Initialize ELBO loss.
        
        Args:
            kl_weight: Weight for KL divergence term
            entropy_weight: Weight for entropy regularization
        """
        super().__init__()
        self.kl_weight = kl_weight
        self.entropy_weight = entropy_weight
    
    def forward(
        self,
        emission_mu: torch.Tensor,
        emission_sigma: torch.Tensor,
        target: torch.Tensor,
        posterior_probs: Optional[torch.Tensor] = None,
        prior_logp: Optional[torch.Tensor] = None,
        posterior_logp: Optional[torch.Tensor] = None,
        stage: str = 'stage1'
    ) -> torch.Tensor:
        """
        Compute ELBO loss.
        
        Args:
            emission_mu: Emission means from all states (n_states, batch, horizon)
            emission_sigma: Emission standard deviations from all states (n_states, batch, horizon)
            target: Target forecast (batch, horizon)
            posterior_probs: Posterior probabilities (batch, horizon, n_states) or (batch, n_states)
            prior_logp: Prior log probabilities (for Stage 2)
            posterior_logp: Posterior log probabilities (for Stage 2)
            stage: Training stage ('stage1' for emissions, 'stage2' for prior)
        
        Returns:
            ELBO loss
        """
        # Reconstruction term: E_q(z|x)[log p(x|z)]
        # For per-timestep states: sum_t sum_k q(z_t=k|x) * log N(x_t | mu_k[t], sigma_k[t])
        if stage == 'stage1':
            # Stage 1: Train emissions/posterior, fix uniform prior
            # Compute log-likelihood: E_q(z|x)[log p(x|z)]
            if posterior_probs is not None and len(posterior_probs.shape) == 3:
                # Per-timestep probabilities: (batch, horizon, n_states)
                # emission_mu: (n_states, batch, horizon)
                # emission_sigma: (n_states, batch, horizon)
                # target: (batch, horizon)
                
                n_states, batch_size, horizon = emission_mu.shape
                
                # Compute log-likelihood for each state and timestep
                # log_p[k, b, t] = log N(target[b, t] | mu[k, b, t], sigma[k, b, t]^2)
                log_likelihoods = []
                for k in range(n_states):
                    log_p_k = gaussian_log_likelihood(
                        target,  # (batch, horizon)
                        emission_mu[k],  # (batch, horizon)
                        emission_sigma[k]  # (batch, horizon)
                    )  # (batch, horizon)
                    log_likelihoods.append(log_p_k)
                
                # Stack: (n_states, batch, horizon)
                log_likelihoods = torch.stack(log_likelihoods, dim=0)
                
                # Weight by posterior: sum_k q(z_t=k|x) * log p(x_t | z_t=k)
                # posterior_probs: (batch, horizon, n_states)
                # log_likelihoods: (n_states, batch, horizon)
                weighted_log_p = torch.sum(
                    posterior_probs.permute(2, 0, 1) * log_likelihoods,  # (n_states, batch, horizon)
                    dim=0  # Sum over states
                )  # (batch, horizon)
                
                # Average over batch and timesteps (negative because we minimize loss)
                recon_loss = -weighted_log_p.mean()
                
                # Entropy regularization: -H(q(z|x)) averaged over timesteps
                entropy = -torch.sum(
                    posterior_probs * torch.log(posterior_probs + EPS),
                    dim=-1
                ).mean()  # Average over batch and timesteps
                
                # KL term (posterior vs uniform prior) per timestep
                uniform_prior = torch.ones_like(posterior_probs) / posterior_probs.shape[-1]
                kl_term = F.kl_div(
                    torch.log(posterior_probs + EPS).view(-1, posterior_probs.shape[-1]),
                    uniform_prior.view(-1, uniform_prior.shape[-1]),
                    reduction='batchmean'
                )
                
                loss = recon_loss + self.kl_weight * kl_term - self.entropy_weight * entropy
            elif posterior_probs is not None:
                # Legacy format: (batch, n_states) - use single state for all timesteps
                # This is a fallback for compatibility
                n_states, batch_size, horizon = emission_mu.shape
                
                # Use mean emission across timesteps for each state
                mean_mu = emission_mu.mean(dim=2)  # (n_states, batch)
                mean_sigma = emission_sigma.mean(dim=2)  # (n_states, batch)
                mean_target = target.mean(dim=1)  # (batch,)
                
                log_likelihoods = []
                for k in range(n_states):
                    log_p_k = gaussian_log_likelihood(
                        mean_target,  # (batch,)
                        mean_mu[k],  # (batch,)
                        mean_sigma[k]  # (batch,)
                    )  # (batch,)
                    log_likelihoods.append(log_p_k)
                
                log_likelihoods = torch.stack(log_likelihoods, dim=0)  # (n_states, batch)
                weighted_log_p = torch.sum(
                    posterior_probs.t() * log_likelihoods,  # (n_states, batch)
                    dim=0  # Sum over states
                )  # (batch,)
                
                recon_loss = -weighted_log_p.mean()
                
                entropy = -torch.sum(
                    posterior_probs * torch.log(posterior_probs + EPS),
                    dim=-1
                ).mean()
                
                uniform_prior = torch.ones_like(posterior_probs) / posterior_probs.shape[-1]
                kl_term = F.kl_div(
                    torch.log(posterior_probs + EPS),
                    uniform_prior,
                    reduction='batchmean'
                )
                
                loss = recon_loss + self.kl_weight * kl_term - self.entropy_weight * entropy
            else:
                # No posterior probabilities - just compute log-likelihood with uniform weighting
                # Average over states
                log_likelihoods = []
                for k in range(emission_mu.shape[0]):
                    log_p_k = gaussian_log_likelihood(target, emission_mu[k], emission_sigma[k])
                    log_likelihoods.append(log_p_k)
                log_likelihoods = torch.stack(log_likelihoods, dim=0)
                recon_loss = -log_likelihoods.mean()
                loss = recon_loss
        
        elif stage == 'stage2':
            # Stage 2: Train prior, freeze posterior/emissions
            # Only KL divergence term, no reconstruction loss
            if prior_logp is not None and posterior_logp is not None and posterior_probs is not None:
                # KL(q(z|x) || p(z)) = E_q[log q - log p]
                # posterior_probs: (batch, horizon, n_states)
                # posterior_logp: (batch, horizon, n_states)
                # prior_logp: (batch, horizon, n_states)
                kl_term = (posterior_probs * (posterior_logp - prior_logp)).sum(dim=-1).mean()
                loss = self.kl_weight * kl_term
            elif prior_logp is not None and posterior_logp is not None:
                # Fallback: simple difference (less accurate)
                kl_term = (posterior_logp - prior_logp).mean()
                loss = self.kl_weight * kl_term
            else:
                loss = torch.tensor(0.0, device=target.device)
        
        else:
            # Fallback: compute simple log-likelihood
            log_likelihoods = []
            for k in range(emission_mu.shape[0]):
                log_p_k = gaussian_log_likelihood(target, emission_mu[k], emission_sigma[k])
                log_likelihoods.append(log_p_k)
            log_likelihoods = torch.stack(log_likelihoods, dim=0)
            loss = -log_likelihoods.mean()
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for DELPHI training.
    """
    
    def __init__(
        self,
        recon_weight: float = 1.0,
        kl_weight: float = 0.1,
        entropy_weight: float = 0.01
    ):
        """
        Initialize combined loss.
        
        Args:
            recon_weight: Weight for reconstruction loss
            kl_weight: Weight for KL divergence
            entropy_weight: Weight for entropy regularization
        """
        super().__init__()
        self.recon_weight = recon_weight
        self.elbo_loss = ELBOLoss(kl_weight, entropy_weight)
    
    def forward(
        self,
        emission_mu: torch.Tensor,
        emission_sigma: torch.Tensor,
        target: torch.Tensor,
        posterior_probs: Optional[torch.Tensor] = None,
        prior_logp: Optional[torch.Tensor] = None,
        posterior_logp: Optional[torch.Tensor] = None,
        stage: str = 'stage1'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            emission_mu: Emission means from all states (n_states, batch, horizon)
            emission_sigma: Emission standard deviations from all states (n_states, batch, horizon)
            target: Target values (batch, horizon)
            posterior_probs: Posterior probabilities (batch, horizon, n_states) or (batch, n_states)
            prior_logp: Prior log probabilities (for Stage 2)
            posterior_logp: Posterior log probabilities (for Stage 2)
            stage: Training stage
        
        Returns:
            Dictionary of loss components
        """
        # For Stage 2, skip reconstruction loss (only train prior)
        if stage == 'stage2':
            # ELBO loss with prior/posterior for Stage 2
            elbo_loss = self.elbo_loss(
                emission_mu, emission_sigma, target,
                prior_logp=prior_logp,
                posterior_logp=posterior_logp,
                posterior_probs=posterior_probs,
                stage=stage
            )
            
            return {
                'total_loss': elbo_loss,
                'recon_loss': torch.tensor(0.0, device=target.device),
                'elbo_loss': elbo_loss
            }
        
        # Stage 1: Log-likelihood reconstruction loss + ELBO
        # ELBO loss includes the log-likelihood term
        elbo_loss = self.elbo_loss(
            emission_mu, emission_sigma, target,
            posterior_probs=posterior_probs,
            stage=stage
        )
        
        # Extract reconstruction loss component for reporting
        # We compute it separately for logging purposes
        if posterior_probs is not None and len(posterior_probs.shape) == 3:
            # Per-timestep log-likelihood
            n_states = emission_mu.shape[0]
            log_likelihoods = []
            for k in range(n_states):
                log_p_k = gaussian_log_likelihood(target, emission_mu[k], emission_sigma[k])
                log_likelihoods.append(log_p_k)
            log_likelihoods = torch.stack(log_likelihoods, dim=0)
            weighted_log_p = torch.sum(
                posterior_probs.permute(2, 0, 1) * log_likelihoods,
                dim=0
            )
            recon_loss = -weighted_log_p.mean()
        else:
            # Fallback
            recon_loss = torch.tensor(0.0, device=target.device)
        
        total_loss = elbo_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'elbo_loss': elbo_loss
        }

