"""
Loss functions for DELPHI training: ELBO, MSE, KL divergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


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
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        prior_logp: Optional[torch.Tensor] = None,
        posterior_logp: Optional[torch.Tensor] = None,
        posterior_probs: Optional[torch.Tensor] = None,
        stage: str = 'stage1'
    ) -> torch.Tensor:
        """
        Compute ELBO loss.
        
        Args:
            reconstruction: Reconstructed forecast
            target: Target forecast
            prior_logp: Prior log probabilities
            posterior_logp: Posterior log probabilities
            posterior_probs: Posterior probabilities
            stage: Training stage ('stage1' for emissions, 'stage2' for prior)
        
        Returns:
            ELBO loss
        """
        # Reconstruction term (MSE)
        recon_loss = self.mse_loss(reconstruction, target)
        
        if stage == 'stage1':
            # Stage 1: Train emissions/posterior, fix uniform prior
            if posterior_probs is not None:
                # Entropy regularization: -H(q(z|x))
                entropy = -torch.sum(
                    posterior_probs * torch.log(posterior_probs + 1e-8),
                    dim=-1
                ).mean()
                
                # KL term (posterior vs uniform prior)
                uniform_prior = torch.ones_like(posterior_probs) / posterior_probs.shape[-1]
                kl_term = F.kl_div(
                    torch.log(posterior_probs + 1e-8),
                    uniform_prior,
                    reduction='batchmean'
                )
                
                loss = recon_loss + self.kl_weight * kl_term - self.entropy_weight * entropy
            else:
                loss = recon_loss
        
        elif stage == 'stage2':
            # Stage 2: Train prior, freeze posterior/emissions
            if prior_logp is not None and posterior_logp is not None:
                # KL(q(z|x) || p(z))
                kl_term = (posterior_logp - prior_logp).mean()
                loss = self.kl_weight * kl_term
            else:
                loss = torch.tensor(0.0, device=reconstruction.device)
        
        else:
            loss = recon_loss
        
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
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        posterior_probs: Optional[torch.Tensor] = None,
        prior_logp: Optional[torch.Tensor] = None,
        posterior_logp: Optional[torch.Tensor] = None,
        stage: str = 'stage1'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            prediction: Model prediction
            target: Target values
            posterior_probs: Posterior probabilities
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
                prediction, target,
                prior_logp=prior_logp,
                posterior_logp=posterior_logp,
                posterior_probs=posterior_probs,
                stage=stage
            )
            
            return {
                'total_loss': elbo_loss,
                'recon_loss': torch.tensor(0.0, device=prediction.device),
                'elbo_loss': elbo_loss
            }
        
        # Stage 1: Reconstruction loss + ELBO
        recon_loss = self.mse_loss(prediction, target)
        
        # ELBO loss
        elbo_loss = self.elbo_loss(
            prediction, target,
            posterior_probs=posterior_probs,
            stage=stage
        )
        
        total_loss = self.recon_weight * recon_loss + elbo_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'elbo_loss': elbo_loss
        }

