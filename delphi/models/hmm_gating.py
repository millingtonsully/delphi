"""
Variational HMM Gating Module for DELPHI.
Implements regime detection and routing via Markovian latent states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np

from .xlstm_layer import xLSTM


class VariationalHMMGating(nn.Module):
    """
    Variational HMM Gating Module for regime detection and routing.
    
    Implements per-timestep state probabilities:
    - Posterior: q(z_t | x_past, y_future) outputs (horizon, K) probabilities
    - Prior: p(z_t | x_past, mu_future) outputs initial state + (horizon-1, K, K) transition matrices
    
    Models temporal regime persistence via Markov chains with variational
    posterior for tractable inference. Supports two-stage ELBO training.
    Uses xLSTM with exponential gating for enhanced long-term memory.
    """
    
    def __init__(
        self,
        input_dim: int = 4,  # [z_t, w_t, ŷ_pred_t, vol_t]
        n_states: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 26  # Forecast horizon for per-timestep probabilities
    ):
        """
        Initialize Variational HMM Gating module.
        
        Args:
            input_dim: Dimension of input features
            n_states: Number of latent discrete states
            hidden_size: Hidden size for xLSTM layers
            num_layers: Number of xLSTM layers
            dropout: Dropout rate
            horizon: Forecast horizon (for per-timestep state probabilities)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.n_states = n_states
        self.hidden_size = hidden_size
        self.horizon = horizon
        
        # Prior model: takes past input + future emission predictions (mu)
        # Past input xLSTM
        self.prior_past_xlstm = xLSTM(
            input_dim, hidden_size, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Future emission predictions xLSTM (takes mu: horizon × K)
        self.prior_future_xlstm = xLSTM(
            n_states, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Prior outputs: initial state + transition matrices
        # Concatenate past and future features
        self.prior_fc_initial = nn.Linear(hidden_size * 2, n_states)
        self.prior_fc_transitions = nn.Linear(
            hidden_size * 2, 
            (horizon - 1) * n_states * n_states
        )
        
        # Posterior model: takes past input + future observations
        # Past input xLSTM
        self.posterior_past_xlstm = xLSTM(
            input_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Future observations xLSTM (takes y: horizon × 1)
        self.posterior_future_xlstm = xLSTM(
            1, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Posterior outputs: per-timestep state probabilities (horizon × K)
        # Concatenate past and future features
        self.posterior_fc = nn.Linear(hidden_size * 2, horizon * n_states)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x_past: torch.Tensor,
        x_future: Optional[torch.Tensor] = None,
        mode: str = 'posterior'
    ) -> torch.Tensor:
        """
        Forward pass through HMM gating module.
        
        Args:
            x_past: Past input tensor of shape (batch, past_seq_len, input_dim)
            x_future: Future input tensor
                - For posterior: future observations (batch, horizon, 1)
                - For prior: future emission predictions (batch, horizon, n_states)
            mode: 'posterior' for variational posterior, 'prior' for prior
        
        Returns:
            - Posterior mode: State probabilities (batch, horizon, n_states)
            - Prior mode: Tuple of (initial_probs, transition_matrices)
                - initial_probs: (batch, n_states)
                - transition_matrices: (batch, horizon-1, n_states, n_states)
        """
        batch_size = x_past.shape[0]
        
        if mode == 'posterior':
            # Variational posterior: q(z_t | x_past, y_future)
            # Output: (batch, horizon, n_states) - per-timestep state probabilities
            if x_future is None:
                raise ValueError("Posterior mode requires x_future (future observations)")
            
            # Process past input
            past_out, _ = self.posterior_past_xlstm(x_past)
            past_feat = past_out.mean(dim=1)  # (batch, hidden_size)
            past_feat = self.dropout(past_feat)
            
            # Process future observations
            future_out, _ = self.posterior_future_xlstm(x_future)
            future_feat = future_out.mean(dim=1)  # (batch, hidden_size)
            future_feat = self.dropout(future_feat)
            
            # Concatenate past and future features
            combined_feat = torch.cat([past_feat, future_feat], dim=-1)  # (batch, 2*hidden_size)
            
            # Output per-timestep state probabilities
            logits = self.posterior_fc(combined_feat)  # (batch, horizon * n_states)
            logits = logits.view(batch_size, self.horizon, self.n_states)
            probs = F.softmax(logits, dim=-1)  # (batch, horizon, n_states)
            
            return probs
        
        elif mode == 'prior':
            # Prior: p(z_t | x_past, mu_future)
            # Output: (initial_probs, transition_matrices)
            if x_future is None:
                raise ValueError("Prior mode requires x_future (future emission predictions mu)")
            
            # Process past input
            past_out, _ = self.prior_past_xlstm(x_past)
            past_feat = past_out.mean(dim=1)  # (batch, hidden_size)
            past_feat = self.dropout(past_feat)
            
            # Process future emission predictions (mu: horizon × n_states)
            future_out, _ = self.prior_future_xlstm(x_future)
            future_feat = future_out.mean(dim=1)  # (batch, hidden_size)
            future_feat = self.dropout(future_feat)
            
            # Concatenate past and future features
            combined_feat = torch.cat([past_feat, future_feat], dim=-1)  # (batch, 2*hidden_size)
            
            # Initial state probabilities
            init_logits = self.prior_fc_initial(combined_feat)  # (batch, n_states)
            init_probs = F.softmax(init_logits, dim=-1)  # (batch, n_states)
            
            # Per-timestep transition matrices
            trans_logits = self.prior_fc_transitions(combined_feat)  # (batch, (horizon-1)*K*K)
            trans_logits = trans_logits.view(
                batch_size, self.horizon - 1, self.n_states, self.n_states
            )
            trans_matrices = F.softmax(trans_logits, dim=-1)  # (batch, horizon-1, K, K)
            
            return init_probs, trans_matrices
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'posterior' or 'prior'")
    
    def sample_states_from_prior(
        self,
        x_past: torch.Tensor,
        x_future: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample state trajectory from prior using Markov chain.
        
        Args:
            x_past: Past input (batch, past_seq_len, input_dim)
            x_future: Future emission predictions (batch, horizon, n_states)
        
        Returns:
            Sampled states (batch, horizon) - one state per timestep
        """
        init_probs, trans_matrices = self.forward(x_past, x_future, mode='prior')
        
        batch_size = x_past.shape[0]
        states = []
        
        # Sample initial state
        z_0 = torch.multinomial(init_probs, 1).squeeze(-1)  # (batch,)
        states.append(z_0)
        
        # Sample subsequent states using transition matrices
        z_prev = z_0
        for t in range(self.horizon - 1):
            # Get transition probabilities for current timestep
            trans_matrix = trans_matrices[:, t, :, :]  # (batch, K, K)
            
            # Get probabilities for next state given previous state
            z_prev_onehot = F.one_hot(z_prev, self.n_states).float()  # (batch, K)
            next_probs = torch.bmm(
                z_prev_onehot.unsqueeze(1),  # (batch, 1, K)
                trans_matrix  # (batch, K, K)
            ).squeeze(1)  # (batch, K)
            
            # Sample next state
            z_next = torch.multinomial(next_probs, 1).squeeze(-1)  # (batch,)
            states.append(z_next)
            z_prev = z_next
        
        # Stack to get (batch, horizon)
        trajectory = torch.stack(states, dim=1)
        return trajectory
    
    def get_state_probs(
        self, 
        x_past: torch.Tensor, 
        x_future: torch.Tensor
    ) -> torch.Tensor:
        """
        Get posterior state probabilities.
        
        Args:
            x_past: Past input (batch, past_seq_len, input_dim)
            x_future: Future observations (batch, horizon, 1)
        
        Returns:
            State probabilities (batch, horizon, n_states)
        """
        return self.forward(x_past, x_future, mode='posterior')
    
    def get_prior_components(
        self,
        x_past: torch.Tensor,
        x_future: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prior initial state and transition matrices.
        
        Args:
            x_past: Past input (batch, past_seq_len, input_dim)
            x_future: Future emission predictions (batch, horizon, n_states)
        
        Returns:
            Tuple of (initial_probs, transition_matrices)
        """
        return self.forward(x_past, x_future, mode='prior')


