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
    
    Models temporal regime persistence via Markov chains with variational
    posterior for tractable inference. Supports two-stage ELBO training.
    Uses xLSTM with exponential gating for enhanced long-term memory.
    """
    
    def __init__(
        self,
        input_dim: int = 3,  # [z_t, w_t, ŷ_pred_t]
        n_states: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize Variational HMM Gating module.
        
        Args:
            input_dim: Dimension of input features
            n_states: Number of latent discrete states
            hidden_size: Hidden size for xLSTM layers
            num_layers: Number of xLSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.n_states = n_states
        self.hidden_size = hidden_size
        
        # Initial state xLSTM (for prior)
        self.initial_xlstm = xLSTM(
            input_dim, hidden_size, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc_initial = nn.Linear(hidden_size, n_states)
        
        # Transition xLSTM (for prior)
        self.transition_xlstm = xLSTM(
            input_dim + n_states, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc_trans = nn.Linear(hidden_size, n_states * n_states)
        
        # Posterior xLSTM
        self.posterior_xlstm = xLSTM(
            input_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc_posterior = nn.Linear(hidden_size, n_states)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mode: str = 'posterior'
    ) -> torch.Tensor:
        """
        Forward pass through HMM gating module.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            mode: 'posterior' for variational posterior, 'prior' for prior sampling
        
        Returns:
            State probabilities or sampled states
        """
        batch_size, seq_len, _ = x.shape
        
        if mode == 'posterior':
            # Variational posterior: q(z|x)
            out, _ = self.posterior_xlstm(x)
            out = self.dropout(out)
            # Average over sequence for global posterior
            logits = self.fc_posterior(out.mean(dim=1))
            probs = F.softmax(logits, dim=-1)
            return probs
        
        elif mode == 'prior':
            # Prior sampling: p(z|x) via Markov chain
            trajectories = []
            
            # Initial state
            init_out, _ = self.initial_xlstm(x[:, :1, :])
            init_logits = self.fc_initial(self.dropout(init_out.squeeze(1)))
            init_probs = F.softmax(init_logits, dim=-1)
            
            # Sample initial state
            states = torch.multinomial(init_probs, 1)  # (batch, 1)
            trajectories.append(states)
            
            # Transition states
            for t in range(1, seq_len):
                # One-hot encode previous state
                prev_onehot = F.one_hot(states.squeeze(1), self.n_states).float()
                
                # Concatenate input and previous state
                trans_input = torch.cat([
                    x[:, t:t+1, :], 
                    prev_onehot.unsqueeze(1)
                ], dim=-1)
                
                # Get transition probabilities
                trans_out, _ = self.transition_xlstm(trans_input)
                trans_logits = self.fc_trans(self.dropout(trans_out.squeeze(1)))
                trans_matrix = F.softmax(
                    trans_logits.view(-1, self.n_states, self.n_states), 
                    dim=-1
                )
                
                # Compute next state probabilities
                # p(z_t | z_{t-1}, x_t) = sum_{z_{t-1}} p(z_t | z_{t-1}) * p(z_{t-1} | x_{:t})
                prev_probs = init_probs.unsqueeze(1)  # (batch, 1, n_states)
                next_probs = torch.bmm(prev_probs, trans_matrix).squeeze(1)
                
                # Sample next state
                states = torch.multinomial(next_probs, 1)
                trajectories.append(states)
            
            # Stack trajectories
            trajectory = torch.cat(trajectories, dim=1)  # (batch, seq_len)
            return trajectory
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'posterior' or 'prior'")
    
    def get_state_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get posterior state probabilities."""
        return self.forward(x, mode='posterior')
    
    def get_transition_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get learned transition matrix from input.
        
        Args:
            x: Input tensor
        
        Returns:
            Transition matrix of shape (batch, n_states, n_states)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Use first timestep for initial state
        init_out, _ = self.initial_xlstm(x[:, :1, :])
        init_probs = F.softmax(self.fc_initial(self.dropout(init_out.squeeze(1))), dim=-1)
        
        # Get transition matrix from middle timestep
        mid_idx = seq_len // 2
        prev_onehot = F.one_hot(torch.argmax(init_probs, dim=-1), self.n_states).float()
        trans_input = torch.cat([x[:, mid_idx:mid_idx+1, :], prev_onehot.unsqueeze(1)], dim=-1)
        trans_out, _ = self.transition_xlstm(trans_input)
        trans_logits = self.fc_trans(self.dropout(trans_out.squeeze(1)))
        trans_matrix = F.softmax(trans_logits.view(-1, self.n_states, self.n_states), dim=-1)
        
        return trans_matrix


