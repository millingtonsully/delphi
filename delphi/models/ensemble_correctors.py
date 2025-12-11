"""
Deep Ensemble of Specialized RNN Correctors for DELPHI.
M=4 ensemble members with specialized roles matching 4 HMM states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .xlstm_layer import xLSTM


class EnsembleMember(nn.Module):
    """
    Base ensemble member: xLSTM-based corrector network.
    Uses Extended LSTM with exponential gating for enhanced long-term memory.
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_dim: int = 26,  # Forecast horizon
        dropout: float = 0.2
    ):
        """
        Initialize ensemble member.
        
        Args:
            input_dim: Input feature dimension
            hidden_size: xLSTM hidden size
            num_layers: Number of xLSTM layers
            output_dim: Output forecast horizon
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        
        self.xlstm = xLSTM(
            input_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc_mu = nn.Linear(hidden_size, output_dim)
        self.fc_sigma = nn.Linear(hidden_size, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ensemble member.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
        
        Returns:
            Tuple of (mu, sigma):
                - mu: Mean forecast of shape (batch, output_dim)
                - sigma: Standard deviation of shape (batch, output_dim)
        """
        out, _ = self.xlstm(x)
        # Use last timestep
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        mu = self.fc_mu(last_hidden)
        sigma_logits = self.fc_sigma(last_hidden)
        # Apply softplus to ensure sigma > 0, add small epsilon for numerical stability
        sigma = F.softplus(sigma_logits) + 1e-6
        return mu, sigma


class TrendCorrectorRNN(EnsembleMember):
    """
    Trend-Corrector RNN: Specialized for long-term trend deviations.
    Uses xLSTM with exponential gating for enhanced trend capture.
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_dim: int = 26,
        dropout: float = 0.2
    ):
        super().__init__(input_dim, hidden_size, num_layers, output_dim, dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with trend-focused processing."""
        return super().forward(x)


class SeasonalityCorrectorRNN(EnsembleMember):
    """
    Seasonality-Corrector RNN: Specialized for seasonal pattern corrections.
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_dim: int = 26,
        dropout: float = 0.2
    ):
        super().__init__(input_dim, hidden_size, num_layers, output_dim, dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with seasonality-focused processing."""
        return super().forward(x)


class VolatilityShiftCorrectorRNN(EnsembleMember):
    """
    Volatility/Shift-Corrector RNN: Adapts to regime shifts and volatility changes.
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_dim: int = 26,
        dropout: float = 0.2
    ):
        super().__init__(input_dim, hidden_size, num_layers, output_dim, dropout)
        # Additional layer for volatility modeling
        self.volatility_fc = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with volatility modeling."""
        out, _ = self.xlstm(x)
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        
        # Combine trend and volatility corrections for mu
        trend_correction = self.fc_mu(last_hidden)
        volatility_correction = self.volatility_fc(last_hidden)
        mu = trend_correction + 0.5 * volatility_correction
        
        # Sigma from base class
        sigma_logits = self.fc_sigma(last_hidden)
        sigma = F.softplus(sigma_logits) + 1e-6
        
        return mu, sigma


class ExternalSignalSpecialist(EnsembleMember):
    """
    External Signal Specialist: Always integrates weak signals (influencer behavior).
    Specialized for detecting early trend signals from fashion-forward users.
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_dim: int = 26,
        dropout: float = 0.2
    ):
        super().__init__(input_dim, hidden_size, num_layers, output_dim, dropout)
        # Layer for weak signal integration (always used)
        self.signal_fc = nn.Linear(hidden_size + 1, output_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with weak signal integration.
        
        Args:
            x: Input tensor where x[:, :, 1] is weak signal ratio
        
        Returns:
            Tuple of (mu, sigma):
                - mu: Mean forecast with weak signal integration
                - sigma: Standard deviation
        """
        out, _ = self.xlstm(x)
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        
        # Extract weak signal (second feature) and always integrate
        weak_signal = x[:, -1, 1]  # Last timestep, weak signal feature
        signal_input = torch.cat([last_hidden, weak_signal.unsqueeze(1)], dim=-1)
        mu = self.signal_fc(signal_input)
        
        # Sigma from base class (doesn't use weak signal)
        sigma_logits = self.fc_sigma(last_hidden)
        sigma = F.softplus(sigma_logits) + 1e-6
        
        return mu, sigma


class DeepEnsembleCorrectors(nn.Module):
    """
    Deep Ensemble of M=4 specialized xLSTM corrector networks.
    Each corrector maps 1:1 with HMM states for regime-specific corrections.
    All correctors use xLSTM with exponential gating for enhanced memory.
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_dim: int = 26,
        dropout: float = 0.2,
        n_members: int = 4
    ):
        """
        Initialize deep ensemble of correctors.
        
        Args:
            input_dim: Input feature dimension
            hidden_size: xLSTM hidden size
            num_layers: Number of xLSTM layers
            output_dim: Output forecast horizon
            dropout: Dropout rate
            n_members: Number of ensemble members (M=4, must match HMM states)
        """
        super().__init__()
        
        # Validate that n_members equals 4 (fixed to match 4 specialized correctors)
        assert n_members == 4, f"n_members must be 4 (got {n_members}). This is fixed to match the 4 specialized correctors."
        self.n_members = n_members
        self.output_dim = output_dim
        
        # Create 4 specialized correctors (1:1 mapping with HMM states)
        # State 0 -> Trend, State 1 -> Seasonality, State 2 -> Volatility, State 3 -> External
        self.correctors = nn.ModuleList([
            TrendCorrectorRNN(input_dim, hidden_size, num_layers, output_dim, dropout),
            SeasonalityCorrectorRNN(input_dim, hidden_size, num_layers, output_dim, dropout),
            VolatilityShiftCorrectorRNN(input_dim, hidden_size, num_layers, output_dim, dropout),
            ExternalSignalSpecialist(input_dim, hidden_size, num_layers, output_dim, dropout)
        ])
        
        # Validate that number of correctors matches n_members
        assert len(self.correctors) == self.n_members, \
            f"Number of correctors ({len(self.correctors)}) must match n_members ({self.n_members})"
    
    def forward(
        self,
        x: torch.Tensor,
        states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ensemble with state-based routing.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            states: HMM states for routing
                - If (batch, horizon): per-timestep states for forecast horizon
                - If (batch, seq_len): legacy format, uses last state
                - If None: ensemble average
        
        Returns:
            Tuple of (correction_mu, correction_sigma):
                - correction_mu: Mean correction of shape (batch, output_dim)
                - correction_sigma: Standard deviation of shape (batch, output_dim)
        """
        # Run all correctors to get emission predictions (mu, sigma)
        emission_mus = []
        emission_sigmas = []
        for corrector in self.correctors:
            mu, sigma = corrector(x)
            emission_mus.append(mu)
            emission_sigmas.append(sigma)
        
        # Stack: (n_members, batch, output_dim)
        emission_mu = torch.stack(emission_mus, dim=0)  # (n_members, batch, output_dim)
        emission_sigma = torch.stack(emission_sigmas, dim=0)  # (n_members, batch, output_dim)
        
        # Route based on HMM states
        if states is not None:
            # Check if states are per-timestep (horizon) or legacy (seq_len)
            if states.shape[1] == self.output_dim:
                # Per-timestep states: (batch, horizon)
                # For each timestep t, use emission law z_t
                batch_size = states.shape[0]
                horizon = states.shape[1]
                
                # For each timestep, select the corrector based on state
                batch_indices = torch.arange(batch_size, device=states.device).unsqueeze(1).expand(-1, horizon)
                timestep_indices = torch.arange(horizon, device=states.device).unsqueeze(0).expand(batch_size, -1)
                
                # Select mu and sigma for each (batch, timestep) pair based on state
                selected_mu = emission_mu[
                    states,  # (batch, horizon) - which corrector to use
                    batch_indices,  # (batch, horizon) - batch index
                    timestep_indices  # (batch, horizon) - timestep index
                ]  # (batch, horizon)
                
                selected_sigma = emission_sigma[
                    states,
                    batch_indices,
                    timestep_indices
                ]  # (batch, horizon)
                
                return selected_mu, selected_sigma
            else:
                # Legacy format: use last timestep state for all timesteps
                final_states = states[:, -1]  # (batch,)
                state_weights = F.one_hot(final_states, num_classes=self.n_members).float()
                state_weights = state_weights.unsqueeze(-1)  # (batch, n_members, 1)
                
                # Weighted combination based on state
                weighted_mu = (emission_mu.permute(1, 0, 2) * state_weights).sum(dim=1)
                weighted_sigma = (emission_sigma.permute(1, 0, 2) * state_weights).sum(dim=1)
                return weighted_mu, weighted_sigma
        else:
            # Ensemble average if no states provided
            return emission_mu.mean(dim=0), emission_sigma.mean(dim=0)

