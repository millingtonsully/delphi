"""
Deep Ensemble of Specialized RNN Correctors for DELPHI.
M=4 ensemble members with specialized roles matching 4 HMM states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

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
        self.fc = nn.Linear(hidden_size, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble member.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
        
        Returns:
            Correction forecast of shape (batch, output_dim)
        """
        out, _ = self.xlstm(x)
        # Use last timestep
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        correction = self.fc(last_hidden)
        return correction


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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with volatility modeling."""
        out, _ = self.xlstm(x)
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        
        # Combine trend and volatility corrections
        trend_correction = self.fc(last_hidden)
        volatility_correction = self.volatility_fc(last_hidden)
        
        return trend_correction + 0.5 * volatility_correction


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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with weak signal integration.
        
        Args:
            x: Input tensor where x[:, :, 1] is weak signal ratio
        
        Returns:
            Correction forecast
        """
        out, _ = self.xlstm(x)
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        
        # Extract weak signal (second feature) and always integrate
        weak_signal = x[:, -1, 1]  # Last timestep, weak signal feature
        signal_input = torch.cat([last_hidden, weak_signal.unsqueeze(1)], dim=-1)
        correction = self.signal_fc(signal_input)
        
        return correction


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
        
        self.n_members = 4  # Fixed to match 4 HMM states
        self.output_dim = output_dim
        
        # Create 4 specialized correctors (1:1 mapping with HMM states)
        # State 0 -> Trend, State 1 -> Seasonality, State 2 -> Volatility, State 3 -> External
        self.correctors = nn.ModuleList([
            TrendCorrectorRNN(input_dim, hidden_size, num_layers, output_dim, dropout),
            SeasonalityCorrectorRNN(input_dim, hidden_size, num_layers, output_dim, dropout),
            VolatilityShiftCorrectorRNN(input_dim, hidden_size, num_layers, output_dim, dropout),
            ExternalSignalSpecialist(input_dim, hidden_size, num_layers, output_dim, dropout)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through ensemble with state-based routing.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            states: HMM states for routing (batch, seq_len)
        
        Returns:
            Ensemble correction of shape (batch, output_dim)
        """
        # Run all correctors
        corrections = []
        for corrector in self.correctors:
            correction = corrector(x)
            corrections.append(correction)
        
        # Stack corrections: (n_members, batch, output_dim)
        corrections = torch.stack(corrections, dim=0)
        
        # Route based on HMM states
        if states is not None:
            # Use last timestep state to select corrector
            # State 0 -> Trend, State 1 -> Seasonality, State 2 -> Volatility, State 3 -> External
            final_states = states[:, -1]  # (batch,)
            state_weights = F.one_hot(final_states, num_classes=self.n_members).float()
            state_weights = state_weights.unsqueeze(-1)  # (batch, n_members, 1)
            
            # Weighted combination based on state
            weighted_corrections = (corrections.permute(1, 0, 2) * state_weights).sum(dim=1)
            return weighted_corrections
        else:
            # Ensemble average if no states provided
            return corrections.mean(dim=0)

