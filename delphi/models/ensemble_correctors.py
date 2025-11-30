"""
Deep Ensemble of Specialized RNN Correctors for DELPHI.
M=5 ensemble members with specialized roles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
import numpy as np
import warnings


class EnsembleMember(nn.Module):
    """
    Base ensemble member: LSTM-based corrector network.
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
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            output_dim: Output forecast horizon
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(
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
        out, _ = self.lstm(x)
        # Use last timestep
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        correction = self.fc(last_hidden)
        return correction


class TrendCorrectorRNN(EnsembleMember):
    """
    Trend-Corrector RNN: Specialized for long-term trend deviations.
    Optionally uses xLSTMTime architecture for enhanced long-term dependencies.
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_dim: int = 26,
        dropout: float = 0.2,
        use_xlstm: bool = False
    ):
        super().__init__(input_dim, hidden_size, num_layers, output_dim, dropout)
        self.use_xlstm = use_xlstm
        
        if use_xlstm:
            # Use xLSTMTime model for enhanced long-term dependencies
            try:
                from .xlstm_time import xLSTMTimeModel
                # Replace LSTM with xLSTMTime
                self.xlstm_model = xLSTMTimeModel(
                    input_size=hidden_size,
                    forecast_horizon=output_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout
                )
                # Keep original LSTM as fallback
                self.use_xlstm_model = True
            except ImportError:
                warnings.warn("xLSTMTime not available, using standard LSTM")
                self.use_xlstm_model = False
        else:
            self.use_xlstm_model = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with trend-focused processing."""
        if self.use_xlstm and hasattr(self, 'use_xlstm_model') and self.use_xlstm_model:
            # Use xLSTMTime model
            return self.xlstm_model(x)
        else:
            # Use standard LSTM
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
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        
        # Combine trend and volatility corrections
        trend_correction = self.fc(last_hidden)
        volatility_correction = self.volatility_fc(last_hidden)
        
        return trend_correction + 0.5 * volatility_correction


class ExternalSignalSpecialist(EnsembleMember):
    """
    External Signal Specialist: Integrates weak signals (influencer behavior)
    when predictive. Conditionally activates based on signal strength.
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_dim: int = 26,
        dropout: float = 0.2,
        signal_threshold: float = 0.3
    ):
        super().__init__(input_dim, hidden_size, num_layers, output_dim, dropout)
        self.signal_threshold = signal_threshold
        # Additional layer for weak signal processing
        self.signal_fc = nn.Linear(hidden_size + 1, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with selective weak signal integration.
        
        Args:
            x: Input tensor where x[:, :, 1] is weak signal ratio
        
        Returns:
            Correction forecast
        """
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        
        # Extract weak signal (assuming it's the second feature)
        weak_signal = x[:, -1, 1]  # Last timestep, weak signal feature
        signal_strength = torch.abs(weak_signal - 0.5)  # Distance from neutral
        
        # Conditionally integrate weak signal
        if signal_strength.mean() > self.signal_threshold:
            # Strong signal: integrate it
            signal_input = torch.cat([last_hidden, weak_signal.unsqueeze(1)], dim=-1)
            correction = self.signal_fc(signal_input)
        else:
            # Weak signal: use standard correction
            correction = self.fc(last_hidden)
        
        return correction


class DeepEnsembleCorrectors(nn.Module):
    """
    Deep Ensemble of M=5 specialized corrector networks.
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_dim: int = 26,
        dropout: float = 0.2,
        n_members: int = 5,
        use_xlstm_for_trend: bool = False
    ):
        """
        Initialize deep ensemble of correctors.
        
        Args:
            input_dim: Input feature dimension
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            output_dim: Output forecast horizon
            dropout: Dropout rate
            n_members: Number of ensemble members (M=5)
            use_xlstm_for_trend: Whether to use xLSTMTime for trend corrector
        """
        super().__init__()
        
        self.n_members = n_members
        self.output_dim = output_dim
        
        # Create specialized correctors
        self.correctors = nn.ModuleList([
            TrendCorrectorRNN(input_dim, hidden_size, num_layers, output_dim, dropout, use_xlstm_for_trend),
            SeasonalityCorrectorRNN(input_dim, hidden_size, num_layers, output_dim, dropout),
            VolatilityShiftCorrectorRNN(input_dim, hidden_size, num_layers, output_dim, dropout),
            ExternalSignalSpecialist(input_dim, hidden_size, num_layers, output_dim, dropout),
            EnsembleMember(input_dim, hidden_size, num_layers, output_dim, dropout)  # General corrector
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through ensemble with optional state-based routing.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            states: HMM states for routing (batch, seq_len) or None for ensemble average
        
        Returns:
            Ensemble correction of shape (batch, output_dim)
        """
        batch_size = x.shape[0]
        corrections = []
        
        for corrector in self.correctors:
            correction = corrector(x)
            corrections.append(correction)
        
        # Stack corrections: (n_members, batch, output_dim)
        corrections = torch.stack(corrections, dim=0)
        
        # Route based on HMM states if provided
        if states is not None:
            # Use state to select/weight correctors
            # State 0 -> Trend, State 1 -> Seasonality, State 2 -> Volatility, State 3 -> External
            # State >= 4 -> General corrector
            batch_size, seq_len = states.shape
            state_weights = F.one_hot(states[:, -1], num_classes=self.n_members).float()
            state_weights = state_weights.unsqueeze(-1)  # (batch, n_members, 1)
            
            # Weighted combination
            weighted_corrections = (corrections.permute(1, 0, 2) * state_weights).sum(dim=1)
            return weighted_corrections
        else:
            # Simple ensemble average
            return corrections.mean(dim=0)
    
    def forward_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty quantification via ensemble variance.
        
        Args:
            x: Input tensor
            num_samples: Number of samples for uncertainty
        
        Returns:
            Tuple of (mean_correction, std_correction)
        """
        # Get predictions from all members
        corrections = []
        for corrector in self.correctors:
            correction = corrector(x)
            corrections.append(correction)
        
        corrections = torch.stack(corrections, dim=0)  # (n_members, batch, output_dim)
        
        # Compute mean and std
        mean_correction = corrections.mean(dim=0)
        std_correction = corrections.std(dim=0)
        
        return mean_correction, std_correction

