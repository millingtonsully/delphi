"""
xLSTMTime model implementation for long-term time series forecasting.
Extended LSTM with exponential gating for enhanced long-term dependencies.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class xLSTMTimeModel(nn.Module):
    """
    xLSTMTime: Extended LSTM for time series forecasting.
    
    Implements exponential gating and memory mechanisms for better
    long-term dependency modeling.
    """
    
    def __init__(
        self,
        input_size: int = 52,
        forecast_horizon: int = 26,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize xLSTMTime model.
        
        Args:
            input_size: Input sequence length
            forecast_horizon: Forecast horizon
            hidden_size: Hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            device: Device for computation
        """
        super().__init__()
        self.input_size = input_size
        self.forecast_horizon = forecast_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        # Enhanced LSTM with exponential gating
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Exponential gating mechanism
        self.gate_linear = nn.Linear(hidden_size, hidden_size)
        self.gate_activation = nn.Sigmoid()
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len) or (batch_size, seq_len, 1)
        
        Returns:
            Forecast tensor of shape (batch_size, forecast_horizon)
        """
        # Ensure input is 3D: (batch_size, seq_len, features)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch_size, hidden_size)
        
        # Apply exponential gating
        gate = self.gate_activation(self.gate_linear(last_hidden))
        gated_hidden = last_hidden * torch.exp(gate)
        
        # Project to forecast horizon
        forecast = self.output_projection(gated_hidden)
        
        return forecast
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict method for numpy input.
        
        Args:
            x: Input array of shape (seq_len,) or (batch_size, seq_len)
        
        Returns:
            Forecast array of shape (forecast_horizon,) or (batch_size, forecast_horizon)
        """
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            x = x.to(self.device)
            forecast = self.forward(x)
            return forecast.cpu().numpy()


