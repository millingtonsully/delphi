"""
xLSTM Layer - LSTM wrapper with compatible API.

Originally implemented Extended LSTM with exponential gating, but now uses
cuDNN-optimized nn.LSTM for dramatically faster training (~10-50x speedup).

The xLSTM class maintains the same API for drop-in compatibility.
To restore original xLSTM behavior, see xLSTM_Original class below.

Reference: xLSTMTime (https://github.com/muslehal/xLSTMTime)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class xLSTM(nn.Module):
    """
    LSTM wrapper with xLSTM-compatible API.
    
    Uses cuDNN-optimized nn.LSTM for fast training while maintaining
    the same interface as the original xLSTM implementation.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units
        num_layers: Number of stacked LSTM layers
        batch_first: If True, input/output tensors are (batch, seq, feature)
        dropout: Dropout probability between layers (applied if num_layers > 1)
        bidirectional: Whether to use bidirectional LSTM
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # Use cuDNN-optimized LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Layer normalization on output for stability (similar to xLSTM)
        self.layer_norm = nn.LayerNorm(hidden_size * (2 if bidirectional else 1))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights similar to xLSTM for consistent behavior."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better gradient flow
                hidden_size = self.hidden_size
                param.data[hidden_size:2*hidden_size].fill_(1.0)
    
    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LSTM.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size) if batch_first
               else (seq_len, batch, input_size)
            hx: Optional tuple of (h_0, c_0) with shape (num_layers, batch, hidden_size)
        
        Returns:
            output: Hidden states for each timestep
                    Shape: (batch, seq_len, hidden_size) if batch_first
            (h_n, c_n): Final hidden and cell states
                        Shape: (num_layers, batch, hidden_size)
        """
        output, (h_n, c_n) = self.lstm(x, hx)
        
        # Apply layer normalization for stability
        output = self.layer_norm(output)
        
        return output, (h_n, c_n)


# =============================================================================
# Original xLSTM implementation (kept for reference/future use)
# =============================================================================

class xLSTMCell_Original(nn.Module):
    """
    Extended LSTM Cell with exponential gating.
    
    Implements the sLSTM variant with exponential gates for better
    gradient flow and memory retention in long sequences.
    
    NOTE: This is slower than cuDNN LSTM. Use xLSTM class above for training.
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Combined gate computation for efficiency
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        
        # Exponential gate scaling parameters (learnable)
        self.exp_forget_scale = nn.Parameter(torch.ones(hidden_size))
        self.exp_input_scale = nn.Parameter(torch.ones(hidden_size))
        
        # Layer normalization for stability
        self.ln_cell = nn.LayerNorm(hidden_size)
        self.ln_hidden = nn.LayerNorm(hidden_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                if 'gates' in name:
                    param.data[self.hidden_size:2*self.hidden_size].fill_(1.0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = x.size(0)
        
        if hx is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h, c = hx
        
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, g, o = gates.chunk(4, dim=1)
        
        # Exponential gating
        i_exp = torch.exp(torch.clamp(i * self.exp_input_scale, -10, 10))
        f_exp = torch.exp(torch.clamp(f * self.exp_forget_scale, -10, 10))
        
        gate_sum = i_exp + f_exp + 1e-8
        i_norm = i_exp / gate_sum
        f_norm = f_exp / gate_sum
        
        g_tanh = torch.tanh(g)
        c_new = f_norm * c + i_norm * g_tanh
        c_new = self.ln_cell(c_new)
        
        o_sig = torch.sigmoid(o)
        h_new = o_sig * torch.tanh(c_new)
        h_new = self.ln_hidden(h_new)
        
        return h_new, (h_new, c_new)


class xLSTM_Original(nn.Module):
    """
    Original xLSTM with exponential gating - SLOW but potentially more accurate.
    
    Use this class if you need the original xLSTM behavior.
    For fast training, use xLSTM class instead.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False
    ):
        super().__init__()
        
        if bidirectional:
            raise NotImplementedError("Bidirectional xLSTM not yet supported")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.cells.append(xLSTMCell_Original(layer_input_size, hidden_size))
        
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 and num_layers > 1 else None
    
    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not self.batch_first:
            x = x.transpose(0, 1)
        
        batch_size, seq_len, _ = x.shape
        
        if hx is None:
            h_list = [None] * self.num_layers
            c_list = [None] * self.num_layers
        else:
            h_0, c_0 = hx
            h_list = [h_0[i] for i in range(self.num_layers)]
            c_list = [c_0[i] for i in range(self.num_layers)]
        
        # Process sequence
        current_input = x
        h_n_list = []
        c_n_list = []
        
        for layer_idx, cell in enumerate(self.cells):
            h = h_list[layer_idx]
            c = c_list[layer_idx]
            
            if h is None:
                h = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
                c = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            
            outputs = []
            for t in range(seq_len):
                h, (h, c) = cell(current_input[:, t, :], (h, c))
                outputs.append(h)
            
            current_input = torch.stack(outputs, dim=1)
            h_n_list.append(h)
            c_n_list.append(c)
            
            if self.dropout_layer is not None and layer_idx < self.num_layers - 1:
                current_input = self.dropout_layer(current_input)
        
        h_n = torch.stack(h_n_list, dim=0)
        c_n = torch.stack(c_n_list, dim=0)
        
        if not self.batch_first:
            current_input = current_input.transpose(0, 1)
        
        return current_input, (h_n, c_n)

