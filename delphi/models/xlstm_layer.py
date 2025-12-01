"""
xLSTM Layer - Extended LSTM with exponential gating and enhanced memory.

Provides a drop-in replacement for nn.LSTM with xLSTM architecture features:
- Exponential gating for improved gradient flow
- Enhanced memory structure for long-term dependencies
- Stabilized hidden state updates

Reference: xLSTMTime (https://github.com/muslehal/xLSTMTime)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class xLSTMCell(nn.Module):
    """
    Extended LSTM Cell with exponential gating.
    
    Implements the sLSTM variant with exponential gates for better
    gradient flow and memory retention in long sequences.
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
                # Only apply xavier to 2D+ tensors (skip LayerNorm weights which are 1D)
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Forget gate bias initialization for better gradient flow
                if 'gates' in name:
                    param.data[self.hidden_size:2*self.hidden_size].fill_(1.0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through xLSTM cell.
        
        Args:
            x: Input tensor of shape (batch, input_size)
            hx: Tuple of (h, c) hidden states, each (batch, hidden_size)
        
        Returns:
            h_new: New hidden state (batch, hidden_size)
            (h_new, c_new): Tuple of new hidden and cell states
        """
        batch_size = x.size(0)
        
        if hx is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h, c = hx
        
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)
        
        # Compute all gates at once
        gates = self.gates(combined)
        
        # Split into individual gates
        i, f, g, o = gates.chunk(4, dim=1)
        
        # Exponential gating (key xLSTM feature)
        # Use softplus for numerical stability instead of raw exp
        i_exp = torch.exp(torch.clamp(i * self.exp_input_scale, -10, 10))
        f_exp = torch.exp(torch.clamp(f * self.exp_forget_scale, -10, 10))
        
        # Normalize exponential gates
        gate_sum = i_exp + f_exp + 1e-8
        i_norm = i_exp / gate_sum
        f_norm = f_exp / gate_sum
        
        # Cell state update with exponential gating
        g_tanh = torch.tanh(g)
        c_new = f_norm * c + i_norm * g_tanh
        c_new = self.ln_cell(c_new)
        
        # Output gate and hidden state
        o_sig = torch.sigmoid(o)
        h_new = o_sig * torch.tanh(c_new)
        h_new = self.ln_hidden(h_new)
        
        return h_new, (h_new, c_new)


class xLSTMLayer(nn.Module):
    """
    Single xLSTM layer processing a sequence.
    """
    
    def __init__(self, input_size: int, hidden_size: int, batch_first: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        
        self.cell = xLSTMCell(input_size, hidden_size)
    
    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process sequence through xLSTM layer.
        
        Args:
            x: Input tensor (batch, seq_len, input_size) if batch_first else (seq_len, batch, input_size)
            hx: Initial hidden state tuple
        
        Returns:
            output: All hidden states (batch, seq_len, hidden_size)
            (h_n, c_n): Final hidden and cell states
        """
        if self.batch_first:
            batch_size, seq_len, _ = x.shape
        else:
            seq_len, batch_size, _ = x.shape
            x = x.transpose(0, 1)  # Convert to batch_first for processing
        
        # Initialize hidden states
        if hx is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h, c = hx
            # Handle stacked layers format (num_layers, batch, hidden)
            if h.dim() == 3:
                h = h[0]
                c = c[0]
        
        outputs = []
        for t in range(seq_len):
            h, (h, c) = self.cell(x[:, t, :], (h, c))
            outputs.append(h)
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden)
        
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        # Return states in format compatible with nn.LSTM
        h_n = h.unsqueeze(0)  # (1, batch, hidden)
        c_n = c.unsqueeze(0)  # (1, batch, hidden)
        
        return output, (h_n, c_n)


class xLSTM(nn.Module):
    """
    Multi-layer xLSTM - Drop-in replacement for nn.LSTM.
    
    Implements Extended LSTM with:
    - Exponential gating for improved long-term memory
    - Layer normalization for training stability
    - Compatible API with torch.nn.LSTM
    
    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units
        num_layers: Number of stacked xLSTM layers
        batch_first: If True, input/output tensors are (batch, seq, feature)
        dropout: Dropout probability between layers (applied if num_layers > 1)
        bidirectional: Not supported (raises error if True)
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
        
        # Build stacked layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.layers.append(xLSTMLayer(layer_input_size, hidden_size, batch_first=True))
        
        # Dropout between layers
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 and num_layers > 1 else None
    
    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through multi-layer xLSTM.
        
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
        # Convert to batch_first for internal processing
        if not self.batch_first:
            x = x.transpose(0, 1)
        
        batch_size = x.size(0)
        
        # Initialize hidden states for all layers
        if hx is None:
            h_list = [None] * self.num_layers
            c_list = [None] * self.num_layers
        else:
            h_0, c_0 = hx
            h_list = [h_0[i] for i in range(self.num_layers)]
            c_list = [c_0[i] for i in range(self.num_layers)]
        
        # Process through each layer
        h_n_list = []
        c_n_list = []
        
        output = x
        for i, layer in enumerate(self.layers):
            if h_list[i] is not None:
                layer_hx = (h_list[i], c_list[i])
            else:
                layer_hx = None
            
            output, (h_n, c_n) = layer(output, layer_hx)
            
            h_n_list.append(h_n.squeeze(0))
            c_n_list.append(c_n.squeeze(0))
            
            # Apply dropout between layers (not after last layer)
            if self.dropout_layer is not None and i < self.num_layers - 1:
                output = self.dropout_layer(output)
        
        # Stack final states: (num_layers, batch, hidden)
        h_n = torch.stack(h_n_list, dim=0)
        c_n = torch.stack(c_n_list, dim=0)
        
        # Convert back if not batch_first
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        return output, (h_n, c_n)

