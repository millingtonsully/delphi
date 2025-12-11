"""
Regime Explainer: State-specific explanations for each of the 4 correctors.

Provides explanations for:
- Trend corrector (State 0)
- Seasonality corrector (State 1)
- Volatility corrector (State 2)
- External Signal specialist (State 3)
"""

import torch
import numpy as np
from typing import Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class RegimeExplainer:
    """
    Explainer for regime-specific behavior of DELPHI correctors.
    
    Maps HMM states to specialized correctors and explains:
    - Why certain states activate
    - How each corrector contributes to forecast
    - Regime transition patterns
    """
    
    def __init__(self):
        """Initialize regime explainer."""
        self.state_to_corrector = {
            0: 'Trend',
            1: 'Seasonality',
            2: 'Volatility',
            3: 'External Signal'
        }
        self.corrector_names = ['Trend', 'Seasonality', 'Volatility', 'External Signal']
    
    def explain_state_activation(
        self,
        model,
        x: torch.Tensor,
        states: torch.Tensor
    ) -> Dict:
        """
        Explain why certain states activate at specific timesteps.
        
        Args:
            model: DELPHICore model instance
            x: Input tensor (batch, seq_len, input_dim)
            states: HMM states (batch, horizon)
        
        Returns:
            Dictionary with state activation explanations
        """
        model.eval()
        with torch.no_grad():
            batch_size, horizon = states.shape
            
            # Get state probabilities for context
            results = model(x, return_states=True)
            state_probs = results['state_probs']  # (batch, horizon, n_states)
            
            # Analyze state activations
            activation_counts = torch.zeros(model.n_states, device=x.device)
            state_durations = []
            
            for b in range(batch_size):
                prev_state = None
                duration = 0
                state_start = 0  # Track when current state started
                
                for t in range(horizon):
                    curr_state = int(states[b, t].item())  # Ensure integer
                    activation_counts[curr_state] += 1
                    
                    if prev_state is None:
                        # First timestep: initialize
                        prev_state = curr_state
                        duration = 1
                        state_start = t
                    elif prev_state == curr_state:
                        # Same state: increment duration
                        duration += 1
                    else:
                        # State changed: record previous state duration
                        state_durations.append({
                            'state': prev_state,
                            'duration': duration,
                            'timestep_start': state_start,
                            'timestep_end': t - 1,
                            'batch_idx': b
                        })
                        # Start tracking new state
                        prev_state = curr_state
                        duration = 1
                        state_start = t
                
                # Record final state duration (always exists since horizon > 0)
                if prev_state is not None:
                    state_durations.append({
                        'state': prev_state,
                        'duration': duration,
                        'timestep_start': state_start,
                        'timestep_end': horizon - 1,
                        'batch_idx': b
                    })
            
            # Compute activation statistics
            activation_stats = {}
            for state_id, corrector_name in self.state_to_corrector.items():
                count = activation_counts[state_id].item()
                percentage = (count / (batch_size * horizon)) * 100
                activation_stats[corrector_name] = {
                    'count': int(count),
                    'percentage': float(percentage),
                    'state_id': state_id
                }
            
            return {
                'activation_counts': activation_counts.cpu().numpy(),
                'activation_stats': activation_stats,
                'state_durations': state_durations,
                'state_probs': state_probs.cpu().numpy(),
                'corrector_names': self.corrector_names
            }
    
    def corrector_contributions(
        self,
        model,
        x: torch.Tensor,
        states: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Quantify how each corrector contributes to final forecast.
        
        Args:
            model: DELPHICore model instance
            x: Input tensor (batch, seq_len, input_dim)
            states: HMM states (batch, horizon)
            parametric_forecast: Parametric baseline forecast (batch, output_dim)
        
        Returns:
            Dictionary with corrector contributions
        """
        model.eval()
        with torch.no_grad():
            # Get corrections from each corrector
            all_corrections = []
            for corrector in model.ensemble.correctors:
                mu, _ = corrector(x)  # (batch, horizon)
                all_corrections.append(mu)
            
            all_corrections = torch.stack(all_corrections, dim=0)  # (n_states, batch, horizon)
            
            # Get actual correction used (based on states)
            batch_size, horizon = states.shape
            actual_corrections = torch.zeros(batch_size, horizon, device=x.device)
            
            for b in range(batch_size):
                for t in range(horizon):
                    state = states[b, t].item()
                    actual_corrections[b, t] = all_corrections[state, b, t]
            
            # Compute contribution per corrector
            corrector_contributions = torch.zeros(model.n_states, device=x.device)
            for state_id in range(model.n_states):
                mask = (states == state_id)
                if mask.any():
                    contributions = all_corrections[state_id][mask]
                    corrector_contributions[state_id] = torch.abs(contributions).mean()
            
            # Normalize contributions
            total_contribution = corrector_contributions.sum()
            if total_contribution > 0:
                corrector_contributions = corrector_contributions / total_contribution
            
            # Get final forecast for comparison
            results = model(x, parametric_forecast=parametric_forecast, return_states=True)
            final_forecast = results['forecast']
            correction = results['correction']
            
            # Compute contribution breakdown
            contribution_breakdown = {}
            for state_id, corrector_name in self.state_to_corrector.items():
                contribution_breakdown[corrector_name] = {
                    'mean_contribution': float(corrector_contributions[state_id].item()),
                    'state_id': state_id
                }
            
            return {
                'corrector_contributions': corrector_contributions.cpu().numpy(),
                'contribution_breakdown': contribution_breakdown,
                'actual_corrections': actual_corrections.cpu().numpy(),
                'all_corrections': all_corrections.cpu().numpy(),
                'final_forecast': final_forecast.cpu().numpy(),
                'correction': correction.cpu().numpy(),
                'corrector_names': self.corrector_names
            }
    
    def regime_transition_explanation(
        self,
        state_probs: torch.Tensor,
        transition_matrices: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Explain regime transitions using state probabilities and transition matrices.
        
        Args:
            state_probs: State probabilities (batch, horizon, n_states)
            transition_matrices: Transition matrices (batch, horizon-1, n_states, n_states) or None
        
        Returns:
            Dictionary with transition explanations
        """
        batch_size, horizon, n_states = state_probs.shape
        
        # Get dominant states
        dominant_states = torch.argmax(state_probs, dim=-1)  # (batch, horizon)
        
        # Count transitions
        transition_counts = torch.zeros(n_states, n_states, device=state_probs.device)
        
        for b in range(batch_size):
            for t in range(horizon - 1):
                from_state = dominant_states[b, t].item()
                to_state = dominant_states[b, t+1].item()
                transition_counts[from_state, to_state] += 1
        
        # Normalize transition counts
        transition_probs = transition_counts / (transition_counts.sum() + 1e-8)
        
        # Analyze transition patterns
        transition_patterns = []
        for from_state in range(n_states):
            for to_state in range(n_states):
                if transition_probs[from_state, to_state] > 0.01:  # Threshold
                    transition_patterns.append({
                        'from': self.state_to_corrector[from_state],
                        'to': self.state_to_corrector[to_state],
                        'probability': float(transition_probs[from_state, to_state].item()),
                        'count': int(transition_counts[from_state, to_state].item())
                    })
        
        # Sort by probability
        transition_patterns.sort(key=lambda x: x['probability'], reverse=True)
        
        # Compute transition uncertainty (entropy)
        transition_entropy = torch.zeros(horizon - 1, device=state_probs.device)
        if transition_matrices is not None:
            for t in range(horizon - 1):
                trans_matrix = transition_matrices[:, t, :, :].mean(dim=0)  # Average over batch
                # Compute entropy of transition probabilities
                probs = trans_matrix.flatten()
                probs = probs[probs > 1e-8]  # Remove zeros
                entropy = -(probs * torch.log(probs)).sum()
                transition_entropy[t] = entropy
        
        return {
            'transition_counts': transition_counts.cpu().numpy(),
            'transition_probabilities': transition_probs.cpu().numpy(),
            'transition_patterns': transition_patterns,
            'transition_entropy': transition_entropy.cpu().numpy(),
            'dominant_states': dominant_states.cpu().numpy(),
            'corrector_names': self.corrector_names
        }

