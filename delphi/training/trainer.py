"""
Training utilities for DELPHI: Two-stage ELBO training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Optional, List
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm

from ..models.delphi_core import DELPHICore
from .losses import ELBOLoss, CombinedLoss


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting."""
    
    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        parametric_forecasts: Optional[np.ndarray] = None
    ):
        """
        Initialize dataset.
        
        Args:
            inputs: Input sequences (n_samples, seq_len, input_dim)
            targets: Target forecasts (n_samples, forecast_horizon)
            parametric_forecasts: Parametric baseline forecasts (n_samples, forecast_horizon)
        """
        self.inputs = torch.FloatTensor(inputs)
        self.targets = torch.FloatTensor(targets)
        if parametric_forecasts is not None:
            self.parametric_forecasts = torch.FloatTensor(parametric_forecasts)
        else:
            self.parametric_forecasts = None
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        item = {
            'input': self.inputs[idx],
            'target': self.targets[idx]
        }
        if self.parametric_forecasts is not None:
            item['parametric_forecast'] = self.parametric_forecasts[idx]
        return item


class DELPHITrainer:
    """
    Trainer for DELPHI model with two-stage ELBO training.
    """
    
    def __init__(
        self,
        model: DELPHICore,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        kl_weight: float = 0.1,
        entropy_weight: float = 0.01,
        stage1_epochs: int = 50,
        stage2_epochs: int = 30,
        kl_anneal: bool = False,
        kl_start: Optional[float] = None,
        kl_end: Optional[float] = None,
        kl_warmup_epochs: int = 0,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 1e-4,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize DELPHI trainer.
        
        Args:
            model: DELPHI core model
            device: Device for training
            learning_rate: Learning rate
            weight_decay: Weight decay
            kl_weight: KL divergence weight
            entropy_weight: Entropy regularization weight
            stage1_epochs: Number of epochs for stage 1 (emissions/posterior)
            stage2_epochs: Number of epochs for stage 2 (prior)
            early_stopping_patience: Number of epochs to wait before stopping if no improvement
            early_stopping_min_delta: Minimum change to qualify as an improvement
            checkpoint_dir: Directory to save checkpoints (for best model)
        """
        self.model = model.to(device)
        self.device = device
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.base_kl_weight = kl_weight

        # KL annealing configuration
        self.kl_anneal = kl_anneal
        self.kl_start = kl_start if kl_start is not None else kl_weight
        self.kl_end = kl_end if kl_end is not None else kl_weight
        self.kl_warmup_epochs = max(0, kl_warmup_epochs)
        
        # Early stopping configuration
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.checkpoint_dir = checkpoint_dir
        
        # Track best validation loss and patience counter (per stage)
        self.best_val_loss_stage1 = float('inf')
        self.best_val_loss_stage2 = float('inf')
        self.patience_counter_stage1 = 0
        self.patience_counter_stage2 = 0
        self.best_epoch_stage1 = 0
        self.best_epoch_stage2 = 0
        
        # Optimizers
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss functions
        self.loss_fn = CombinedLoss(
            recon_weight=1.0,
            kl_weight=kl_weight,
            entropy_weight=entropy_weight
        )
        
        # Training history
        self.history = {
            'stage1': {'train_loss': [], 'val_loss': []},
            'stage2': {'train_loss': [], 'val_loss': []}
        }
        
        # Print GPU status on initialization
        self._print_gpu_status()
    
    def _print_gpu_status(self):
        """Print current GPU memory status and device information."""
        if self.device.startswith('cuda') and torch.cuda.is_available():
            print("\n" + "="*70)
            print("GPU Status")
            print("="*70)
            gpu_id = int(self.device.split(':')[1]) if ':' in self.device else 0
            print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
            print(f"Device: {self.device}")
            
            # Check model is on GPU
            model_device = next(self.model.parameters()).device
            print(f"Model device: {model_device}")
            
            # Memory info
            memory_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
            memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
            print(f"GPU Memory:")
            print(f"  Total: {memory_total:.2f} GB")
            print(f"  Reserved: {memory_reserved:.2f} GB")
            print(f"  Allocated: {memory_allocated:.2f} GB")
            print(f"  Free: {memory_total - memory_reserved:.2f} GB")
            print("="*70 + "\n")
        else:
            print(f"\n⚠️  Using CPU (device: {self.device})")
            if torch.cuda.is_available():
                print("   Note: CUDA is available but not being used.")
                print("   Set device='cuda' to use GPU.\n")
    
    def _get_gpu_memory_info(self) -> str:
        """Get current GPU memory usage as a formatted string."""
        if self.device.startswith('cuda') and torch.cuda.is_available():
            gpu_id = int(self.device.split(':')[1]) if ':' in self.device else 0
            allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
            return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        return ""
    
    def train_stage1(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None
    ):
        """
        Stage 1 training: Train emissions and posterior, fix uniform prior.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (overrides default)
        """
        num_epochs = num_epochs or self.stage1_epochs
        self.model.train()
        
        for epoch in range(num_epochs):
            train_losses = []

            # KL annealing: update KL weight for this epoch
            if self.kl_anneal and self.kl_warmup_epochs > 0:
                # epoch is zero-based; use epoch+1 for human-friendly schedule
                factor = min(1.0, float(epoch + 1) / float(self.kl_warmup_epochs))
                current_kl = self.kl_start + factor * (self.kl_end - self.kl_start)
            else:
                current_kl = self.kl_end
            # Apply to underlying ELBO loss
            self.loss_fn.elbo_loss.kl_weight = current_kl
            
            # Print GPU memory at start of first epoch
            if epoch == 0:
                mem_info = self._get_gpu_memory_info()
                if mem_info:
                    print(f"\n{mem_info}\n")
            
            for batch in tqdm(train_loader, desc=f"Stage 1 Epoch {epoch+1}/{num_epochs}"):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                parametric_forecasts = batch.get('parametric_forecast')
                if parametric_forecasts is not None:
                    parametric_forecasts = parametric_forecasts.to(self.device)
                
                # Verify tensors are on correct device (only check first batch)
                if epoch == 0 and len(train_losses) == 0:
                    if self.device.startswith('cuda'):
                        assert inputs.device.type == 'cuda', f"Inputs on {inputs.device}, expected {self.device}"
                        assert targets.device.type == 'cuda', f"Targets on {targets.device}, expected {self.device}"
                
                # Forward pass with future observations for posterior
                # Reshape targets to (batch, horizon, 1) for posterior
                future_obs = targets.unsqueeze(-1)  # (batch, horizon, 1)
                outputs = self.model(
                    inputs, 
                    parametric_forecast=parametric_forecasts,
                    future_observations=future_obs
                )
                
                # Get posterior probabilities (batch, horizon, n_states)
                posterior_probs = outputs['state_probs']
                
                # Get emission mu and sigma for loss computation
                emission_mu = outputs['emission_mu']  # (n_states, batch, horizon)
                emission_sigma = outputs['emission_sigma']  # (n_states, batch, horizon)
                
                # Compute loss with log-likelihood
                loss_dict = self.loss_fn(
                    emission_mu,
                    emission_sigma,
                    targets,
                    posterior_probs=posterior_probs,
                    stage='stage1'
                )
                
                loss = loss_dict['total_loss']
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            self.history['stage1']['train_loss'].append(avg_train_loss)
            
            # Print GPU memory usage periodically (every 10 epochs)
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                mem_info = self._get_gpu_memory_info()
                if mem_info:
                    print(f"\n  {mem_info}")
            
            # Validation
            if val_loader is not None:
                # Compute metrics every 10 epochs or on last epoch
                compute_metrics = (epoch + 1) % 10 == 0 or epoch == num_epochs - 1
                val_result = self.validate(val_loader, stage='stage1', compute_metrics=compute_metrics)
                
                if compute_metrics and isinstance(val_result, dict):
                    val_loss = val_result['val_loss']
                    self.history['stage1']['val_loss'].append(val_loss)
                    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
                    print(f"  Validation Metrics - MSE: {val_result['val_mse']:.4f}, "
                          f"MAE: {val_result['val_mae']:.4f}")
                else:
                    val_loss = val_result
                    self.history['stage1']['val_loss'].append(val_loss)
                    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
                
                # Early stopping check
                if val_loss < (self.best_val_loss_stage1 - self.early_stopping_min_delta):
                    # Improvement detected
                    self.best_val_loss_stage1 = val_loss
                    self.best_epoch_stage1 = epoch + 1
                    self.patience_counter_stage1 = 0
                    
                    # Save best model checkpoint
                    if self.checkpoint_dir is not None:
                        best_checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model_stage1.pt')
                        self.save_checkpoint(best_checkpoint_path, epoch + 1, is_best=True)
                        print(f"  ✓ Best model saved (Val Loss: {val_loss:.4f})")
                else:
                    # No improvement
                    self.patience_counter_stage1 += 1
                    if self.patience_counter_stage1 >= self.early_stopping_patience:
                        print(f"\nEarly stopping triggered at epoch {epoch+1}")
                        print(f"Best validation loss: {self.best_val_loss_stage1:.4f} at epoch {self.best_epoch_stage1}")
                        print(f"No improvement for {self.early_stopping_patience} epochs.")
                        return
            else:
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")
    
    def train_stage2(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None
    ):
        """
        Stage 2 training: Train prior, freeze posterior/emissions.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (overrides default)
        """
        num_epochs = num_epochs or self.stage2_epochs
        
        # For Stage 2, use the final KL weight (no annealing needed here)
        self.loss_fn.elbo_loss.kl_weight = self.kl_end
        
        # Freeze all parameters except HMM prior
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze HMM prior parameters
        # Prior uses prior_past_xlstm, prior_future_xlstm, prior_fc_initial, prior_fc_transitions
        for param in self.model.hmm_gating.prior_past_xlstm.parameters():
            param.requires_grad = True
        for param in self.model.hmm_gating.prior_future_xlstm.parameters():
            param.requires_grad = True
        for param in self.model.hmm_gating.prior_fc_initial.parameters():
            param.requires_grad = True
        for param in self.model.hmm_gating.prior_fc_transitions.parameters():
            param.requires_grad = True
        
        # Create optimizer for prior only
        prior_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        prior_optimizer = optim.Adam(prior_params, lr=1e-4)
        
        self.model.train()
        
        for epoch in range(num_epochs):
            train_losses = []
            
            # Print GPU memory at start of first epoch
            if epoch == 0:
                mem_info = self._get_gpu_memory_info()
                if mem_info:
                    print(f"\n{mem_info}\n")
            
            for batch in tqdm(train_loader, desc=f"Stage 2 Epoch {epoch+1}/{num_epochs}"):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                parametric_forecasts = batch.get('parametric_forecast')
                if parametric_forecasts is not None:
                    parametric_forecasts = parametric_forecasts.to(self.device)
                
                # Verify tensors are on correct device (only check first batch)
                if epoch == 0 and len(train_losses) == 0:
                    if self.device.startswith('cuda'):
                        assert inputs.device.type == 'cuda', f"Inputs on {inputs.device}, expected {self.device}"
                        assert targets.device.type == 'cuda', f"Targets on {targets.device}, expected {self.device}"
                
                # Forward pass
                with torch.no_grad():
                    # Get posterior probabilities (frozen) using future observations
                    future_obs = targets.unsqueeze(-1)  # (batch, horizon, 1)
                    posterior_probs = self.model.hmm_gating(
                        x_past=inputs,
                        x_future=future_obs,
                        mode='posterior'
                    )  # (batch, horizon, n_states)
                
                # Get prior probabilities (trainable)
                # First compute emission predictions for prior
                all_mus = []
                all_sigmas = []
                for corrector in self.model.ensemble.correctors:
                    mu, sigma = corrector(inputs)
                    all_mus.append(mu)
                    all_sigmas.append(sigma)
                emission_mu = torch.stack(all_mus, dim=0)  # (n_states, batch, horizon)
                emission_sigma = torch.stack(all_sigmas, dim=0)  # (n_states, batch, horizon)
                emission_mu_for_prior = emission_mu.permute(1, 2, 0)  # (batch, horizon, n_states)
                
                # Get prior components
                init_probs, trans_matrices = self.model.hmm_gating.get_prior_components(
                    x_past=inputs,
                    x_future=emission_mu_for_prior
                )
                
                # Compute per-timestep prior probabilities
                # Build as list to avoid in-place operations that break autograd
                batch_size = inputs.shape[0]
                prior_probs_list = [init_probs]
                for t in range(1, self.model.output_dim):
                    prev_probs = prior_probs_list[-1].unsqueeze(1)  # (batch, 1, n_states)
                    trans_matrix = trans_matrices[:, t-1, :, :]  # (batch, n_states, n_states)
                    next_probs = torch.bmm(prev_probs, trans_matrix).squeeze(1)  # (batch, n_states)
                    prior_probs_list.append(next_probs)
                prior_probs = torch.stack(prior_probs_list, dim=1)  # (batch, horizon, n_states)
                
                # Compute log probabilities for KL divergence (per-timestep)
                posterior_logp = torch.log(posterior_probs + 1e-8)  # (batch, horizon, n_states)
                prior_logp = torch.log(prior_probs + 1e-8)  # (batch, horizon, n_states)
                
                # Compute loss (just the KL term for Stage 2)
                # Pass full tensors (not .mean()) - loss function will handle averaging
                loss_dict = self.loss_fn(
                    emission_mu,
                    emission_sigma,
                    targets,
                    prior_logp=prior_logp,
                    posterior_logp=posterior_logp,
                    posterior_probs=posterior_probs,
                    stage='stage2'
                )
                
                # Use the loss from loss function (should be KL term)
                loss = loss_dict['total_loss']
                
                # Backward pass
                prior_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(prior_params, max_norm=1.0)
                prior_optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            self.history['stage2']['train_loss'].append(avg_train_loss)
            
            # Print GPU memory usage periodically (every 10 epochs)
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                mem_info = self._get_gpu_memory_info()
                if mem_info:
                    print(f"\n  {mem_info}")
            
            # Validation
            if val_loader is not None:
                # Compute metrics every 10 epochs or on last epoch
                compute_metrics = (epoch + 1) % 10 == 0 or epoch == num_epochs - 1
                val_result = self.validate(val_loader, stage='stage2', compute_metrics=compute_metrics)
                
                if compute_metrics and isinstance(val_result, dict):
                    val_loss = val_result['val_loss']
                    self.history['stage2']['val_loss'].append(val_loss)
                    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
                    print(f"  Validation Metrics - MSE: {val_result['val_mse']:.4f}, "
                          f"MAE: {val_result['val_mae']:.4f}")
                else:
                    val_loss = val_result
                    self.history['stage2']['val_loss'].append(val_loss)
                    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
                
                # Early stopping check
                if val_loss < (self.best_val_loss_stage2 - self.early_stopping_min_delta):
                    # Improvement detected
                    self.best_val_loss_stage2 = val_loss
                    self.best_epoch_stage2 = epoch + 1
                    self.patience_counter_stage2 = 0
                    
                    # Save best model checkpoint
                    if self.checkpoint_dir is not None:
                        best_checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model_stage2.pt')
                        self.save_checkpoint(best_checkpoint_path, epoch + 1, is_best=True)
                        print(f"  ✓ Best model saved (Val Loss: {val_loss:.4f})")
                else:
                    # No improvement
                    self.patience_counter_stage2 += 1
                    if self.patience_counter_stage2 >= self.early_stopping_patience:
                        print(f"\nEarly stopping triggered at epoch {epoch+1}")
                        print(f"Best validation loss: {self.best_val_loss_stage2:.4f} at epoch {self.best_epoch_stage2}")
                        print(f"No improvement for {self.early_stopping_patience} epochs.")
                        return
            else:
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")
        
        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True
    
    def validate(
        self,
        val_loader: DataLoader,
        stage: str = 'stage1',
        compute_metrics: bool = False
    ):
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            stage: Training stage
            compute_metrics: Whether to compute evaluation metrics
        
        Returns:
            Average validation loss, or dict with loss and metrics if compute_metrics=True
        """
        self.model.eval()
        val_losses = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                parametric_forecasts = batch.get('parametric_forecast')
                if parametric_forecasts is not None:
                    parametric_forecasts = parametric_forecasts.to(self.device)
                
                # Get emission mu and sigma first (needed for both prior and posterior)
                all_mus = []
                all_sigmas = []
                for corrector in self.model.ensemble.correctors:
                    mu, sigma = corrector(inputs)
                    all_mus.append(mu)
                    all_sigmas.append(sigma)
                emission_mu = torch.stack(all_mus, dim=0)  # (n_states, batch, horizon)
                emission_sigma = torch.stack(all_sigmas, dim=0)  # (n_states, batch, horizon)
                
                if stage == 'stage2':
                    # For Stage 2 validation, we need both prior and posterior to compute KL divergence
                    # Even though we're validating, we can use targets (future observations) to compute
                    # the posterior, since we're just evaluating, not training
                    
                    # Compute prior probabilities
                    emission_mu_for_prior = emission_mu.permute(1, 2, 0)  # (batch, horizon, n_states)
                    init_probs, trans_matrices = self.model.hmm_gating.get_prior_components(
                        x_past=inputs,
                        x_future=emission_mu_for_prior
                    )
                    
                    # Build prior probabilities (same as in training)
                    batch_size = inputs.shape[0]
                    prior_probs_list = [init_probs]
                    for t in range(1, self.model.output_dim):
                        prev_probs = prior_probs_list[-1].unsqueeze(1)  # (batch, 1, n_states)
                        trans_matrix = trans_matrices[:, t-1, :, :]  # (batch, n_states, n_states)
                        next_probs = torch.bmm(prev_probs, trans_matrix).squeeze(1)  # (batch, n_states)
                        prior_probs_list.append(next_probs)
                    prior_probs = torch.stack(prior_probs_list, dim=1)  # (batch, horizon, n_states)
                    
                    # Compute posterior probabilities using future observations (targets)
                    future_obs = targets.unsqueeze(-1)  # (batch, horizon, 1)
                    posterior_probs = self.model.hmm_gating(
                        x_past=inputs,
                        x_future=future_obs,
                        mode='posterior'
                    )  # (batch, horizon, n_states)
                    
                    # Compute log probabilities
                    prior_logp = torch.log(prior_probs + 1e-8)  # (batch, horizon, n_states)
                    posterior_logp = torch.log(posterior_probs + 1e-8)  # (batch, horizon, n_states)
                    
                    # Get model outputs for metrics
                    outputs = self.model(inputs, parametric_forecast=parametric_forecasts)
                    
                    # Compute loss with prior and posterior log probabilities
                    loss_dict = self.loss_fn(
                        emission_mu,
                        emission_sigma,
                        targets,
                        prior_logp=prior_logp,
                        posterior_logp=posterior_logp,
                        posterior_probs=posterior_probs,
                        stage=stage
                    )
                else:
                    # Stage 1 validation: use prior (no future observations available)
                    outputs = self.model(inputs, parametric_forecast=parametric_forecasts)
                    posterior_probs = outputs['state_probs']  # (batch, horizon, n_states)
                    
                    loss_dict = self.loss_fn(
                        emission_mu,
                        emission_sigma,
                        targets,
                        posterior_probs=posterior_probs,
                        stage=stage
                    )
                
                val_losses.append(loss_dict['total_loss'].item())
                
                if compute_metrics:
                    all_predictions.append(outputs['forecast'].cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
        
        avg_loss = np.mean(val_losses)
        
        if compute_metrics:
            # Compute metrics (MSE and MAE only)
            from ..evaluation.metrics import mse, mae
            
            predictions = np.concatenate(all_predictions, axis=0)
            targets = np.concatenate(all_targets, axis=0)
            
            # Flatten for metric computation
            pred_flat = predictions.flatten()
            target_flat = targets.flatten()
            
            metrics = {
                'val_loss': avg_loss,
                'val_mse': mse(target_flat, pred_flat),
                'val_mae': mae(target_flat, pred_flat)
            }
            
            self.model.train()
            return metrics
        else:
            self.model.train()
            return avg_loss
    
    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        
        torch.save(checkpoint, filepath)
        
        # Note: Best model checkpoints are saved explicitly in train_stage1/train_stage2
        # with specific names (best_model_stage1.pt, best_model_stage2.pt)
        # The is_best flag is kept for compatibility but doesn't create additional files
    
    def load_checkpoint(self, filepath: str) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        
        # Restore best epoch info if available
        if 'history' in checkpoint and 'stage1' in checkpoint['history']:
            val_losses = checkpoint['history']['stage1'].get('val_loss', [])
            if val_losses:
                best_idx = np.argmin(val_losses)
                self.best_epoch_stage1 = best_idx + 1
                self.best_val_loss_stage1 = val_losses[best_idx]
        
        return checkpoint  # Return full checkpoint dict

