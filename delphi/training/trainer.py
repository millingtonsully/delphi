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
        stage2_epochs: int = 30
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
        """
        self.model = model.to(device)
        self.device = device
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        
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
            
            for batch in tqdm(train_loader, desc=f"Stage 1 Epoch {epoch+1}/{num_epochs}"):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                parametric_forecasts = batch.get('parametric_forecast')
                if parametric_forecasts is not None:
                    parametric_forecasts = parametric_forecasts.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs, parametric_forecast=parametric_forecasts)
                
                # Get posterior probabilities
                posterior_probs = outputs['state_probs']
                
                # Compute loss
                loss_dict = self.loss_fn(
                    outputs['forecast'],
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
            
            # Validation
            if val_loader is not None:
                # Compute metrics every 10 epochs or on last epoch
                compute_metrics = (epoch + 1) % 10 == 0 or epoch == num_epochs - 1
                val_result = self.validate(val_loader, stage='stage1', compute_metrics=compute_metrics)
                
                if compute_metrics and isinstance(val_result, dict):
                    val_loss = val_result['val_loss']
                    self.history['stage1']['val_loss'].append(val_loss)
                    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
                    print(f"  Validation Metrics - MASE: {val_result['val_mase']:.4f}, "
                          f"MAE: {val_result['val_mae']:.4f}, RMSE: {val_result['val_rmse']:.4f}")
                else:
                    val_loss = val_result
                    self.history['stage1']['val_loss'].append(val_loss)
                    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
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
        
        # Freeze all parameters except HMM prior
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze HMM prior parameters
        for param in self.model.hmm_gating.initial_xlstm.parameters():
            param.requires_grad = True
        for param in self.model.hmm_gating.transition_xlstm.parameters():
            param.requires_grad = True
        for param in self.model.hmm_gating.fc_initial.parameters():
            param.requires_grad = True
        for param in self.model.hmm_gating.fc_trans.parameters():
            param.requires_grad = True
        
        # Create optimizer for prior only
        prior_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        prior_optimizer = optim.Adam(prior_params, lr=1e-4)
        
        self.model.train()
        
        for epoch in range(num_epochs):
            train_losses = []
            
            for batch in tqdm(train_loader, desc=f"Stage 2 Epoch {epoch+1}/{num_epochs}"):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                parametric_forecasts = batch.get('parametric_forecast')
                if parametric_forecasts is not None:
                    parametric_forecasts = parametric_forecasts.to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    # Get posterior probabilities (frozen)
                    posterior_probs = self.model.hmm_gating(inputs, mode='posterior')
                
                # Get prior initial state probabilities (trainable)
                # This computes p(z_0 | x) using the prior xLSTM
                init_out, _ = self.model.hmm_gating.initial_xlstm(inputs[:, :1, :])
                init_logits = self.model.hmm_gating.fc_initial(
                    self.model.hmm_gating.dropout(init_out.squeeze(1))
                )
                prior_probs = torch.nn.functional.softmax(init_logits, dim=-1)
                
                # Compute log probabilities for KL divergence
                # KL(q||p) = sum_z q(z) * (log q(z) - log p(z))
                posterior_logp = torch.log(posterior_probs + 1e-8)
                prior_logp = torch.log(prior_probs + 1e-8)
                
                # KL divergence: E_q[log q - log p]
                kl_div = (posterior_probs * (posterior_logp - prior_logp)).sum(dim=-1).mean()
                
                # Compute loss (just the KL term for Stage 2)
                loss_dict = self.loss_fn(
                    torch.zeros_like(targets),  # Dummy reconstruction
                    torch.zeros_like(targets),  # Dummy target
                    prior_logp=prior_logp.mean(),
                    posterior_logp=posterior_logp.mean(),
                    posterior_probs=posterior_probs,
                    stage='stage2'
                )
                
                # Use the computed KL divergence as the loss
                loss = kl_div
                
                # Backward pass
                prior_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(prior_params, max_norm=1.0)
                prior_optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            self.history['stage2']['train_loss'].append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                # Compute metrics every 10 epochs or on last epoch
                compute_metrics = (epoch + 1) % 10 == 0 or epoch == num_epochs - 1
                val_result = self.validate(val_loader, stage='stage2', compute_metrics=compute_metrics)
                
                if compute_metrics and isinstance(val_result, dict):
                    val_loss = val_result['val_loss']
                    self.history['stage2']['val_loss'].append(val_loss)
                    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
                    print(f"  Validation Metrics - MASE: {val_result['val_mase']:.4f}, "
                          f"MAE: {val_result['val_mae']:.4f}, RMSE: {val_result['val_rmse']:.4f}")
                else:
                    val_loss = val_result
                    self.history['stage2']['val_loss'].append(val_loss)
                    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
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
                
                outputs = self.model(inputs, parametric_forecast=parametric_forecasts)
                posterior_probs = outputs['state_probs']
                
                loss_dict = self.loss_fn(
                    outputs['forecast'],
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
            # Compute metrics
            from ..evaluation.metrics import mase, mae, rmse
            
            predictions = np.concatenate(all_predictions, axis=0)
            targets = np.concatenate(all_targets, axis=0)
            
            # Flatten for metric computation
            pred_flat = predictions.flatten()
            target_flat = targets.flatten()
            
            metrics = {
                'val_loss': avg_loss,
                'val_mase': mase(target_flat, pred_flat),
                'val_mae': mae(target_flat, pred_flat),
                'val_rmse': rmse(target_flat, pred_flat)
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
        
        if is_best:
            best_path = str(Path(filepath).parent / 'best_model.pt')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint['epoch']

