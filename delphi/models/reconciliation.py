"""
MinT (Minimum Trace) Reconciliation for hierarchical forecast coherence.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
import warnings

try:
    from hierarchicalforecast.reconciliation import MinT
    from hierarchicalforecast.core import HierarchicalReconciliation
    HIERARCHICALFORECAST_AVAILABLE = True
except ImportError:
    HIERARCHICALFORECAST_AVAILABLE = False
    warnings.warn("hierarchicalforecast not available. MinT reconciliation will be simplified.")

warnings.filterwarnings('ignore')


class MinTReconciliation:
    """
    Minimum Trace (MinT) reconciliation for hierarchical forecasts.
    
    Ensures coherence across different segmentation levels (e.g., regional → global,
    category → total) by minimizing the trace of the forecast error covariance matrix.
    """
    
    def __init__(
        self,
        method: str = 'mint',
        hierarchy_levels: Optional[List[str]] = None
    ):
        """
        Initialize MinT reconciliation.
        
        Args:
            method: Reconciliation method ('mint' for Minimum Trace)
            hierarchy_levels: List of hierarchy level names (if applicable)
        """
        self.method = method
        self.hierarchy_levels = hierarchy_levels or []
        self.reconciler = None
        
        if HIERARCHICALFORECAST_AVAILABLE:
            try:
                self.reconciler = MinT(method=method)
            except Exception as e:
                print(f"Warning: Could not initialize MinT reconciler: {e}")
                self.reconciler = None
    
    def _build_summing_matrix(
        self,
        hierarchy: Dict[str, List[str]],
        series_ids: List[str]
    ) -> np.ndarray:
        """
        Build summing matrix S from hierarchy definition.
        
        Args:
            hierarchy: Dictionary mapping parent_id -> [child_ids]
            series_ids: List of all series IDs (bottom level first, then aggregates)
        
        Returns:
            Summing matrix S of shape (n_bottom, n_total)
        """
        n_total = len(series_ids)
        # Find bottom-level series (those that don't appear as parents)
        parent_ids = set(hierarchy.keys())
        bottom_ids = [sid for sid in series_ids if sid not in parent_ids]
        n_bottom = len(bottom_ids)
        
        # Create mapping from series_id to index
        id_to_idx = {sid: i for i, sid in enumerate(series_ids)}
        bottom_to_idx = {sid: i for i, sid in enumerate(bottom_ids)}
        
        # Initialize summing matrix
        S = np.zeros((n_bottom, n_total))
        
        # Bottom-level series map to themselves
        for i, bottom_id in enumerate(bottom_ids):
            if bottom_id in id_to_idx:
                S[i, id_to_idx[bottom_id]] = 1.0
        
        # Aggregate series are sums of their children
        for parent_id, children_ids in hierarchy.items():
            if parent_id in id_to_idx:
                parent_idx = id_to_idx[parent_id]
                for child_id in children_ids:
                    if child_id in bottom_to_idx:
                        bottom_idx = bottom_to_idx[child_id]
                        S[bottom_idx, parent_idx] = 1.0
        
        return S
    
    def reconcile(
        self,
        forecasts: Dict[str, np.ndarray],
        hierarchy: Optional[Dict[str, List[str]]] = None,
        forecast_covariance: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Reconcile forecasts to ensure hierarchical coherence using MinT method.
        
        Args:
            forecasts: Dictionary of series_id -> forecast array
            hierarchy: Optional hierarchy definition (parent -> [children])
            forecast_covariance: Optional forecast error covariance matrix
        
        Returns:
            Reconciled forecasts
        """
        if hierarchy is None or len(hierarchy) == 0:
            # If no hierarchy, return original forecasts
            return forecasts
        
        try:
            # Build summing matrix
            series_ids = list(forecasts.keys())
            S = self._build_summing_matrix(hierarchy, series_ids)
            
            # Use MinT reconciliation with covariance
            if forecast_covariance is not None:
                return self.reconcile_with_covariance(
                    forecasts, forecast_covariance, S
                )
            else:
                # Use simple reconciliation if no covariance
                return self.reconcile_simple_from_matrix(forecasts, S, series_ids)
                
        except Exception as e:
            print(f"Error in reconciliation: {e}")
            return forecasts
    
    def reconcile_simple_from_matrix(
        self,
        forecasts: Dict[str, np.ndarray],
        S: np.ndarray,
        series_ids: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Simple reconciliation using summing matrix.
        
        Args:
            forecasts: Dictionary of forecasts
            S: Summing matrix
            series_ids: List of series IDs
        
        Returns:
            Reconciled forecasts
        """
        reconciled = forecasts.copy()
        
        # Convert forecasts to matrix
        forecast_horizon = len(forecasts[series_ids[0]])
        Y_base = np.array([forecasts[sid] for sid in series_ids])
        
        # Simple reconciliation: y_recon = S * (S^T * S)^-1 * S^T * y_base
        try:
            S_T_S = S.T @ S
            if np.linalg.cond(S_T_S) > 1e12:
                S_T_S_inv = np.linalg.pinv(S_T_S)
            else:
                S_T_S_inv = np.linalg.inv(S_T_S)
            
            P = S @ S_T_S_inv @ S.T
            Y_reconciled = P @ Y_base
            
            # Convert back to dictionary
            for i, sid in enumerate(series_ids):
                reconciled[sid] = Y_reconciled[i, :]
        except Exception as e:
            print(f"Error in simple reconciliation: {e}")
        
        return reconciled
    
    def reconcile_simple(
        self,
        forecasts: Dict[str, np.ndarray],
        parent_series: Optional[str] = None,
        child_series: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Simple reconciliation for parent-child relationships.
        
        Ensures sum of child forecasts equals parent forecast.
        
        Args:
            forecasts: Dictionary of series_id -> forecast array
            parent_series: Parent series ID
            child_series: List of child series IDs
        
        Returns:
            Reconciled forecasts
        """
        reconciled = forecasts.copy()
        
        if parent_series is None or child_series is None:
            return reconciled
        
        if parent_series not in forecasts:
            return reconciled
        
        # Get parent forecast
        parent_forecast = forecasts[parent_series]
        
        # Get child forecasts
        child_forecasts = []
        valid_children = []
        for child_id in child_series:
            if child_id in forecasts:
                child_forecasts.append(forecasts[child_id])
                valid_children.append(child_id)
        
        if len(child_forecasts) == 0:
            return reconciled
        
        # Sum of child forecasts
        child_sum = np.sum(child_forecasts, axis=0)
        
        # Compute adjustment factor
        # Adjust children proportionally to match parent
        if np.sum(np.abs(child_sum)) > 1e-6:
            adjustment_factor = parent_forecast / child_sum
        else:
            adjustment_factor = np.ones_like(parent_forecast)
        
        # Apply adjustment to children
        for i, child_id in enumerate(valid_children):
            reconciled[child_id] = child_forecasts[i] * adjustment_factor
        
        return reconciled
    
    def reconcile_with_covariance(
        self,
        forecasts: Dict[str, np.ndarray],
        forecast_covariance: Optional[np.ndarray] = None,
        hierarchy_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        MinT reconciliation using forecast error covariance matrix.
        
        Implements: y_reconciled = S * (S^T * W^-1 * S)^-1 * S^T * W^-1 * y_base
        
        Where:
            S = summing matrix (hierarchy structure)
            W = forecast error covariance matrix
            y_base = base forecasts
        
        Args:
            forecasts: Dictionary of base forecasts
            forecast_covariance: Forecast error covariance matrix
            hierarchy_matrix: Summing matrix S (n_bottom x n_total)
        
        Returns:
            Reconciled forecasts
        """
        if hierarchy_matrix is None:
            return forecasts
        
        # Convert forecasts to matrix
        series_ids = list(forecasts.keys())
        n_series = len(series_ids)
        forecast_horizon = len(forecasts[series_ids[0]])
        
        # Base forecasts matrix: (n_series, forecast_horizon)
        Y_base = np.array([forecasts[sid] for sid in series_ids])
        
        # Summing matrix S: (n_bottom, n_total)
        S = hierarchy_matrix
        n_bottom, n_total = S.shape
        
        if n_total != n_series:
            print(f"Warning: Hierarchy matrix size {n_total} doesn't match number of series {n_series}")
            return forecasts
        
        # Forecast error covariance W
        if forecast_covariance is None:
            # Use identity if not provided
            W = np.eye(n_series)
        else:
            W = forecast_covariance
        
        # Ensure W is positive definite
        W = (W + W.T) / 2  # Symmetrize
        W += np.eye(n_series) * 1e-6  # Add small diagonal for numerical stability
        
        try:
            # Compute reconciliation: y_recon = S * (S^T * W^-1 * S)^-1 * S^T * W^-1 * y_base
            W_inv = np.linalg.inv(W)
            S_T_W_inv = S.T @ W_inv
            S_T_W_inv_S = S_T_W_inv @ S
            
            # Check if invertible
            if np.linalg.cond(S_T_W_inv_S) > 1e12:
                print("Warning: S^T * W^-1 * S is ill-conditioned, using pseudo-inverse")
                S_T_W_inv_S_inv = np.linalg.pinv(S_T_W_inv_S)
            else:
                S_T_W_inv_S_inv = np.linalg.inv(S_T_W_inv_S)
            
            # Reconciliation matrix
            P = S @ S_T_W_inv_S_inv @ S_T_W_inv
            
            # Apply reconciliation to each forecast step
            Y_reconciled = P @ Y_base
            
            # Convert back to dictionary
            reconciled = {}
            for i, sid in enumerate(series_ids):
                reconciled[sid] = Y_reconciled[i, :]
            
            return reconciled
            
        except Exception as e:
            print(f"Error in MinT reconciliation: {e}")
            return forecasts

