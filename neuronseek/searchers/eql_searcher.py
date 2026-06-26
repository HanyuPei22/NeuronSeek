import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any

from .base import BaseStructureSearcher
from neuronseek.utils.seeds import setup_seed

class EQLNetwork(nn.Module):
    """
    True EQL Network implementation.
    Architecture: Input -> Projection -> Basis Functions -> Aggregation -> Output
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Layer 1: Linear Projection (inner weights)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        
        # Layer 2: Aggregation (outer weights)
        # Input size * 4 because we apply 4 types of basis functions to each hidden node
        self.linear2 = nn.Linear(hidden_dim * 4, 1)

    def forward(self, x):
        # 1. Projection: z = W1*x + b1
        z = self.linear1(x)
        
        # 2. Basis Functions Application
        # f(z) = [z, sin(z), cos(z), sigmoid(z)]
        out = torch.cat([
            z,                  # Identity
            torch.sin(z),       # Sine
            torch.cos(z),       # Cosine
            torch.sigmoid(z)    # Sigmoid
        ], dim=1)
        
        # 3. Aggregation: y = W2*f(z) + b2
        return self.linear2(out)

    def prune_weights(self, threshold: float):
        """Zero out weights below threshold to enforce sparsity."""
        with torch.no_grad():
            self.linear1.weight.data[self.linear1.weight.abs() < threshold] = 0
            self.linear2.weight.data[self.linear2.weight.abs() < threshold] = 0

class EQLSearcher(BaseStructureSearcher):
    """
    Searcher based on Equation Learner (EQL).
    Uses L1 regularization to discover sparse symbolic structures.
    """
    def __init__(self, input_dim: int, hidden_dim=20, epochs=1000, 
                 lambda_l1=1e-4, threshold=1e-3, lr=1e-3):
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lambda_l1 = lambda_l1
        self.threshold = threshold
        self.lr = lr
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray,callback = None) -> None:
        setup_seed(42)
        
        # Convert data
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        
        # Init model
        self.model = EQLNetwork(self.input_dim, self.hidden_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mse_fn = nn.MSELoss()
        
        # Training Loop
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pred = self.model(X_t)
            mse = mse_fn(pred, y_t)
            
            # L1 Regularization for sparsity
            reg_loss = 0
            for param in self.model.parameters():
                reg_loss += torch.sum(torch.abs(param))
            
            loss = mse + self.lambda_l1 * reg_loss
            loss.backward()
            optimizer.step()
            
            # Optional: Hard thresholding during training (Test Phase logic)
            if epoch % 100 == 0:
                self.model.prune_weights(self.threshold)

            if callback:
                should_stop = callback(epoch=epoch, model=self.model, loss=mse.item())
                if should_stop:
                    print(f"EQL stopped by callback at epoch {epoch}")
                    break

        # Final Pruning
        self.model.prune_weights(self.threshold)

    def get_structure_info(self) -> dict:
        """
        Parses the sparse network to identify active inputs.
        """
        if self.model is None:
            return {'type': 'explicit_terms', 'terms': []}
            
        # Analyze W1 (Input -> Hidden)
        # Shape: [Hidden, Input]
        w1 = self.model.linear1.weight.detach().cpu().numpy()
        
        # Calculate feature importance: L2 norm of columns in W1
        # If feature 'i' has 0 weights to all hidden units, it is pruned.
        feat_importance = np.linalg.norm(w1, axis=0)
        
        # Identify active indices
        active_indices = np.where(feat_importance > 1e-5)[0].tolist()
        active_indices.sort()
        
        # Identify Complexity (Number of active basis functions)
        w2 = self.model.linear2.weight.detach().cpu().numpy()
        active_funcs = np.sum(np.abs(w2) > 1e-5)
        
        print(f"[EQL] Structure Found: {len(active_indices)} active inputs, {active_funcs} active basis terms.")

        # EQL often acts as a Feature Selector in high-dim.
        # We transform result into 'linear' terms for Stage 2 compatibility
        terms = []
        for idx in active_indices:
            terms.append({
                'type': 'linear',
                'indices': [idx],
                'raw': f'eql_feature_{idx}'
            })
            
        return {
            'type': 'explicit_terms',
            'description': f"EQL_Net(In={len(active_indices)}, Nodes={active_funcs})",
            'terms': terms
        }