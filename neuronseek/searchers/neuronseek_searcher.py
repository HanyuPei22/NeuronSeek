import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
from .base import BaseStructureSearcher 
from neuronseek.models.sparse_search_agent import SparseSearchAgent

class NeuronSeekSearcher(BaseStructureSearcher):
    def __init__(self, input_dim: int, num_classes: int = 1, rank: int = 8, 
                 epochs: int = 200, batch_size: int = 64, reg_lambda: float = 0.05): # [NEW] Added reg_lambda
        super().__init__(input_dim)
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.rank = rank
        self.reg_lambda = reg_lambda # Store it
        
        self.agent = SparseSearchAgent(
            input_dim=input_dim, 
            num_classes=num_classes, 
            rank=rank
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.agent.to(self.device)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        if self.num_classes > 1:
            y_t = torch.tensor(y, dtype=torch.long).reshape(-1).to(self.device)
            loss_fn = nn.CrossEntropyLoss()
        else:
            y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device)
            loss_fn = nn.MSELoss()
        
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # [CRITICAL FIX]
        # While Reference used 0.02, D=100 caused divergence (Loss > 1.7).
        # We MUST lower LR to 0.01 to ensure stability while keeping the diff-LR logic.
        optimizer = torch.optim.Adam([
            {'params': self.agent.core.coeffs_pure.parameters(), 'lr': 0.005, 'weight_decay': 1e-4},
            {'params': [self.agent.core.beta], 'lr': 0.005},
            {'params': self.agent.core.factors.parameters(), 'lr': 0.01, 'weight_decay': 1e-5}, # Adjusted for D=100
            {'params': list(self.agent.bn_pure.parameters()) + list(self.agent.bn_int.parameters()), 'lr': 0.01},
            {'params': list(self.agent.gates_pure.parameters()) + list(self.agent.gates_int.parameters()), 'lr': 0.01}
        ])
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)
        
        warmup_end = int(self.epochs * 0.25)
        anneal_end = int(self.epochs * 0.75)
        
        # [NEW] Use dynamic lambda
        max_lambda = self.reg_lambda 

        print(f"Starting Search | Rank={self.rank} | Lambda={max_lambda}")

        self.agent.train()
        for epoch in range(self.epochs):
            
            # Dynamic Lambda Logic (Matches Reference)
            current_lambda = 0.0
            if epoch >= warmup_end:
                if epoch < anneal_end:
                    progress = (epoch - warmup_end) / (anneal_end - warmup_end)
                    current_lambda = max_lambda * progress
                else:
                    current_lambda = max_lambda

            is_frozen = epoch < warmup_end
            for p in self.agent.gates_pure.parameters(): p.requires_grad = not is_frozen
            for p in self.agent.gates_int.parameters(): p.requires_grad = not is_frozen

            total_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                pred = self.agent(bx, training=True)
                task_loss = loss_fn(pred, by)
                
                reg_loss = 0.0
                if current_lambda > 0:
                    reg_loss = current_lambda * self.agent.calculate_regularization()
                
                loss = task_loss + reg_loss
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            
            scheduler.step()

            if (epoch + 1) % 20 == 0:
                print(f"Ep {epoch+1} | Loss: {total_loss/len(loader):.4f} | Lambda: {current_lambda:.4f}")

    def get_structure_info(self) -> Dict[str, Any]:
        pure_idx, int_idx = self.agent.get_structure(threshold=0.5)

        print(f"Structure Found - Pure: {pure_idx}, Interact: {int_idx}")
        
        return {
            'type': 'neuronseek',
            'pure_indices': pure_idx,
            'interact_indices': int_idx,
            'rank': self.rank
        }