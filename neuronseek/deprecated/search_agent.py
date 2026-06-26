import torch
import torch.nn as nn
import torch.optim as optim
from neuronseek.core.tensor_interaction import TensorInteractionLayer

class SearchAgent(nn.Module):
    def __init__(self, input_dim=512, num_classes=10, rank=32, poly_order=3, reg_lambda=0.1, task_type='classification'):
        super().__init__()
        self.core = TensorInteractionLayer(input_dim, num_classes, rank, poly_order)
        self.reg_lambda = reg_lambda
        self.criterion = nn.CrossEntropyLoss() if task_type == 'classification' else nn.MSELoss()

    def fit_stridge(self, dataset, epochs=50, batch_size=64, device='cuda', prune_threshold=0.05):
            self.to(device)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            optimizer = optim.Adam(self.parameters(), lr=1e-3)
            
            # [Strategy] Warmup: Do not apply regularization in the first 20% of epochs
            # This allows the complex Interaction Stream (CP) to learn initial patterns
            # before the sparsity constraint kicks in.
            warmup_epochs = int(epochs * 0.2) 
            
            print(f"--- Search Start: Order={self.core.poly_order}, Reg={self.reg_lambda}, Warmup={warmup_epochs} ---")
            
            for epoch in range(epochs):
                self.train()
                
                # Dynamic Regularization Schedule
                current_reg_lambda = 0.0 if epoch < warmup_epochs else self.reg_lambda

                if epoch == epochs // 2:
                    self._prune_terms(threshold=prune_threshold)

                total_loss = 0
                for X, y in loader:
                    X, y = X.to(device), y.to(device)
                    optimizer.zero_grad()
                    
                    logits, loss_coeffs = self.core(X)
                    
                    loss_factors = 0.0
                    for i in range(self.core.poly_order):
                        if self.core.mask_interact[i] == 1.0:
                            for f in self.core.factors[i]:
                                loss_factors += torch.norm(f, p='fro')
                    
                    # Reduced factor penalty to 0.01 to avoid killing learning capability
                    total_reg = loss_coeffs + 0.01 * loss_factors
                    
                    loss = self.criterion(logits, y) + current_reg_lambda * total_reg
                    loss.backward()
                    
                    self._mask_gradients()     
                    optimizer.step()
                    total_loss += loss.item()
                
                if (epoch+1) % 10 == 0:
                    print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.4f}")

    def _prune_terms(self, threshold):
        print("\n[Pruning] Analyzing Effective Contribution...")
        with torch.no_grad():
            stats = [] 
            
            for i in range(self.core.poly_order):
                # 1. Pure Stream Energy (RMS Normalized)
                # Norm / sqrt(numel) makes it independent of input dimension
                w_pure = self.core.coeffs_pure[i]
                mag_pure = w_pure.norm(p='fro') / (w_pure.numel() ** 0.5)
                stats.append(('pure', i, mag_pure.item()))

                # 2. Interaction Stream Energy (Proxy Upper Bound)
                # Est = ||U|| * ||V|| * ||K|| (Each normalized)
                factor_prod = 1.0
                for f in self.core.factors[i]:
                    factor_prod *= (f.norm(p='fro') / (f.shape[0] ** 0.5))
                
                k = self.core.coeffs_interact[i]
                k_norm = k.norm(p='fro') / (k.shape[0] ** 0.5)
                
                stats.append(('interact', i, (factor_prod * k_norm).item()))

            # Pruning Logic: Relative ratio to the strongest term
            max_mag = max([s[2] for s in stats]) + 1e-9
            
            for (type_str, i, mag) in stats:
                ratio = mag / max_mag
                is_kept = ratio >= threshold
                
                if not is_kept:
                    if type_str == 'pure':
                        self.core.mask_pure[i] = 0.0
                        self.core.coeffs_pure[i].fill_(0.0)
                    else:
                        self.core.mask_interact[i] = 0.0
                        self.core.coeffs_interact[i].fill_(0.0)
                
                status = "KEEP" if is_kept else "KILL"
                print(f" -> {status} {type_str.upper()} Order {i+1} (Mag: {mag:.4f}, Ratio: {ratio:.3f})")
        print("-" * 30 + "\n")

    def _mask_gradients(self):
        """Ensures gradients of pruned terms are zero."""
        for i in range(self.core.poly_order):
            # Mask Interaction
            if self.core.mask_interact[i] == 0:
                if self.core.coeffs_interact[i].grad is not None:
                    self.core.coeffs_interact[i].grad.fill_(0.0)
                for f in self.core.factors[i]:
                    if f.grad is not None: f.grad.fill_(0.0)
            
            # Mask Pure
            if self.core.mask_pure[i] == 0:
                if self.core.coeffs_pure[i].grad is not None:
                    self.core.coeffs_pure[i].grad.fill_(0.0)

    def get_discovered_orders(self):
        return {
            'pure': [i+1 for i, m in enumerate(self.core.mask_pure) if m == 1.0],
            'interact': [i+1 for i, m in enumerate(self.core.mask_interact) if m == 1.0]
        }