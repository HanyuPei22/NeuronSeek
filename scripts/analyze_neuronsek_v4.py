import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==========================================
# 1. Core Components
# ==========================================

class L0Gate(nn.Module):
    """
    Differentiable L0 Gate using Hard Concrete distribution.
    """
    def __init__(self, temperature=0.66, limit_l=-0.1, limit_r=1.1, init_prob=0.9):
        super().__init__()
        self.temp = temperature
        self.limit_l = limit_l
        self.limit_r = limit_r
        # Initialize log_alpha for desired probability
        init_val = np.log(init_prob / (1 - init_prob))
        self.log_alpha = nn.Parameter(torch.Tensor([init_val]))

    def forward(self, x, training=True):
        if training:
            # Sampling with noise
            u = torch.rand_like(self.log_alpha)
            s = torch.sigmoid((torch.log(u + 1e-8) - torch.log(1 - u + 1e-8) + self.log_alpha) / self.temp)
            s = s * (self.limit_r - self.limit_l) + self.limit_l
        else:
            # Deterministic
            s = torch.sigmoid(self.log_alpha) * (self.limit_r - self.limit_l) + self.limit_l
        
        # Hard clamp
        z = torch.clamp(s, min=0.0, max=1.0)
        return x * z

    def regularization_term(self):
        # Expected L0 penalty
        return torch.sigmoid(self.log_alpha - self.temp * np.log(-self.limit_l / self.limit_r))
    
    def get_prob(self):
        return torch.sigmoid(self.log_alpha).item()

class DualStreamInteractionLayer(nn.Module):
    """
    Computes Pure (Power) terms and Interaction (CP-Decomposition) terms.
    """
    def __init__(self, input_dim, num_classes, rank, poly_order):
        super().__init__()
        self.rank = rank
        self.poly_order = poly_order
        
        # Interaction Stream: Factors for CP decomposition
        self.factors = nn.ModuleList()
        for i in range(1, poly_order + 1):
            # Order i requires i factor matrices
            order_params = nn.ParameterList([
                nn.Parameter(torch.empty(input_dim, rank, num_classes)) 
                for _ in range(i)
            ])
            self.factors.append(order_params)

        # Pure Stream: Linear coefficients for power terms
        self.coeffs_pure = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, num_classes))
            for _ in range(poly_order)
        ])
        
        self.beta = nn.Parameter(torch.zeros(num_classes))
        self.reset_parameters()

    def reset_parameters(self):
        # Increased init std to 0.05 to prevent vanishing gradients in high-order terms
        for order_params in self.factors:
            for p in order_params:
                nn.init.normal_(p, std=0.05) 
        for p in self.coeffs_pure:
            nn.init.normal_(p, std=0.05)

    def get_pure_term(self, x, i):
        order = i + 1
        term = x if order == 1 else x.pow(order)
        return term @ self.coeffs_pure[i]

    def get_interaction_term(self, x, i):
        factors = self.factors[i]
        # Calculate parallel projections: (X @ U)
        projections = [torch.einsum('bd, drc -> brc', x, u) for u in factors]
        
        # Element-wise product of projections
        combined = projections[0]
        for p in projections[1:]:
            combined = combined * p
            
        # Sum over rank dimension
        return torch.sum(combined, dim=1)

class SparseSearchAgent(nn.Module):
    """
    Orchestrates the DualStreamLayer, BatchNorm, and L0Gates.
    """
    def __init__(self, input_dim=10, num_classes=1, rank=8, max_order=5):
        super().__init__()
        self.max_order = max_order
        self.core = DualStreamInteractionLayer(input_dim, num_classes, rank, max_order)
        self.gates_pure = nn.ModuleList([L0Gate() for _ in range(max_order)])
        self.gates_int  = nn.ModuleList([L0Gate() for _ in range(max_order)])
        
        # BatchNorm is critical for gradient flow in interaction terms
        self.bn_pure = nn.ModuleList(nn.BatchNorm1d(num_classes, affine=True) for _ in range(max_order))
        self.bn_int = nn.ModuleList(nn.BatchNorm1d(num_classes, affine=True) for _ in range(max_order))

    def forward(self, x, training=True):
        output = self.core.beta
        
        # Pure Stream
        for i, gate in enumerate(self.gates_pure):
            term = self.core.get_pure_term(x, i)
            term = self.bn_pure[i](term)
            output = output + gate(term, training=training)
            
        # Interaction Stream
        for i, gate in enumerate(self.gates_int):
            term = self.core.get_interaction_term(x, i)
            term = self.bn_int[i](term)
            output = output + gate(term, training=training)
            
        return output
    
    def calculate_regularization(self):
        reg = 0.0
        # Heavier penalty (5.0) for Pure terms to prioritize Interaction search
        for i, g in enumerate(self.gates_pure):
            reg += 1.0 * g.regularization_term()
        for i, g in enumerate(self.gates_int):
            reg += 1.0 * g.regularization_term()
        return reg

# ==========================================
# 2. Searcher (Trainer)
# ==========================================

class FullMonitorSearcher:
    def __init__(self, input_dim, rank=8, max_order=5, epochs=200):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.max_order = max_order
        self.agent = SparseSearchAgent(input_dim=input_dim, rank=rank, max_order=max_order).to(self.device)
        
        # Logging containers
        self.logs = {
            'loss': [],
            'gate_pure': {o: [] for o in range(1, max_order+1)},
            'gate_int':  {o: [] for o in range(1, max_order+1)},
            'weight_pure': {o: [] for o in range(1, max_order+1)},
            'weight_int':  {o: [] for o in range(1, max_order+1)},
            'lambda_val': []
        }

    def _get_tensor_norm(self, parameter_list):
        # Calculate mean absolute value of tensors in the list
        norms = [p.detach().abs().mean().item() for p in parameter_list]
        return np.mean(norms)

    def fit(self, X, y):
        # Data setup
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        # Optimizer Setup: Differential Learning Rates + Low Weight Decay
        optimizer = optim.Adam([
            {'params': self.agent.core.coeffs_pure.parameters(), 'lr': 0.005, 'weight_decay': 1e-4},
            {'params': [self.agent.core.beta], 'lr': 0.005},
            {'params': self.agent.core.factors.parameters(), 'lr': 0.02, 'weight_decay': 1e-5},
            {'params': list(self.agent.bn_pure.parameters()) + list(self.agent.bn_int.parameters()), 'lr': 0.01},
            {'params': list(self.agent.gates_pure.parameters()) + list(self.agent.gates_int.parameters()), 'lr': 0.01}
        ])
        
        # Scheduler: Smooth decay to minimize late-stage oscillation
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)
        
        loss_fn = nn.MSELoss()
        
        # Annealing Schedule
        warmup_end = 50
        anneal_end = 150
        max_lambda = 0.05
        
        print(f"Starting Search | Dim={X.shape[1]} | Annealing: {warmup_end}->{anneal_end}")

        for epoch in range(self.epochs):
            self.agent.train()
            
            # Calculate dynamic lambda
            current_lambda = 0.0
            if epoch >= warmup_end:
                if epoch < anneal_end:
                    progress = (epoch - warmup_end) / (anneal_end - warmup_end)
                    current_lambda = max_lambda * progress
                else:
                    current_lambda = max_lambda
            self.logs['lambda_val'].append(current_lambda)

            # Freeze/Unfreeze Gates
            is_frozen = epoch < warmup_end
            for p in self.agent.gates_pure.parameters(): p.requires_grad = not is_frozen
            for p in self.agent.gates_int.parameters(): p.requires_grad = not is_frozen
            
            total_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                pred = self.agent(bx, training=True)
                mse = loss_fn(pred, by)
                
                # Apply Regularization
                reg = 0.0
                if current_lambda > 0:
                    reg = current_lambda * self.agent.calculate_regularization()
                
                loss = mse + reg
                loss.backward()
                
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
            
            # Step Scheduler
            scheduler.step()
            
            # Monitoring
            avg_loss = total_loss / len(loader)
            self.logs['loss'].append(avg_loss)
            
            for i in range(self.max_order):
                order = i + 1
                # Track Gates
                self.logs['gate_pure'][order].append(self.agent.gates_pure[i].get_prob())
                self.logs['gate_int'][order].append(self.agent.gates_int[i].get_prob())
                
                # Track Weight Norms
                w_pure = self.agent.core.coeffs_pure[i].norm(p=2).item()
                self.logs['weight_pure'][order].append(w_pure)
                
                w_int = self._get_tensor_norm(self.agent.core.factors[i])
                self.logs['weight_int'][order].append(w_int)

            if epoch % 20 == 0:
                print(f"Ep {epoch} | Loss: {avg_loss:.4f} | Lambda: {current_lambda:.4f}")

    def get_structure_str(self):
        pure_active = []
        int_active = []
        for i in range(self.max_order):
            if self.agent.gates_pure[i].get_prob() > 0.5: pure_active.append(i+1)
            if self.agent.gates_int[i].get_prob() > 0.5: int_active.append(i+1)
        return f"Pure: {pure_active}\nInteract: {int_active}"

# ==========================================
# 3. Execution & Plotting
# ==========================================

def main():
    DIM = 100
    print("Generating Synthetic Data (Truth: Interaction Order 2)...")
    
    # Data Gen
    X = torch.randn(2000, DIM)
    y = torch.zeros(2000, 1)
    
    # Ground Truth: Rank-4 Order-2 Interaction
    # Signal multiplied by 5.0 to ensure it is learnable
    rank_truth = 4
    for _ in range(rank_truth):
        u = torch.randn(DIM, 1) / np.sqrt(DIM)
        v = torch.randn(DIM, 1) / np.sqrt(DIM)
        y += (X @ u) * (X @ v) * 1.0 
    
    y = y + 0.1 * torch.randn_like(y)
    y = (y - y.mean()) / (y.std() + 1e-8) # Standardize
    baseline_mse = torch.var(y).item()
    print(f"Baseline MSE: {baseline_mse:.4f}")

    # Run Search
    searcher = FullMonitorSearcher(DIM, rank=8, max_order=5, epochs=200)
    searcher.fit(X, y)
    
    final_struct = searcher.get_structure_str()
    print(f"\nFinal Structure:\n{final_struct}")

    # Plotting setup
    logs = searcher.logs
    epochs = range(len(logs['loss']))
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig)
    
    # 1. Loss Curve (Span Top Row)
    ax_loss = fig.add_subplot(gs[0, :])
    ax_loss.plot(epochs, logs['loss'], 'k-', label='Total Loss')
    ax_loss.axhline(baseline_mse, color='gray', linestyle=':', label='Baseline')
    ax_loss.set_yscale('log')
    ax_loss.set_title(f'Training Loss (Baseline={baseline_mse:.2f})')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    
    # 2. Pure Weights (Middle Left)
    ax_wp = fig.add_subplot(gs[1, 0])
    for order in range(1, 6):
        ax_wp.plot(epochs, logs['weight_pure'][order], label=f'Pure {order}')
    ax_wp.set_title('Pure Stream Weights (L2 Norm)')
    ax_wp.legend(fontsize='small')
    ax_wp.grid(True, alpha=0.3)
    
    # 3. Int Weights (Middle Right)
    ax_wi = fig.add_subplot(gs[1, 1])
    for order in range(1, 6):
        lw = 3 if order == 2 else 1
        ax_wi.plot(epochs, logs['weight_int'][order], label=f'Int {order}', linewidth=lw)
    ax_wi.set_title('Interaction Stream Weights (Mean Abs)')
    ax_wi.legend(fontsize='small')
    ax_wi.grid(True, alpha=0.3)
    
    # 4. Pure Gates (Bottom Left)
    ax_gp = fig.add_subplot(gs[2, 0])
    for order in range(1, 6):
        ax_gp.plot(epochs, logs['gate_pure'][order], label=f'Pure {order}')
    ax_gp.set_title('Pure Gates (Prob)')
    ax_gp.set_ylim(-0.1, 1.1)
    ax_gp.grid(True, alpha=0.3)
    
    # 5. Int Gates (Bottom Right)
    ax_gi = fig.add_subplot(gs[2, 1])
    ax_gi.plot(epochs, [l/0.05 for l in logs['lambda_val']], 'k:', alpha=0.3, label='Lambda (Scaled)')
    for order in range(1, 6):
        lw = 3 if order == 2 else 1
        ax_gi.plot(epochs, logs['gate_int'][order], label=f'Int {order}', linewidth=lw)
    ax_gi.set_title(f'Interaction Gates (Prob) | Result: {final_struct.replace(chr(10), " | ")}')
    ax_gi.set_ylim(-0.1, 1.1)
    ax_gi.legend(fontsize='small')
    ax_gi.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('neuronseek_v5_final.png')
    print("\nPlot saved to 'neuronseek_v5_final.png'")

if __name__ == "__main__":
    main()