import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 基础组件 (L0Gate, Core, Agent)
# ==========================================

class L0Gate(nn.Module):
    def __init__(self, temperature=0.66, limit_l=-0.1, limit_r=1.1, init_prob=0.9):
        super().__init__()
        self.temp = temperature
        self.limit_l = limit_l
        self.limit_r = limit_r
        init_val = np.log(init_prob / (1 - init_prob))
        self.log_alpha = nn.Parameter(torch.Tensor([init_val]))

    def forward(self, x, training=True):
        if training:
            u = torch.rand_like(self.log_alpha)
            s = torch.sigmoid((torch.log(u + 1e-8) - torch.log(1 - u + 1e-8) + self.log_alpha) / self.temp)
            s = s * (self.limit_r - self.limit_l) + self.limit_l
        else:
            s = torch.sigmoid(self.log_alpha) * (self.limit_r - self.limit_l) + self.limit_l
        z = torch.clamp(s, min=0.0, max=1.0)
        return x * z

    def regularization_term(self):
        return torch.sigmoid(self.log_alpha - self.temp * np.log(-self.limit_l / self.limit_r))
    
    def get_prob(self):
        return torch.sigmoid(self.log_alpha).item()

class DualStreamInteractionLayer(nn.Module):
    def __init__(self, input_dim, num_classes, rank, poly_order):
        super().__init__()
        self.rank = rank
        self.poly_order = poly_order
        
        self.factors = nn.ModuleList()
        for i in range(1, poly_order + 1):
            order_params = nn.ParameterList([
                nn.Parameter(torch.empty(input_dim, rank, num_classes)) 
                for _ in range(i)
            ])
            self.factors.append(order_params)

        self.coeffs_pure = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, num_classes))
            for _ in range(poly_order)
        ])
        
        self.beta = nn.Parameter(torch.zeros(num_classes))
        self.reset_parameters()

    def reset_parameters(self):
        # [Fix 1] Increase Init Std from 0.01 -> 0.05 to prevent "Silent Death"
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
        projections = [torch.einsum('bd, drc -> brc', x, u) for u in factors]
        combined = projections[0]
        for p in projections[1:]:
            combined = combined * p
        return torch.sum(combined, dim=1)

class SparseSearchAgent(nn.Module):
    def __init__(self, input_dim=10, num_classes=1, rank=8, max_order=5):
        super().__init__()
        self.max_order = max_order
        self.core = DualStreamInteractionLayer(input_dim, num_classes, rank, max_order)
        self.gates_pure = nn.ModuleList([L0Gate() for _ in range(max_order)])
        self.gates_int  = nn.ModuleList([L0Gate() for _ in range(max_order)])
        # Use Identity to see raw weight effects
        self.bn_pure = nn.ModuleList(nn.Identity() for _ in range(max_order))
        self.bn_int = nn.ModuleList(nn.Identity() for _ in range(max_order))

    def forward(self, x, training=True):
        output = self.core.beta
        for i, gate in enumerate(self.gates_pure):
            term = self.core.get_pure_term(x, i)
            output = output + gate(term, training=training)
        for i, gate in enumerate(self.gates_int):
            term = self.core.get_interaction_term(x, i)
            output = output + gate(term, training=training)
        return output
    
    def calculate_regularization(self):
        reg = 0.0
        # Penalize Pure MORE to verify if we can force Interaction
        for i, g in enumerate(self.gates_pure):
            reg += 5.0 * g.regularization_term()
        for i, g in enumerate(self.gates_int):
            reg += 1.0 * g.regularization_term()
        return reg

# ==========================================
# 2. Searcher V3 (With Loss Plot & Structure Print)
# ==========================================

class FullMonitorSearcher:
    def __init__(self, input_dim, rank=8, max_order=5, epochs=200):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.max_order = max_order
        self.agent = SparseSearchAgent(input_dim=input_dim, rank=rank, max_order=max_order).to(self.device)
        
        self.logs = {
            'loss': [],
            'gate_pure': {o: [] for o in range(1, max_order+1)},
            'gate_int':  {o: [] for o in range(1, max_order+1)},
            'weight_pure': {o: [] for o in range(1, max_order+1)},
            'weight_int':  {o: [] for o in range(1, max_order+1)},
        }

    def _get_tensor_norm(self, parameter_list):
        # Mean Abs Value
        norms = [p.detach().abs().mean().item() for p in parameter_list]
        return np.mean(norms)

    def fit(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        # --- OPTIMIZER (Balanced for Survival) ---
        # [Fix 2] Reduced Weight Decay from 1e-3 to 1e-5. 
        # We want to prevent explosion, not cause extinction.
        optimizer = optim.Adam([
            {'params': self.agent.core.coeffs_pure.parameters(), 'lr': 0.005, 'weight_decay': 1e-4},
            {'params': [self.agent.core.beta], 'lr': 0.005},
            {'params': self.agent.core.factors.parameters(), 'lr': 0.02, 'weight_decay': 1e-5}, 
            {'params': list(self.agent.gates_pure.parameters()) + list(self.agent.gates_int.parameters()), 'lr': 0.05}
        ])
        
        loss_fn = nn.MSELoss()
        warmup_end = 50
        
        print(f"Starting Search V3 | Init=0.05 | L2_Int=1e-5 | Warmup={warmup_end}")

        for epoch in range(self.epochs):
            self.agent.train()
            
            # Warmup: Freeze Gates
            is_warmup = epoch < warmup_end
            for p in self.agent.gates_pure.parameters(): p.requires_grad = not is_warmup
            for p in self.agent.gates_int.parameters(): p.requires_grad = not is_warmup
            
            total_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                pred = self.agent(bx, training=True)
                mse = loss_fn(pred, by)
                
                reg = 0.0
                if not is_warmup:
                    # Lambda = 0.05
                    reg = 0.05 * self.agent.calculate_regularization()
                
                loss = mse + reg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            
            # --- MONITORING ---
            self.logs['loss'].append(total_loss / len(loader))
            
            for i in range(self.max_order):
                order = i + 1
                self.logs['gate_pure'][order].append(self.agent.gates_pure[i].get_prob())
                self.logs['gate_int'][order].append(self.agent.gates_int[i].get_prob())
                
                w_pure = self.agent.core.coeffs_pure[i].norm(p=2).item()
                self.logs['weight_pure'][order].append(w_pure)
                
                w_int = self._get_tensor_norm(self.agent.core.factors[i])
                self.logs['weight_int'][order].append(w_int)

            if epoch % 20 == 0:
                print(f"Ep {epoch} | Loss: {self.logs['loss'][-1]:.4f}")

    def get_structure_str(self):
        pure_active = []
        int_active = []
        for i in range(self.max_order):
            if self.agent.gates_pure[i].get_prob() > 0.5: pure_active.append(i+1)
            if self.agent.gates_int[i].get_prob() > 0.5: int_active.append(i+1)
        return f"Pure: {pure_active}\nInteract: {int_active}"

# ==========================================
# 3. Execution & Visualization
# ==========================================

def main():
    DIM = 50
    # Ground Truth: Interaction Order 2 Only
    print("Generating Synthetic Data (Truth: Interaction Order 2)...")
    X = torch.randn(2000, DIM)
    y = torch.zeros(2000, 1)
    
    # Truth Simulation
    rank_truth = 4
    for _ in range(rank_truth):
        u = torch.randn(DIM, 1) / np.sqrt(DIM)
        v = torch.randn(DIM, 1) / np.sqrt(DIM)
        y += (X @ u) * (X @ v)
    y = (y - y.mean()) / (y.std() + 1e-8)
    baseline_mse = torch.var(y).item()
    print(f"Baseline MSE (Variance): {baseline_mse:.4f}")

    # Run Search
    searcher = FullMonitorSearcher(DIM, rank=8, max_order=5, epochs=200)
    searcher.fit(X, y)
    
    final_struct = searcher.get_structure_str()
    print(f"\nFinal Discovered Structure:\n{final_struct}")

    # --- PLOTTING ---
    logs = searcher.logs
    epochs = range(len(logs['loss']))
    
    # Layout: Top row = Loss, Middle = Weights, Bottom = Gates
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2)
    
    # 1. LOSS (Full Width)
    ax_loss = fig.add_subplot(gs[0, :])
    ax_loss.plot(epochs, logs['loss'], 'k-', label='Total Training Loss')
    ax_loss.axhline(baseline_mse, color='gray', linestyle=':', label='Baseline (Var)')
    ax_loss.set_yscale('log')
    ax_loss.set_title(f'Training Loss (Baseline={baseline_mse:.2f})')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    
    # 2. WEIGHTS
    ax_w_pure = fig.add_subplot(gs[1, 0])
    for order in range(1, 6):
        ax_w_pure.plot(epochs, logs['weight_pure'][order], label=f'Pure {order}')
    ax_w_pure.set_title('Pure Weights (L2 Norm)')
    ax_w_pure.legend(fontsize='small')
    ax_w_pure.grid(True, alpha=0.3)

    ax_w_int = fig.add_subplot(gs[1, 1])
    for order in range(1, 6):
        width = 3 if order == 2 else 1
        ax_w_int.plot(epochs, logs['weight_int'][order], label=f'Int {order}', linewidth=width)
    ax_w_int.set_title('Interaction Weights (Mean Abs)')
    ax_w_int.legend(fontsize='small')
    ax_w_int.grid(True, alpha=0.3)
    
    # 3. GATES
    ax_g_pure = fig.add_subplot(gs[2, 0])
    for order in range(1, 6):
        ax_g_pure.plot(epochs, logs['gate_pure'][order], label=f'Pure {order}')
    ax_g_pure.set_title('Pure Gates (Prob)')
    ax_g_pure.set_ylim(-0.1, 1.1)
    ax_g_pure.grid(True, alpha=0.3)

    ax_g_int = fig.add_subplot(gs[2, 1])
    for order in range(1, 6):
        width = 3 if order == 2 else 1
        ax_g_int.plot(epochs, logs['gate_int'][order], label=f'Int {order}', linewidth=width)
    ax_g_int.set_title(f'Interact Gates (Prob)\nResult: {final_struct.replace(chr(10), " | ")}') # Replace newline with pipe
    ax_g_int.set_ylim(-0.1, 1.1)
    ax_g_int.legend(fontsize='small')
    ax_g_int.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('neuronseek_v3_monitor.png')
    print("\nPlot saved to 'neuronseek_v3_monitor.png'")

if __name__ == "__main__":
    main()