import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

# ==========================================
# 1. Data Generation (Ground Truth)
# ==========================================
class SyntheticGenerator:
    """ Generates strict Interaction-only data to test if Searcher can find it. """
    def __init__(self, n_samples=2000, input_dim=20, noise_level=0.1):
        self.n = n_samples
        self.d = input_dim
        self.noise = noise_level

    def get_data(self):
        # Data: Standard Gaussian
        X = torch.randn(self.n, self.d)
        
        # Truth: Strictly Interaction (Order 2)
        # y = sum( (X @ u) * (X @ v) )
        # We simulate a Rank-4 interaction
        y = torch.zeros(self.n, 1)
        rank_truth = 4
        for _ in range(rank_truth):
            u = torch.randn(self.d, 1) / np.sqrt(self.d)
            v = torch.randn(self.d, 1) / np.sqrt(self.d)
            term = (X @ u) * (X @ v)
            y += term
            
        # Add noise
        y = y + self.noise * torch.randn_like(y)
        
        # Standardize y (Crucial for MSE scale consistency)
        y = (y - y.mean()) / (y.std() + 1e-8)
        
        return X, y

# ==========================================
# 2. Core Modules (Provided by You)
# ==========================================
class DualStreamInteractionLayer(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, rank: int, poly_order: int):
        super().__init__()
        self.rank = rank
        self.num_classes = num_classes
        self.poly_order = poly_order
        
        # Interaction Stream (Implicit CP)
        self.factors = nn.ModuleList()
        for i in range(1, poly_order + 1):
            order_params = nn.ParameterList([
                nn.Parameter(torch.empty(input_dim, rank, num_classes)) 
                for _ in range(i)
            ])
            self.factors.append(order_params)

        # Pure Stream
        self.coeffs_pure = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, num_classes))
            for _ in range(poly_order)
        ])
        
        self.beta = nn.Parameter(torch.zeros(num_classes))
        self.register_buffer('mask_interact', torch.ones(poly_order))
        self.register_buffer('mask_pure', torch.ones(poly_order))
        self.reset_parameters()

    def reset_parameters(self):
        for order_params in self.factors:
            for p in order_params:
                # Use smaller init for interaction to simulate vanishing gradient issues
                nn.init.normal_(p, std=0.01) 
        for p in self.coeffs_pure:
            nn.init.normal_(p, std=0.01)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        logits = self.beta.unsqueeze(0).expand(batch_size, -1).clone()
        
        for i in range(self.poly_order):
            order = i + 1
            if self.mask_interact[i] == 1.0:
                logits = logits + self.get_interaction_term(x, i)
            if self.mask_pure[i] == 1.0:
                logits = logits + self.get_pure_term(x, i)
        return logits
    
    def get_pure_term(self, x: torch.Tensor, order_idx: int) -> torch.Tensor:
        order = order_idx + 1
        term = x if order == 1 else x.pow(order)
        return term @ self.coeffs_pure[order_idx]

    def get_interaction_term(self, x: torch.Tensor, order_idx: int) -> torch.Tensor:
        factors = self.factors[order_idx]
        projections = [torch.einsum('bd, drc -> brc', x, u) for u in factors]
        combined = projections[0]
        for p in projections[1:]:
            combined = combined * p
        return torch.sum(combined, dim=1)

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

class SparseSearchAgent(nn.Module):
    def __init__(self, input_dim=10, num_classes=1, rank=8, max_order=3): # Reduced max_order for clarity
        super().__init__()
        self.core = DualStreamInteractionLayer(input_dim, num_classes, rank, max_order)
        self.gates_pure = nn.ModuleList([L0Gate() for _ in range(max_order)])
        self.gates_int  = nn.ModuleList([L0Gate() for _ in range(max_order)])
        # Use Identity instead of BatchNorm to isolate gradient dynamics pure vs interact
        # (BatchNorm can mask scale issues)
        self.bn_pure = nn.ModuleList(nn.Identity() for _ in range(max_order))
        self.bn_int = nn.ModuleList(nn.Identity() for _ in range(max_order))

    def forward(self, x, training=True):
        output = self.core.beta
        for i, gate in enumerate(self.gates_pure):
            term = self.core.get_pure_term(x, i)
            term = self.bn_pure[i](term)
            output = output + gate(term, training=training)
        for i, gate in enumerate(self.gates_int):
            term = self.core.get_interaction_term(x, i)
            term = self.bn_int[i](term)
            output = output + gate(term, training=training)
        return output
    
    def calculate_regularization(self):
        reg = 0.0
        # HEAVY PENALTY for Pure to force Interaction search
        for i, g in enumerate(self.gates_pure):
            reg += 2.0 * g.regularization_term() 
        for i, g in enumerate(self.gates_int):
            reg += 1.0 * g.regularization_term()
        return reg

# ==========================================
# 3. The Modified Searcher (With Diagnostics)
# ==========================================
class NeuronSeekSearcher:
    def __init__(self, input_dim, rank=8, epochs=200, lr_pure=0.01, lr_int=0.01):
        self.epochs = epochs
        self.rank = rank
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.agent = SparseSearchAgent(input_dim=input_dim, rank=rank, max_order=3).to(self.device)
        self.lr_pure = lr_pure
        self.lr_int = lr_int
        
        # History for plotting
        self.history = {
            'loss': [], 'mse_pure_solo': [], 'mse_int_solo': [],
            'gate_pure_prob': [], 'gate_int_prob': []
        }

    def fit(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        # --- DIFFERENTIAL LEARNING RATE SETUP ---
        # We explicitly separate Interaction params to give them higher LR
        int_params = list(self.agent.core.factors.parameters())
        pure_params = list(self.agent.core.coeffs_pure.parameters()) + [self.agent.core.beta]
        gate_params = list(self.agent.gates_pure.parameters()) + list(self.agent.gates_int.parameters())
        
        optimizer = optim.Adam([
            {'params': pure_params, 'lr': self.lr_pure},   # Pure LR
            {'params': int_params,  'lr': self.lr_int},    # Interaction LR (Boosted?)
            {'params': gate_params, 'lr': 0.05}            # Gate LR
        ])
        
        loss_fn = nn.MSELoss()
        warmup_end = 50 # Protected Warmup
        
        print(f"Search Started | LR_Pure: {self.lr_pure} | LR_Int: {self.lr_int}")
        
        for epoch in range(self.epochs):
            self.agent.train()
            
            # Freeze/Unfreeze Gates based on Warmup
            is_warmup = epoch < warmup_end
            for p in gate_params: p.requires_grad = not is_warmup
            
            total_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                pred = self.agent(bx, training=True)
                mse = loss_fn(pred, by)
                
                reg = 0.0
                if not is_warmup:
                    reg = 0.1 * self.agent.calculate_regularization() # Lambda
                
                loss = mse + reg
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # --- DIAGNOSTICS (The "Race" Monitor) ---
            self.monitor_step(X_t, y_t, loss_fn)
            self.history['loss'].append(total_loss / len(loader))
            
            if epoch % 20 == 0:
                print(f"Ep {epoch} | Loss: {self.history['loss'][-1]:.4f} | "
                      f"PureMSE: {self.history['mse_pure_solo'][-1]:.4f} | "
                      f"IntMSE: {self.history['mse_int_solo'][-1]:.4f}")

    def monitor_step(self, X, y, loss_fn):
        """ Calculates MSE for each stream independently to see who is winning. """
        self.agent.eval()
        with torch.no_grad():
            # 1. Pure Only MSE
            # Manually sum only pure terms + bias
            out_pure = self.agent.core.beta
            for i in range(len(self.agent.gates_pure)):
                term = self.agent.core.get_pure_term(X, i)
                # Note: We assume gate is OPEN (1.0) to test capacity, 
                # OR we can multiply by current gate prob to test contribution.
                # Let's test CAPACITY (Potential):
                out_pure = out_pure + term 
            mse_pure = loss_fn(out_pure, y).item()
            
            # 2. Interact Only MSE
            out_int = self.agent.core.beta
            for i in range(len(self.agent.gates_int)):
                term = self.agent.core.get_interaction_term(X, i)
                out_int = out_int + term
            mse_int = loss_fn(out_int, y).item()
            
            # 3. Gate Probs (Mean across orders)
            p_pure = np.mean([g.get_prob() for g in self.agent.gates_pure])
            p_int = np.mean([g.get_prob() for g in self.agent.gates_int])
            
            self.history['mse_pure_solo'].append(mse_pure)
            self.history['mse_int_solo'].append(mse_int)
            self.history['gate_pure_prob'].append(p_pure)
            self.history['gate_int_prob'].append(p_int)

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    # Settings
    DIM = 50 # High enough to cause issues
    RANK = 8
    EPOCHS = 200
    
    # Generate Interaction-Only Data
    print("Generating Data...")
    gen = SyntheticGenerator(input_dim=DIM)
    X, y = gen.get_data()
    baseline_mse = torch.var(torch.tensor(y)).item()
    print(f"Baseline MSE: {baseline_mse:.4f}")

    # --- EXPERIMENT 1: EQUAL LEARNING RATES (The "Problem" Case) ---
    print("\n>>> Running Exp 1: Equal LR (0.01) <<<")
    searcher_bad = NeuronSeekSearcher(DIM, RANK, EPOCHS, lr_pure=0.01, lr_int=0.01)
    searcher_bad.fit(X, y)
    
    # --- EXPERIMENT 2: DIFFERENTIAL LEARNING RATES (The "Solution" Case) ---
    print("\n>>> Running Exp 2: Boosted Interaction LR (0.1) <<<")
    searcher_good = NeuronSeekSearcher(DIM, RANK, EPOCHS, lr_pure=0.005, lr_int=0.1)
    searcher_good.fit(X, y)
    
    # --- PLOTTING ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: MSE Dynamics (Bad)
    ax = axes[0, 0]
    ax.plot(searcher_bad.history['mse_pure_solo'], label='Pure Stream', color='blue', linestyle='--')
    ax.plot(searcher_bad.history['mse_int_solo'], label='Int Stream', color='red')
    ax.axhline(baseline_mse, color='gray', linestyle=':', label='Baseline')
    ax.set_title('Exp 1: Equal LR (MSE Dynamics)')
    ax.set_yscale('log')
    ax.legend()
    
    # Plot 2: Gate Dynamics (Bad)
    ax = axes[0, 1]
    ax.plot(searcher_bad.history['gate_pure_prob'], label='Pure Gate', color='blue', linestyle='--')
    ax.plot(searcher_bad.history['gate_int_prob'], label='Int Gate', color='red')
    ax.set_title('Exp 1: Gate Probabilities')
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    # Plot 3: MSE Dynamics (Good)
    ax = axes[1, 0]
    ax.plot(searcher_good.history['mse_pure_solo'], label='Pure Stream', color='blue', linestyle='--')
    ax.plot(searcher_good.history['mse_int_solo'], label='Int Stream', color='red')
    ax.axhline(baseline_mse, color='gray', linestyle=':', label='Baseline')
    ax.set_title('Exp 2: Differential LR (MSE Dynamics)')
    ax.set_yscale('log')
    ax.legend()
    
    # Plot 4: Gate Dynamics (Good)
    ax = axes[1, 1]
    ax.plot(searcher_good.history['gate_pure_prob'], label='Pure Gate', color='blue', linestyle='--')
    ax.plot(searcher_good.history['gate_int_prob'], label='Int Gate', color='red')
    ax.set_title('Exp 2: Gate Probabilities')
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('neuronseek_diagnosis.png')
    print("\nDiagnosis plot saved to 'neuronseek_diagnosis.png'")

if __name__ == "__main__":
    main()