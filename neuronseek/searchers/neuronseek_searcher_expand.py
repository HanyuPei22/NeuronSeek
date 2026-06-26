import torch
import torch.nn as nn
import numpy as np
from neuronseek.searchers.neuronseek_searcher import NeuronSeekSearcher

class NeuronSeekSearcherExpand(NeuronSeekSearcher):
    """
    Extended NeuronSeek searcher with logging and plotting capabilities.
    """
    def __init__(self, input_dim, num_classes=1, rank=8, epochs=200, batch_size=64, reg_lambda=0.05):
        super().__init__(input_dim, num_classes, rank, epochs, batch_size, reg_lambda)
        
        # Log containers
        self.logs = {
            'loss': [],
            'lambda_val': [],
            'gate_pure': {o: [] for o in range(1, 6)},
            'gate_int':  {o: [] for o in range(1, 6)},
            'weight_pure': {o: [] for o in range(1, 6)},
            'weight_int':  {o: [] for o in range(1, 6)},
        }

    def _get_tensor_norm(self, parameter_list):
        # Mean absolute value of tensors
        norms = [p.detach().abs().mean().item() for p in parameter_list]
        return np.mean(norms)

    def fit(self, X: np.ndarray, y: np.ndarray, baseline_mse: float = 1.0, callback = None) -> None:
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # V5 Optimizer Settings (Reduced Int LR for stability)
        optimizer = torch.optim.Adam([
            {'params': self.agent.core.coeffs_pure.parameters(), 'lr': 0.005, 'weight_decay': 1e-4},
            {'params': [self.agent.core.beta], 'lr': 0.005},
            {'params': self.agent.core.factors.parameters(), 'lr': 0.01, 'weight_decay': 1e-5}, 
            {'params': list(self.agent.bn_pure.parameters()) + list(self.agent.bn_int.parameters()), 'lr': 0.01},
            {'params': list(self.agent.gates_pure.parameters()) + list(self.agent.gates_int.parameters()), 'lr': 0.01}
        ])
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)
        loss_fn = nn.MSELoss()
        
        warmup_end = int(self.epochs * 0.25)
        anneal_end = int(self.epochs * 0.75)
        max_lambda = self.reg_lambda

        print(f"Starting Diagnostic Search | Rank={self.rank} | Lambda={max_lambda}")

        self.agent.train()
        for epoch in range(self.epochs):
            # Annealing logic
            current_lambda = 0.0
            if epoch >= warmup_end:
                if epoch < anneal_end:
                    progress = (epoch - warmup_end) / (anneal_end - warmup_end)
                    current_lambda = max_lambda * progress
                else:
                    current_lambda = max_lambda
            self.logs['lambda_val'].append(current_lambda)

            # Freeze logic
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
            
            # --- Logging ---
            avg_loss = total_loss / len(loader)
            self.logs['loss'].append(avg_loss)
            
            for i in range(self.agent.max_order):
                order = i + 1
                # Track Gates
                self.logs['gate_pure'][order].append(self.agent.gates_pure[i].get_prob())
                self.logs['gate_int'][order].append(self.agent.gates_int[i].get_prob())
                # Track Weights
                w_pure = self.agent.core.coeffs_pure[i].norm(p=2).item()
                self.logs['weight_pure'][order].append(w_pure)
                w_int = self._get_tensor_norm(self.agent.core.factors[i])
                self.logs['weight_int'][order].append(w_int)

            if callback:
                if callback(epoch=epoch, model=self.agent, loss=avg_loss):
                    print(f"[NeuronSeek] Stopped by callback at epoch {epoch}")
                    break

            if (epoch + 1) % 20 == 0:
                print(f"Ep {epoch+1} | Loss: {avg_loss:.4f} | Lambda: {current_lambda:.4f}")

    def plot_diagnostics(self, title="NeuronSeek Diagnostics", filename="diagnostic_plot.png", baseline_mse=1.0):
        """Generates a 5-panel diagnostic plot."""
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt

        logs = self.logs
        epochs = range(len(logs['loss']))
        
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.2)
        
        # 1. Loss
        ax_loss = fig.add_subplot(gs[0, :])
        ax_loss.plot(epochs, logs['loss'], 'k-', label='Total Loss')
        ax_loss.axhline(baseline_mse, color='red', linestyle='--', label=f'Baseline (Var): {baseline_mse:.4f}')
        ax_loss.set_yscale('log')
        ax_loss.set_title(f'Training Loss (Final={logs["loss"][-1]:.4f})')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
        
        # 2. Pure Weights
        ax_wp = fig.add_subplot(gs[1, 0])
        for order in range(1, 6):
            ax_wp.plot(epochs, logs['weight_pure'][order], label=f'Pure {order}')
        ax_wp.set_title('Pure Stream Weights (L2 Norm)')
        ax_wp.legend(fontsize='small')
        ax_wp.grid(True, alpha=0.3)
        
        # 3. Int Weights
        ax_wi = fig.add_subplot(gs[1, 1])
        for order in range(1, 6):
            lw = 2 if order == 2 else 1
            ax_wi.plot(epochs, logs['weight_int'][order], label=f'Int {order}', linewidth=lw)
        ax_wi.set_title('Interaction Stream Weights (Mean Abs)')
        ax_wi.legend(fontsize='small')
        ax_wi.grid(True, alpha=0.3)
        
        # 4. Pure Gates
        ax_gp = fig.add_subplot(gs[2, 0])
        for order in range(1, 6):
            ax_gp.plot(epochs, logs['gate_pure'][order], label=f'Pure {order}')
        ax_gp.set_title('Pure Gates (Prob)')
        ax_gp.set_ylim(-0.1, 1.1)
        ax_gp.grid(True, alpha=0.3)
        
        # 5. Int Gates
        ax_gi = fig.add_subplot(gs[2, 1])
        # Overlay Lambda scaled
        if self.reg_lambda > 0:
            ax_gi.plot(epochs, [l/self.reg_lambda for l in logs['lambda_val']], 'k:', alpha=0.3, label='Lambda (Scaled)')
        
        for order in range(1, 6):
            lw = 2 if order == 2 else 1
            ax_gi.plot(epochs, logs['gate_int'][order], label=f'Int {order}', linewidth=lw)
        
        # Title with structure result
        p, i = self.agent.get_structure()
        struct_str = f"Pure: {p} | Int: {i}"
        
        ax_gi.set_title(f'Interaction Gates (Prob)\nResult: {struct_str}')
        ax_gi.set_ylim(-0.1, 1.1)
        ax_gi.legend(fontsize='small')
        ax_gi.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.savefig(filename)
        print(f"\n[Diagnostic] Plot saved to {filename}")
        plt.close()


# Backward compatibility alias
DiagnosticNeuronSeekSearcher = NeuronSeekSearcherExpand
