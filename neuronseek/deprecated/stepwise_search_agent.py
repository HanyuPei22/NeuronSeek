import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from neuronseek.core.tensor_interaction import TensorInteractionLayer

class StepwiseSearchAgent(nn.Module):
    """
    Agent performing Constructive Stepwise Structure Discovery.
    
    Mechanism:
    1. Forward Selection: Iteratively probes Pure vs Interaction terms order-by-order.
    2. Competition: Explicitly compares validation gain of Pure vs Interact terms using BIC.
    3. Backward Pruning: Checks for redundancy in existing terms after every acceptance.
    4. Metric: Uses BIC (Bayesian Information Criterion) to strictly penalize model complexity.
    """
    def __init__(self, input_dim=512, num_classes=1, rank=32, max_order=5, n_samples=2500):
        super().__init__()
        # Core layer initialized with all masks closed (0.0) initially via _update_masks
        self.core = TensorInteractionLayer(input_dim, num_classes, rank, max_order)
        self.criterion = nn.MSELoss()
        
        self.max_order = max_order
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.rank = rank
        
        # Track active terms
        self.active_pure = set()     
        self.active_interact = set() 
        self.best_bic = float('inf')

        # Initialize masks to 0
        self._update_masks()

    def fit_stepwise(self, train_loader, val_loader):
        """
        Main entry point for the stepwise search process.
        """
        print(f"--- Starting Stepwise Search (Max Order={self.max_order}) ---")

        LR_PURE = 0.005
        LR_INTERACT = 0.02
        
        # 1. Baseline: Train Bias only
        mse, bic = self._evaluate_configuration(self.active_pure, self.active_interact, train_loader, val_loader, epochs=10,lr_pure=LR_PURE, lr_int=LR_INTERACT)
        self.best_bic = bic
        print(f"[Step 0] Baseline (Bias): MSE={mse:.5f}, BIC={bic:.2f}")

        # 2. Forward Loop
        for order in range(1, self.max_order + 1):
            print(f"\n[Step {order}] Scanning Order {order}...")
            
            # --- Probe A: Try adding Pure Term ---
            p_cand = self.active_pure | {order}
            _, bic_p = self._evaluate_configuration(p_cand, self.active_interact, train_loader, val_loader, lr_pure=LR_PURE, lr_int=LR_INTERACT)
            
            # --- Probe B: Try adding Interaction Term ---
            i_cand = self.active_interact | {order}
            _, bic_i = self._evaluate_configuration(self.active_pure, i_cand, train_loader, val_loader, lr_pure=LR_PURE, lr_int=LR_INTERACT)
            
            print(f"Probe Pure: BIC {bic_p:.2f} | Probe Int: BIC {bic_i:.2f} (Current Best: {self.best_bic:.2f})")
            
            # --- Decision ---
            candidates = []
            if bic_p < self.best_bic: candidates.append(('pure', bic_p, p_cand, self.active_interact))
            if bic_i < self.best_bic: candidates.append(('interact', bic_i, self.active_pure, i_cand))
            
            if candidates:
                # Pick winner based on lowest BIC
                winner = min(candidates, key=lambda x: x[1])
                w_type, w_bic, w_p_set, w_i_set = winner
                
                print(f">>> ACCEPT: Adding {w_type.upper()} {order} (BIC Drop: {self.best_bic - w_bic:.2f})")
                self.active_pure = w_p_set
                self.active_interact = w_i_set
                self.best_bic = w_bic
                
                # Commit: Retrain permanently to update weights for next steps
                self._commit_training(train_loader,lr_pure=0.002, lr_int=0.01)
                
                # --- Backward Pruning (Redundancy Check) ---
                self._backward_pruning(train_loader, val_loader, current_order=order,lr_pure=LR_PURE, lr_int=LR_INTERACT)
                
            else:
                print(f">>> REJECT: No improvement in BIC. Order {order} is noise/redundant.")

        return self.active_pure, self.active_interact

    def _evaluate_configuration(self, p_set, i_set, train_loader, val_loader, epochs=15,lr_pure = 0.005,lr_int = 0.02):
        """
        Temporarily trains a specific subset config and returns metrics.
        Crucial: Uses weight snapshots to avoid polluting the main model during probing.
        """
        state = copy.deepcopy(self.state_dict())
        
        # Set masks & Initialize new terms (Warm Start for others)
        self._update_masks(p_set, i_set)
        self._init_new_terms(p_set, i_set)
        optimizer = optim.Adam([
                    {'params': self.core.coeffs_pure.parameters(), 'lr': lr_pure},
                    {'params': self.core.coeffs_interact.parameters(), 'lr': lr_int},
                    {'params': self.core.factors.parameters(), 'lr': lr_int}
                ])

        self.train()
        for _ in range(epochs):
            for X, y in train_loader:
                X, y = X.to(next(self.parameters()).device), y.to(next(self.parameters()).device)
                optimizer.zero_grad()
                logits, _ = self.core(X)
                loss = self.criterion(logits, y)
                loss.backward()
                self._mask_gradients() 
                optimizer.step()
        
        # Validate
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(next(self.parameters()).device), y.to(next(self.parameters()).device)
                logits, _ = self.core(X)
                total_loss += self.criterion(logits, y).item()
        mse = total_loss / len(val_loader)
        
        # Calculate BIC
        k = self._calculate_complexity(p_set, i_set)
        # BIC = n * ln(MSE) + k * ln(n)
        bic = self.n_samples * np.log(max(mse, 1e-9)) + k * np.log(self.n_samples)
        
        # Restore weights
        self.load_state_dict(state)
        
        return mse, bic

    def _backward_pruning(self, train_loader, val_loader, current_order, lr_pure, lr_int):
        """Checks if any EXISTING term has become redundant after adding a new one."""
        print("   [Backward Check] Checking redundancy...")
        improved = True
        
        while improved:
            improved = False
            # Check Pure terms
            for p in list(self.active_pure):
                if p == current_order: continue # Don't prune just added term
                
                test_p = self.active_pure - {p}
                _, bic = self._evaluate_configuration(test_p, self.active_interact, train_loader, val_loader, epochs=10,lr_pure=lr_pure,lr_int=lr_int)
                
                if bic < self.best_bic:
                    print(f"      >>> PRUNE: Pure Order {p} is redundant (BIC -> {bic:.2f})")
                    self.active_pure = test_p
                    self.best_bic = bic
                    self._commit_training(train_loader,lr_pure=0.002, lr_int=0.01)
                    improved = True
                    break 

            if improved: continue

            # Check Interact terms
            for i in list(self.active_interact):
                if i == current_order: continue
                
                test_i = self.active_interact - {i}
                _, bic = self._evaluate_configuration(self.active_pure, test_i, train_loader, val_loader, epochs=10,lr_pure=lr_pure, lr_int=lr_int)
                
                if bic < self.best_bic:
                    print(f"      >>> PRUNE: Interact Order {i} is redundant (BIC -> {bic:.2f})")
                    self.active_interact = test_i
                    self.best_bic = bic
                    self._commit_training(train_loader,lr_pure=0.002, lr_int=0.01)
                    improved = True
                    break

    def _commit_training(self, train_loader, epochs=20, lr_pure = 0.005, lr_int = 0.02):
        """Updates weights permanently after a decision."""
        self._update_masks(self.active_pure, self.active_interact)

        optimizer = optim.Adam([
                    {'params': self.core.coeffs_pure.parameters(), 'lr': lr_pure},
                    {'params': self.core.coeffs_interact.parameters(), 'lr': lr_int},
                    {'params': self.core.factors.parameters(), 'lr': lr_int}
                ])
        
        self.train()
        for _ in range(epochs):
            for X, y in train_loader:
                X, y = X.to(next(self.parameters()).device), y.to(next(self.parameters()).device)
                optimizer.zero_grad()
                logits, _ = self.core(X)
                loss = self.criterion(logits, y)
                loss.backward()
                self._mask_gradients()
                optimizer.step()

    def _calculate_complexity(self, p_set, i_set):
            """
            Calculates 'Effective Complexity' based on Group Lasso theory (Yuan & Lin, 2006).
            
            Instead of summing raw parameter counts (which punishes CP decomposition too hard),
            we use the SQUARE ROOT of the parameter count for each group.
            
            Formula: k_eff = sum(sqrt(p_g)) for active groups g.
            """
            k_eff = 0.0
            
            group_size_pure = self.input_dim
            k_eff += len(p_set) * np.sqrt(group_size_pure)

            group_size_interact = (2 * self.input_dim * self.rank) + self.rank
            k_eff += len(i_set) * np.sqrt(group_size_interact)
        
            return k_eff

    def _update_masks(self, p_set=None, i_set=None):
        if p_set is None: p_set = self.active_pure
        if i_set is None: i_set = self.active_interact
        
        for i in range(self.max_order):
            self.core.mask_pure[i] = 1.0 if (i + 1) in p_set else 0.0
            self.core.mask_interact[i] = 1.0 if (i + 1) in i_set else 0.0

    def _init_new_terms(self, p_set, i_set):

        # initial coefficients/ coeff tensor K
        for p in p_set:
            if p not in self.active_pure:
                nn.init.xavier_normal_(self.core.coeffs_pure[p- 1], gain=1.0)
        for i in i_set:
            if i not in self.active_interact:
                nn.init.xavier_normal_(self.core.coeffs_interact[i-1], gain=1.5)

                # initialize factor tensor U and V
                for f in self.core.factors[i-1]:
                    nn.init.xavier_normal_(f, gain=1.5)


    def _mask_gradients(self):
        for i in range(self.max_order):
            if self.core.mask_pure[i] == 0 and self.core.coeffs_pure[i].grad is not None:
                self.core.coeffs_pure[i].grad.fill_(0.0)
            
            if self.core.mask_interact[i] == 0:
                if self.core.coeffs_interact[i].grad is not None:
                    self.core.coeffs_interact[i].grad.fill_(0.0)
                for f in self.core.factors[i]:
                    if f.grad is not None: f.grad.fill_(0.0)