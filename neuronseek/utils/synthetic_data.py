import torch
from torch.utils.data import TensorDataset
import numpy as np  # 确保导入 numpy

class SyntheticGenerator:
    """
    Holistic Synthetic Generator with DIAGONAL REMOVAL and STANDARDIZATION.
    1. Removes x^2 terms from interaction generation to ensure strict disentanglement.
    2. Standardizes y to unit variance so reg_lambda is consistent across orders.
    """
    def __init__(self, n_samples=2500, input_dim=10, noise_level=0.01):
        self.n = n_samples
        self.d = input_dim
        self.noise = noise_level

    def get_data(self, mode: str, variant: int = 0):
        X = torch.randn(self.n, self.d)
        
        if mode == 'pure':
            y, truth = self._formula_pure(X, variant)
        elif mode == 'interact':
            y, truth = self._formula_interact(X, variant)
        elif mode == 'hybrid':
            y, truth = self._formula_hybrid(X, variant)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # [CRITICAL FIX] Add noise THEN Standardize y
        # This ensures Loss is always roughly in the same scale (e.g., ~1.0)
        # regardless of whether it's x^2 or x^5.
        y = y + self.noise * torch.randn_like(y)
        if y.std() > 1e-8:
            y = (y - y.mean()) / y.std()
        
        return TensorDataset(X, y), truth

    def _get_random_coeff(self, low=0.5, high=3.0):
        mag = torch.FloatTensor(1).uniform_(low, high)
        sign = torch.randint(0, 2, (1,)).float() * 2 - 1
        return (mag * sign).item()

    def _generate_global_interaction(self, X, order):
        """
        Generates global interactions via random projections.
        """
        # 1. Generate random projection weights
        ws = []
        for _ in range(order):
            # Xavier-like initialization for projection weights
            w = torch.randn(self.d, 1) / (self.d ** 0.5)
            ws.append(w)
            
        # 2. Compute full product: (Xw1) * (Xw2) * ...
        full_term = torch.ones(X.size(0), 1)
        for w in ws:
            full_term = full_term * (X @ w)
            
        return full_term

    # --- Formulas ---

    def _formula_pure(self, X, v):
        truth = {'pure': [], 'interact': []}
        c1 = self._get_random_coeff(1.0, 3.0)
        
        if v == 0: # Quadratic
            y = c1 * (X**2).sum(dim=1, keepdim=True)
            truth['pure'] = [2]
        elif v == 1: # Cubic
            y = c1 * (X**3).sum(dim=1, keepdim=True)
            truth['pure'] = [3]
        elif v == 2: # Linear + Quadratic
            y = 10.0 * X.sum(dim=1, keepdim=True) + 0.5 * (X**2).sum(dim=1, keepdim=True)
            truth['pure'] = [1, 2]
        elif v == 3: # Quadratic + Quartic
            y = 5.0 * (X**2).sum(dim=1, keepdim=True) + 0.5 * (X**4).sum(dim=1, keepdim=True)
            truth['pure'] = [2, 4]
        elif v == 4: # High Order
            y = c1 * (X**5).sum(dim=1, keepdim=True)
            truth['pure'] = [5]
        return y, truth

    def _formula_interact(self, X, v):
        truth = {'pure': [], 'interact': []}
        c1 = self._get_random_coeff(1.0, 3.0)
        
        if v == 0: # Strict 2nd Order
            y = c1 * self._generate_global_interaction(X, 2)
            truth['interact'] = [2]
        elif v == 1: # Mixed 2+3
            y = c1 * self._generate_global_interaction(X, 2) + self._get_random_coeff() * self._generate_global_interaction(X, 3)
            truth['interact'] = [2, 3]
        elif v == 2: # 2nd + 3rd (Dynamic Range)
            y = 8.0 * self._generate_global_interaction(X, 2) + 0.5 * self._generate_global_interaction(X, 3)
            truth['interact'] = [2, 3]
        elif v == 3: # 2nd + 4th
            y = c1 * self._generate_global_interaction(X, 2) + self._get_random_coeff() * self._generate_global_interaction(X, 4)
            truth['interact'] = [2, 4]
        elif v == 4: # 4th Order
            y = c1 * self._generate_global_interaction(X, 4)
            truth['interact'] = [4]
        return y, truth

    def _formula_hybrid(self, X, v):
        truth = {'pure': [], 'interact': []}
        c1 = self._get_random_coeff(1.0, 3.0)
        c2 = self._get_random_coeff(1.0, 3.0)
        
        if v == 0: # Pure 2 + Int 2
            y = c1*(X**2).sum(dim=1, keepdim=True) + c2*self._generate_global_interaction(X, 2)
            truth['pure'] = [2]; truth['interact'] = [2]
        elif v == 1: # Pure 1 + Int 2
            y = c1*X.sum(dim=1, keepdim=True) + c2*self._generate_global_interaction(X, 2)
            truth['pure'] = [1]; truth['interact'] = [2]
        elif v == 2: # Pure 3 + Int 2
            y = 5.0*(X**3).sum(dim=1, keepdim=True) + 0.5*self._generate_global_interaction(X, 2)
            truth['pure'] = [3]; truth['interact'] = [2]
        elif v == 3: # Pure 2 + Int 3
            y = 0.5*(X**2).sum(dim=1, keepdim=True) + 5.0*self._generate_global_interaction(X, 3)
            truth['pure'] = [2]; truth['interact'] = [3]
        elif v == 4: # Complex
            y = c1*(X**2).sum(dim=1, keepdim=True) + c2*self._generate_global_interaction(X, 3)
            truth['pure'] = [2]; truth['interact'] = [3]
        return y, truth

def get_synthetic_data(input_dim, n_samples=2500):
    """
    Legacy wrapper for compatibility with experiments.
    """
    gen = SyntheticGenerator(n_samples, input_dim)
    dataset, _ = gen.get_data('hybrid', 0)
    X, y = dataset.tensors
    return X.numpy(), y.numpy()