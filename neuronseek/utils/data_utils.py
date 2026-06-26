import torch
from torch.utils.data import TensorDataset
import numpy as np

class SyntheticGenerator:
    """
    Holistic Synthetic Generator with DIAGONAL REMOVAL and STANDARDIZATION.
    """
    def __init__(self, n_samples=2500, input_dim=10, noise_level=0.01):
        self.n = n_samples
        self.d = input_dim
        self.noise = noise_level
        # [FIX 1] Initialize attribute to prevent AttributeError
        self.current_formula = "Unknown"

    def get_data(self, mode: str, variant: int = 0):
        X = torch.randn(self.n, self.d)
        
        # [FIX 2] Call the description setter
        self._set_formula_description(mode, variant)
        
        if mode == 'pure':
            y, truth = self._formula_pure(X, variant)
        elif mode == 'interact':
            y, truth = self._formula_interact(X, variant)
        elif mode == 'hybrid':
            y, truth = self._formula_hybrid(X, variant)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Add noise THEN Standardize
        y = y + self.noise * torch.randn_like(y)
        if y.std() > 1e-8:
            y = (y - y.mean()) / y.std()
        
        return TensorDataset(X, y), truth
    
    def _set_formula_description(self, mode, v):
        desc = ""
        if mode == 'pure':
            if v == 0: desc = "y = c * sum(x^2)"
            elif v == 1: desc = "y = c * sum(x^3)"
            elif v == 2: desc = "y = 10*sum(x) + 0.5*sum(x^2)"
            elif v == 3: desc = "y = 5*sum(x^2) + 0.5*sum(x^4)"
            elif v == 4: desc = "y = c * sum(x^5)"
        elif mode == 'interact':
            if v == 0: desc = "y = c * Int_Order2(X)"
            elif v == 1: desc = "y = c1*Int_Ord2 + c2*Int_Ord3"
            elif v == 2: desc = "y = 8.0*Int_Ord2 + 0.5*Int_Ord3"
            elif v == 3: desc = "y = c1*Int_Ord2 + c2*Int_Ord4"
            elif v == 4: desc = "y = c * Int_Ord4"
        elif mode == 'hybrid':
            if v == 0: desc = "y = c1*sum(x^2) + c2*Int_Ord2"
            elif v == 1: desc = "y = c1*sum(x) + c2*Int_Ord2"
            elif v == 2: desc = "y = 5*sum(x^3) + 0.5*Int_Ord2"
            elif v == 3: desc = "y = 0.5*sum(x^2) + 5.0*Int_Ord3"
            elif v == 4: desc = "Complex: Pure2 + Int3"
        
        self.current_formula = f"Mode: {mode.upper()} | Var: {v} | {desc}"

    def _get_random_coeff(self, low=0.5, high=3.0):
        mag = torch.FloatTensor(1).uniform_(low, high)
        sign = torch.randint(0, 2, (1,)).float() * 2 - 1
        return (mag * sign).item()

    def _generate_global_interaction(self, X, order):
        ws = []
        for _ in range(order):
            w = torch.randn(self.d, 1) / (self.d ** 0.5)
            ws.append(w)
            
        full_term = torch.ones(X.size(0), 1)
        for w in ws:
            full_term = full_term * (X @ w)
            
        return full_term

    def _formula_pure(self, X, v):
        truth = {'pure': [], 'interact': []}
        c1 = self._get_random_coeff(1.0, 3.0)
        if v == 0:
            y = c1 * (X**2).sum(dim=1, keepdim=True)
            truth['pure'] = [2]
        elif v == 1:
            y = c1 * (X**3).sum(dim=1, keepdim=True)
            truth['pure'] = [3]
        elif v == 2:
            y = 10.0 * X.sum(dim=1, keepdim=True) + 0.5 * (X**2).sum(dim=1, keepdim=True)
            truth['pure'] = [1, 2]
        elif v == 3:
            y = 5.0 * (X**2).sum(dim=1, keepdim=True) + 0.5 * (X**4).sum(dim=1, keepdim=True)
            truth['pure'] = [2, 4]
        elif v == 4:
            y = c1 * (X**5).sum(dim=1, keepdim=True)
            truth['pure'] = [5]
        return y, truth

    def _formula_interact(self, X, v):
        truth = {'pure': [], 'interact': []}
        c1 = self._get_random_coeff(1.0, 3.0)
        if v == 0:
            y = c1 * self._generate_global_interaction(X, 2)
            truth['interact'] = [2]
        elif v == 1:
            y = c1 * self._generate_global_interaction(X, 2) + self._get_random_coeff() * self._generate_global_interaction(X, 3)
            truth['interact'] = [2, 3]
        elif v == 2:
            y = 8.0 * self._generate_global_interaction(X, 2) + 0.5 * self._generate_global_interaction(X, 3)
            truth['interact'] = [2, 3]
        elif v == 3:
            y = c1 * self._generate_global_interaction(X, 2) + self._get_random_coeff() * self._generate_global_interaction(X, 4)
            truth['interact'] = [2, 4]
        elif v == 4:
            y = c1 * self._generate_global_interaction(X, 4)
            truth['interact'] = [4]
        return y, truth

    def _formula_hybrid(self, X, v):
        truth = {'pure': [], 'interact': []}
        c1 = self._get_random_coeff(1.0, 3.0)
        c2 = self._get_random_coeff(1.0, 3.0)
        if v == 0:
            y = c1*(X**2).sum(dim=1, keepdim=True) + c2*self._generate_global_interaction(X, 2)
            truth['pure'] = [2]; truth['interact'] = [2]
        elif v == 1:
            y = c1*X.sum(dim=1, keepdim=True) + c2*self._generate_global_interaction(X, 2)
            truth['pure'] = [1]; truth['interact'] = [2]
        elif v == 2:
            y = 5.0*(X**3).sum(dim=1, keepdim=True) + 0.5*self._generate_global_interaction(X, 2)
            truth['pure'] = [3]; truth['interact'] = [2]
        elif v == 3:
            y = 0.5*(X**2).sum(dim=1, keepdim=True) + 5.0*self._generate_global_interaction(X, 3)
            truth['pure'] = [2]; truth['interact'] = [3]
        elif v == 4:
            y = c1*(X**2).sum(dim=1, keepdim=True) + c2*self._generate_global_interaction(X, 3)
            truth['pure'] = [2]; truth['interact'] = [3]
        return y, truth

def get_synthetic_data(input_dim, n_samples=2500):
    """
    Legacy wrapper.
    [FIX 3] Changed default to 'interact' to avoid hidden Pure terms breaking the penalty logic.
    """
    gen = SyntheticGenerator(n_samples, input_dim)
    dataset, _ = gen.get_data('interact', 0) 
    X, y = dataset.tensors
    return X.numpy(), y.numpy()