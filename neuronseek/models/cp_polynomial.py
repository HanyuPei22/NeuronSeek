import torch
import torch.nn as nn

class CPPolynomialTerm(nn.Module):
    """
    Implements Rank-R CP Decomposition for High-Order Polynomials.
    Formula: Sum_r [ (X @ U1_r) * (X @ U2_r) * ... ]
    """
    def __init__(self, in_dim, rank, order):
        super().__init__()
        self.in_dim = in_dim
        self.rank = rank
        self.order = order
        
        # Factors: order list of (in_dim, rank) matrices
        # Initialization with small variance is crucial for soft start
        self.factors = nn.ParameterList([
            nn.Parameter(torch.randn(in_dim, rank) * 0.01) 
            for _ in range(order)
        ])
    
    def forward(self, x):
        # 1. Projections: [X @ U1, X @ U2, ...]
        # Each projection is [Batch, Rank]
        projections = [x @ U for U in self.factors]
        
        # 2. Element-wise Product (Hadamard Product) across orders
        result = projections[0]
        for p in projections[1:]:
            result = result * p
            
        return result # Output shape: [Batch, Rank]

class RegressionHead(nn.Module):
    """Aggregates rank features to a scalar."""
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        
    def forward(self, x):
        return self.linear(x)

class ClassificationHead(nn.Module):
    """Aggregates rank features to logits."""
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.linear(x)