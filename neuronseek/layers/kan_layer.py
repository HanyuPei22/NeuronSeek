"""
Kolmogorov-Arnold Network Layer.

KAN replaces fixed activations with learnable univariate functions (B-splines).
Each edge has its own activation: y_j = sum_i phi_ij(x_i)

Reference: Liu et al. "KAN: Kolmogorov-Arnold Networks" (2024)
"""

import torch
import torch.nn as nn
import math
from typing import List


class KANLayer(nn.Module):
    """
    Single KAN layer with B-spline basis functions.

    Each input-output connection has a learnable spline function.
    Output: y_j = sum_i [ w_ij * silu(x_i) + spline_ij(x_i) ]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple = (-2.0, 2.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # B-spline needs (grid_size + spline_order) control points
        n_coeffs = grid_size + spline_order

        # Learnable spline coefficients: [out, in, n_coeffs]
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, n_coeffs) * 0.1
        )

        # Base weight for residual connection (SiLU activation)
        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) / math.sqrt(in_features)
        )

        # Grid points for B-spline (extended for boundary handling)
        h = (grid_range[1] - grid_range[0]) / grid_size
        extended_grid = torch.linspace(
            grid_range[0] - h * spline_order,
            grid_range[1] + h * spline_order,
            grid_size + 2 * spline_order + 1
        )
        self.register_buffer('grid', extended_grid)

    def forward(self, x):
        # x: [B, in_features]

        # 1. Base function: SiLU with linear weights
        base_out = torch.nn.functional.silu(x) @ self.base_weight.T  # [B, out]

        # 2. Spline function
        spline_out = self._compute_spline(x)  # [B, out]

        return base_out + spline_out

    def _compute_spline(self, x):
        """Compute B-spline output using Cox-de Boor recursion."""
        # Compute B-spline basis values
        # bases: [B, in_features, n_coeffs]
        bases = self._bspline_basis(x)

        # Weighted sum: spline_weight[out, in, coeffs] @ bases[B, in, coeffs]
        # Result: [B, out]
        spline_out = torch.einsum('oic,bic->bo', self.spline_weight, bases)

        return spline_out

    def _bspline_basis(self, x):
        """
        Compute B-spline basis functions of order k.
        Uses iterative Cox-de Boor formula for numerical stability.
        """
        # x: [B, in] -> expand to [B, in, 1] for broadcasting
        x = x.unsqueeze(-1)
        grid = self.grid  # [G] where G = grid_size + 2*order + 1

        k = self.spline_order
        n_basis = self.grid_size + k  # Number of basis functions

        # Order 0: indicator functions
        # bases[b, i, j] = 1 if grid[j] <= x[b,i] < grid[j+1]
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()  # [B, in, G-1]

        # Iteratively compute higher orders
        for p in range(1, k + 1):
            # Left term: (x - t_j) / (t_{j+p} - t_j) * B_{j,p-1}
            left_num = x - grid[:-(p+1)].unsqueeze(0).unsqueeze(0)
            left_den = grid[p:-1] - grid[:-(p+1)]
            left_den = left_den.unsqueeze(0).unsqueeze(0).clamp(min=1e-8)
            left = left_num / left_den * bases[..., :-1]

            # Right term: (t_{j+p+1} - x) / (t_{j+p+1} - t_{j+1}) * B_{j+1,p-1}
            right_num = grid[(p+1):].unsqueeze(0).unsqueeze(0) - x
            right_den = grid[(p+1):] - grid[1:-p]
            right_den = right_den.unsqueeze(0).unsqueeze(0).clamp(min=1e-8)
            right = right_num / right_den * bases[..., 1:]

            bases = left + right

        # Trim to required number of basis functions
        return bases[..., :n_basis]


class KAN(nn.Module):
    """
    Multi-layer Kolmogorov-Arnold Network.

    Args:
        layer_dims: List of dimensions [in, hidden1, hidden2, ..., out]
        grid_size: Number of grid intervals for splines
        spline_order: B-spline order (3 = cubic)
    """

    def __init__(
        self,
        layer_dims: List[int],
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(
                KANLayer(
                    layer_dims[i],
                    layer_dims[i+1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
