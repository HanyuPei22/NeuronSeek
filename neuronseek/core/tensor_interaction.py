import torch
import torch.nn as nn
from typing import Tuple

class TensorInteractionLayer(nn.Module):
    """
    Core layer for Neuronal Formula Discovery (Stage 1).
    
    Implements a Dual-Stream architecture to distinguish between:
    1. Pure Power Terms (e.g., x^2, x^3) -> Calculated explicitly via element-wise power.
    2. Interaction Terms (e.g., x1*x2) -> Calculated implicitly via CP Decomposition.
    
    Input: [Batch, Input_Dim]
    Output: [Batch, Num_Classes] (Logits)
    """
    def __init__(self, input_dim: int, num_classes: int, rank: int, poly_order: int):
        super().__init__()
        self.poly_order = poly_order
        self.rank = rank
        self.num_classes = num_classes
        
        #Stream 1: Interaction Terms (Implicit CP) ---
        # Shared Factor Matrices U: Input_Dim -> Rank
        self.factors = nn.ModuleList()
        for i in range(1, poly_order + 1):
            order_factors = nn.ParameterList([
                nn.Parameter(torch.empty(input_dim, rank)) 
                for _ in range(i)])
            self.factors.append(order_factors)

        # Interaction Coefficients K: Rank -> Num_Classes
        self.coeffs_interact = nn.ParameterList([
            nn.Parameter(torch.empty(rank, num_classes)) 
            for _ in range(poly_order)
        ])
        
        # Stream 2: Pure Power Terms (Explicit) ---
        # Pure Coefficients W: Input_Dim -> Num_Classes
        self.coeffs_pure = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, num_classes))
            for _ in range(poly_order)
        ])

        self.beta = nn.Parameter(torch.empty(num_classes))
        
        # Masks for pruning (1.0 = Active, 0.0 = Pruned)
        self.register_buffer('mask_interact', torch.ones(poly_order))
        self.register_buffer('mask_pure', torch.ones(poly_order))

    def reset_parameters(self):
            """
            Initialize parameters using standard methods (Xavier/Glorot).
            This helps prevent vanishing gradients in high-order terms.
            """
            for order_factors in self.factors:
                for factor in order_factors:
                    nn.init.xavier_uniform_(factor)

            for coeff in self.coeffs_interact:
                nn.init.xavier_uniform_(coeff)

            for coeff in self.coeffs_pure:
                nn.init.xavier_uniform_(coeff)

            nn.init.zeros_(self.beta)

    def _compute_cp_features(self, x: torch.Tensor, order_idx: int) -> torch.Tensor:
        """
        Computes interaction features in the latent rank space.
        Output: [Batch, Rank]
        """
        order_factors = self.factors[order_idx]
        # [Batch, D] @ [D, R] -> [Batch, R]
        projections = [x @ u for u in order_factors]
        # K*[Batch,R] -> [K,Batch,R]
        combined = torch.stack(projections, dim=0).prod(dim=0)
        return combined

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        # Initialize logits with bias: [Batch, Num_Classes]
        logits = self.beta.unsqueeze(0).expand(batch_size, -1).clone()
        reg_loss = torch.tensor(0.0, device=x.device)

        for i in range(self.poly_order):
            order = i + 1
            
            # --- Path A: Interaction Terms (CP) ---
            if self.mask_interact[i] == 1.0:
                # 1. Compute latent interaction features [Batch, Rank]
                interact_feats = self._compute_cp_features(x, i)
                
                # 2. Map to logits via coefficients [Rank, Num_Classes]
                term_logits = interact_feats @ self.coeffs_interact[i]
                logits = logits + term_logits
                
                # Reg: Group Lasso on Interaction Matrix
                reg_loss = reg_loss + torch.norm(self.coeffs_interact[i], p='fro')

            # --- Path B: Pure Power Terms (Explicit) ---
            if self.mask_pure[i] == 1.0:
                # 1. Compute explicit power term
                if order == 1:
                    term_pure = x
                else:
                    term_pure = x.pow(order)
                
                # 2. Map to logits via channel-wise weights [Input_Dim, Num_Classes]
                term_logits = term_pure @ self.coeffs_pure[i]
                logits = logits + term_logits
                
                # Reg: Group Lasso on Pure Weight Matrix
                reg_loss = reg_loss + torch.norm(self.coeffs_pure[i], p='fro')

        return logits, reg_loss