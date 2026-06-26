import torch
import torch.nn as nn

class DualStreamInteractionLayer(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, rank: int, poly_order: int):
        super().__init__()
        self.rank = rank
        self.num_classes = num_classes
        self.poly_order = poly_order
        
        # --- 1. Interaction Stream Parameters (Strict CP) ---
        # Storage: ModuleList[Order] -> ParameterList[Term] -> Tensor[D, R, C]
        self.factors = nn.ModuleList()
        for i in range(1, poly_order + 1):
            order_params = nn.ParameterList([
                nn.Parameter(torch.empty(input_dim, rank, num_classes)) 
                for _ in range(i) # Order i needs i factors
            ])
            self.factors.append(order_params)

        # --- 2. Pure Stream Parameters ---
        # Storage: ParameterList[Order] -> Tensor[D, C]
        self.coeffs_pure = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, num_classes))
            for _ in range(poly_order)
        ])
        
        # Global Bias per class
        self.beta = nn.Parameter(torch.zeros(num_classes))
        
        # Masks for gating (Controlled by external Agent)
        self.register_buffer('mask_interact', torch.ones(poly_order))
        self.register_buffer('mask_pure', torch.ones(poly_order))
        
        self.reset_parameters()

    def reset_parameters(self):
        for order_params in self.factors:
            for p in order_params:
                nn.init.normal_(p, std = 0.05)
        for p in self.coeffs_pure:
            nn.init.normal_(p, std = 0.05)

    def forward(self, x: torch.Tensor):
        """
        Input: [Batch, D]
        Output: [Batch, Num_Classes]
        """
        batch_size = x.size(0)
        logits = self.beta.unsqueeze(0).expand(batch_size, -1).clone()
        
        for i in range(self.poly_order):
            order = i + 1
            
            # --- Stream A: Interaction (Implicit CP) ---
            if self.mask_interact[i] == 1.0:
                # Retrieve the list of 'order' factor tensors
                factors = self.factors[i] 
                
                # 1. Parallel Projection: x[B,D] @ u[D,R,C] -> [B,R,C]
                projections = [torch.einsum('bd, drc -> brc', x, u) for u in factors]
                
                # 2. Element-wise Product across terms
                combined = projections[0]
                for p in projections[1:]:
                    combined = combined * p
                
                # 3. Summation over Rank (Strict CP aggregation)
                # [Batch, Rank, Class] -> [Batch, Class]
                logits = logits + torch.sum(combined, dim=1)

            # --- Stream B: Pure Power Terms ---
            if self.mask_pure[i] == 1.0:
                term = x if order == 1 else x.pow(order)
                
                # Projection: [Batch, D] @ [D, C] -> [Batch, C]
                logits = logits + (term @ self.coeffs_pure[i])
                
        return logits
    
    def get_pure_term(self, x: torch.Tensor, order_idx: int) -> torch.Tensor:
        """
        Computes the Pure Term (explicit polynomial term) for a specific order.
        order_idx: 0 corresponds to order=1, 1 corresponds to order=2...
        Returns: [Batch, Num_Classes]
        """
        order = order_idx + 1
        # Compute x^k
        term = x if order == 1 else x.pow(order)
        # Projection: [B, D] @ [D, C] -> [B, C]
        return term @ self.coeffs_pure[order_idx]

    def get_interaction_term(self, x: torch.Tensor, order_idx: int) -> torch.Tensor:
        """
        Computes the Interaction Term (CP Decomposition term) for a specific order.
        Returns: [Batch, Num_Classes]
        """
        factors = self.factors[order_idx]
        
        # 1. Parallel Projection: [B, D] @ [D, R, C] -> [B, R, C]
        projections = [torch.einsum('bd, drc -> brc', x, u) for u in factors]
        
        # 2. Product
        combined = projections[0]
        for p in projections[1:]:
            combined = combined * p
        
        # 3. Summation (Strict CP Aggregation) -> [B, C]
        return torch.sum(combined, dim=1)