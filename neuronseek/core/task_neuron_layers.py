import torch
import torch.nn as nn

class PolynomialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 bias=False, structure={'pure': [1], 'interact': []}, rank=16):
        super().__init__()
        self.structure = structure
        self.stride = stride
        self.rank = rank
        
        # 1. Linear Term (Standard Conv) - Always handle order 1 here if present
        if 1 in structure.get('pure', []) or 1 in structure.get('interact', []):
            self.conv_linear = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            self.bn_linear = nn.BatchNorm2d(out_channels)
            
        # 2. High Order Terms Container
        self.poly_terms = nn.ModuleDict()
        
        # --- Path A: Pure Power Terms (x^n) ---
        for order in structure.get('pure', []):
            if order == 1: continue
            # Logic: x -> x^n -> Conv1x1 -> Out
            # We use a  bottleneck to save params if needed, or direct mapping
            # Here we use direct mapping: Element-wise Power -> Conv
            self.poly_terms[f'pure_{order}_conv'] = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
            self.poly_terms[f'pure_{order}_bn'] = nn.BatchNorm2d(out_channels)

        # --- Path B: Interaction Terms (CP) ---
        for order in structure.get('interact', []):
            if order == 1: continue # Order 1 interaction is just linear
            
            # Projections: Input -> Rank (Bottleneck)
            # We need 'order' separate projections
            projections = nn.ModuleList([
                nn.Conv2d(in_channels, rank, kernel_size=1, stride=1, bias=False)
                for _ in range(order)
            ])
            self.poly_terms[f'int_{order}_projs'] = projections
            
            # Mapping: Rank -> Out
            self.poly_terms[f'int_{order}_out'] = nn.Conv2d(rank, out_channels, kernel_size, stride, padding, bias=False)
            self.poly_terms[f'int_{order}_bn'] = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = 0.0
        
        # 1. Linear Term
        if hasattr(self, 'conv_linear'):
            out = out + self.bn_linear(self.conv_linear(x))
            
        # 2. Pure Power Terms
        for order in self.structure.get('pure', []):
            if order == 1: continue
            
            # Calculate x^n element-wise
            x_pow = x.pow(order)
            
            conv = self.poly_terms[f'pure_{order}_conv']
            bn = self.poly_terms[f'pure_{order}_bn']
            
            out = out + bn(conv(x_pow))

        # 3. Interaction Terms (CP)
        for order in self.structure.get('interact', []):
            if order == 1: continue
            
            projs = self.poly_terms[f'int_{order}_projs']
            map_out = self.poly_terms[f'int_{order}_out']
            bn = self.poly_terms[f'int_{order}_bn']
            
            # Projection: [x @ U1, x @ U2 ...]
            proj_feats = [p(x) for p in projs]
            
            # Interaction: Hadamard Product in Rank space
            # Stack: [Order, B, R, H, W] -> Prod: [B, R, H, W]
            interaction = torch.stack(proj_feats, dim=0).prod(dim=0)
            
            out = out + bn(map_out(interaction))
            
        return out