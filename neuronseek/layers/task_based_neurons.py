import torch
import torch.nn as nn
import math

class TaskBasedNeuron(nn.Module):
    """
    Original Independent Rank-1 Implementation.
    Each output neuron learns its own independent interaction parameters.
    """
    def __init__(self, in_features: int, out_features: int,
                 pure_indices: list, interact_indices: list,
                 bias: bool = True, use_batchnorm: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_batchnorm = use_batchnorm

        self.pure_orders = sorted(pure_indices)
        self.int_orders = sorted(interact_indices)

        self.pure_layers = nn.ModuleDict()
        self.int_layers = nn.ModuleDict()
        
        if self.use_batchnorm:
            self.norms = nn.ModuleDict()

        # 1. Define Pure Stream Layers
        for k in self.pure_orders:
            self.pure_layers[f'ord_{k}'] = nn.Linear(in_features, out_features, bias=False)
            if self.use_batchnorm:
                self.norms[f'pure_{k}'] = nn.BatchNorm1d(out_features)

        # 2. Define Interaction Stream Layers (Independent)
        for k in self.int_orders:
            components = nn.ModuleList([
                nn.Linear(in_features, out_features, bias=False) for _ in range(k)
            ])
            self.int_layers[f'ord_{k}'] = components
            
            if self.use_batchnorm:
                self.norms[f'int_{k}'] = nn.BatchNorm1d(out_features)

        # 3. Define Bias
        if bias:
            self.global_bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('global_bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Pure Stream: Kaiming initialization for linear transformations
        for m in self.pure_layers.values():
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

        # Interaction Stream: Small std initialization to prevent product explosion
        for components in self.int_layers.values():
            for m in components:
                nn.init.normal_(m.weight, std=0.02)

        # Global Bias
        if self.global_bias is not None:
            nn.init.zeros_(self.global_bias)


    def forward(self, x):
        # Flatten 3D input (Batch, Seq, Feat) -> (Batch*Seq, Feat)
        is_3d = x.dim() == 3
        if is_3d:
            batch, seq, feat = x.shape
            x = x.reshape(-1, feat)

        total_sum = 0.0

        # Pure Stream Forward
        for k_str, layer in self.pure_layers.items():
            k = int(k_str.split('_')[1])
            x_in = x if k == 1 else torch.pow(x, k)
            term = layer(x_in)
            if self.use_batchnorm:
                term = self.norms[f'pure_{k}'](term)
            total_sum = total_sum + term

        # Interaction Stream Forward
        for k_str, components in self.int_layers.items():
            k = int(k_str.split('_')[1])
            
            # Element-wise product of independent projections
            product_term = components[0](x)
            for i in range(1, k):
                product_term = product_term * components[i](x)
            
            if self.use_batchnorm:
                product_term = self.norms[f'int_{k}'](product_term)
            total_sum = total_sum + product_term

        if self.global_bias is not None:
            total_sum = total_sum + self.global_bias

        if is_3d:
            total_sum = total_sum.reshape(batch, seq, self.out_features)

        return total_sum


class CPTaskBasedNeuron(nn.Module):
    """
    Shared Basis Implementation using Rank-R CP Decomposition.
    All output neurons share 'rank' interaction bases.
    
    """
    def __init__(self, in_features: int, out_features: int,
                 pure_indices: list, interact_indices: list,
                 rank: int = 8, bias: bool = True, use_batchnorm: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.use_batchnorm = use_batchnorm

        self.pure_orders = sorted(pure_indices)
        self.int_orders = sorted(interact_indices)

        self.pure_layers = nn.ModuleDict()
        self.int_params = nn.ParameterDict()
        
        if self.use_batchnorm:
            self.norms = nn.ModuleDict()

        # 1. Define Pure Stream
        for k in self.pure_orders:
            self.pure_layers[f'ord_{k}'] = nn.Linear(in_features, out_features, bias=False)
            if self.use_batchnorm:
                self.norms[f'pure_{k}'] = nn.BatchNorm1d(out_features)

        # 2. Define Interaction Stream (CP Factors)
        for k in self.int_orders:
            # U: Projection to shared rank space (Order, In, Rank)
            self.int_params[f'ord_{k}_U'] = nn.Parameter(torch.empty(k, in_features, rank))
            # W: Aggregation to output neurons (Rank, Out)
            self.int_params[f'ord_{k}_W'] = nn.Parameter(torch.empty(rank, out_features))
            
            if self.use_batchnorm:
                self.norms[f'int_{k}'] = nn.BatchNorm1d(out_features)

        # 3. Define Bias
        if bias:
            self.global_bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('global_bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Pure Stream: Kaiming initialization
        for m in self.pure_layers.values():
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

        # Interaction Stream: Small normal init for stability
        for name, param in self.int_params.items():
            nn.init.normal_(param, mean=0.0, std=0.02)

        # Global Bias
        if self.global_bias is not None:
            nn.init.zeros_(self.global_bias)

    def forward(self, x):
        # Flatten 3D input
        is_3d = x.dim() == 3
        if is_3d:
            batch, seq, feat = x.shape
            x = x.reshape(-1, feat)

        total_sum = 0.0

        # Pure Stream Forward
        for k_str, layer in self.pure_layers.items():
            k = int(k_str.split('_')[1])
            x_in = x if k == 1 else torch.pow(x, k)
            term = layer(x_in)
            if self.use_batchnorm:
                term = self.norms[f'pure_{k}'](term)
            total_sum = total_sum + term

        # Interaction Stream Forward (CP)
        for k in self.int_orders:
            U = self.int_params[f'ord_{k}_U']
            W = self.int_params[f'ord_{k}_W']
            
            # Project: (Batch, In) @ (k, In, Rank) -> (Batch, k, Rank)
            projections = torch.einsum('bi, kir -> bkr', x, U)
            
            # Interaction: Product along order dim -> (Batch, Rank)
            combined = projections[:, 0, :]
            for i in range(1, k):
                combined = combined * projections[:, i, :]
            
            # Aggregate: (Batch, Rank) @ (Rank, Out) -> (Batch, Out)
            term = combined @ W
            
            if self.use_batchnorm:
                term = self.norms[f'int_{k}'](term)
            total_sum = total_sum + term

        if self.global_bias is not None:
            total_sum = total_sum + self.global_bias

        if is_3d:
            total_sum = total_sum.reshape(batch, seq, self.out_features)

        return total_sum