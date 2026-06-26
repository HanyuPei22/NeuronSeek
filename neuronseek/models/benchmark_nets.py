"""
Benchmark Networks for Tabular Data Experiments.

Three architectures with unified interface:
- MLPNet: Standard fully-connected baseline
- KANNet: Kolmogorov-Arnold Network
- TaskBasedNet: Structure-aware network using discovered formula (Supports Independent & CP modes)
"""

import torch.nn as nn
from typing import List, Dict, Any, Optional
from neuronseek.layers.task_based_neurons import TaskBasedNeuron, CPTaskBasedNeuron
from neuronseek.layers.kan_layer import KAN


class MLPNet(nn.Module):
    """Standard MLP baseline."""

    def __init__(
        self,
        in_features: int,
        n_classes: int = 1,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_classes = n_classes

        layers = []
        prev_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Output layer
        out_dim = n_classes if n_classes > 1 else 1
        layers.append(nn.Linear(prev_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class KANNet(nn.Module):
    """Kolmogorov-Arnold Network wrapper."""

    def __init__(
        self,
        in_features: int,
        n_classes: int = 1,
        hidden_dims: List[int] = [64, 32],
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()
        self.n_classes = n_classes
        out_dim = n_classes if n_classes > 1 else 1

        layer_dims = [in_features] + hidden_dims + [out_dim]
        self.net = KAN(layer_dims, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x):
        return self.net(x)


class TaskBasedNet(nn.Module):
    """
    Structure-aware network using TaskBasedNeuron (Independent) or CPTaskBasedNeuron (Shared) for hidden layers.
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int = 1,
        hidden_dims: List[int] = [6, 3],
        structure_info: Optional[Dict[str, Any]] = None,
        dropout: float = 0.0,
        use_cp: bool = False,
        rank: int = 8,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        self.n_classes = n_classes

        # Default structure if not provided
        if structure_info is None:
            structure_info = {'pure_indices': [1], 'interact_indices': [2]}

        pure_indices = structure_info.get('pure_indices', [1])
        interact_indices = structure_info.get('interact_indices', [])
        
        current_rank = structure_info.get('rank', rank)

        # Build all hidden layers
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        prev_dim = in_features
        for h_dim in hidden_dims:
            if use_cp:
                layer = CPTaskBasedNeuron(
                    in_features=prev_dim,
                    out_features=h_dim,
                    pure_indices=pure_indices,
                    interact_indices=interact_indices,
                    rank=current_rank,
                    use_batchnorm=use_batchnorm
                )
            else:
                layer = TaskBasedNeuron(
                    in_features=prev_dim,
                    out_features=h_dim,
                    pure_indices=pure_indices,
                    interact_indices=interact_indices,
                    use_batchnorm=use_batchnorm
                )

            self.layers.append(layer)
            self.activations.append(nn.ReLU())
            
            if dropout > 0:
                self.dropouts.append(nn.Dropout(dropout))
            else:
                self.dropouts.append(nn.Identity())
                
            prev_dim = h_dim

        # Output layer (simple linear)
        out_dim = n_classes if n_classes > 1 else 1
        self.output_layer = nn.Linear(prev_dim, out_dim)

    def forward(self, x):
        for layer, act, drop in zip(self.layers, self.activations, self.dropouts):
            x = drop(act(layer(x)))
        return self.output_layer(x)


# =============================================================================
# Factory Function
# =============================================================================

def build_network(
    arch: str,
    in_features: int,
    n_classes: int = 1,
    hidden_dims: List[int] = [64, 32],
    structure_info: Optional[Dict] = None,
    **kwargs,  
) -> nn.Module:
    """
    Factory function to build networks.

    Args:
        arch: 'mlp', 'kan', or 'tasknet' (can add suffix like 'tasknet_cp')
        in_features: Input dimension
        n_classes: 1 for regression, >1 for classification
        hidden_dims: List of hidden layer sizes
        structure_info: Required for 'tasknet', from Searcher output
        **kwargs: Extra args like dropout, use_cp, rank
    """
    arch = arch.lower()

    if arch == 'mlp':
        return MLPNet(in_features, n_classes, hidden_dims, **kwargs)
    elif arch == 'kan':
        return KANNet(in_features, n_classes, hidden_dims, **kwargs)
    elif arch in ('tasknet', 'task', 'taskbased', 'ns-net', 'tnsr-net'):
        if structure_info is None:
            raise ValueError("TaskBasedNet requires structure_info from Searcher")
        return TaskBasedNet(in_features, n_classes, hidden_dims, structure_info, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose from: mlp, kan, tasknet")


def get_loss_fn(n_classes: int):
    """Return appropriate loss function based on task type."""
    if n_classes == 1:
        return nn.MSELoss()
    else:
        return nn.CrossEntropyLoss()