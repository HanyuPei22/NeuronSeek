import torch
import torch.nn as nn
import numpy as np
from neuronseek.core.dual_interaction_layer import DualStreamInteractionLayer

class L0Gate(nn.Module):
    """
    Implements a differentiable L0 Gate using the Hard Concrete distribution.
    Allows for gradient-based pruning of neural network components.
    """
    def __init__(self, temperature=0.66, limit_l=-0.1, limit_r=1.1, init_prob=0.9):
        super().__init__()
        self.temp = temperature
        self.limit_l = limit_l
        self.limit_r = limit_r

        # Initialize log_alpha to achieve the desired initial probability
        init_val = np.log(init_prob / (1 - init_prob))
        self.log_alpha = nn.Parameter(torch.Tensor([init_val]))

    def forward(self, x, training=True):
        if training:
            # Sample from Hard Concrete during training
            u = torch.rand_like(self.log_alpha)
            s = torch.sigmoid((torch.log(u + 1e-8) - torch.log(1 - u + 1e-8) + self.log_alpha) / self.temp)
            s = s * (self.limit_r - self.limit_l) + self.limit_l
        else:
            # Deterministic gate during inference
            s = torch.sigmoid(self.log_alpha) * (self.limit_r - self.limit_l) + self.limit_l
            
        # Clamp to [0, 1] range to act as a gate
        z = torch.clamp(s, min=0.0, max=1.0)
        # Broadcasting z (scalar) over x [Batch, Num_Classes]
        return x * z

    def regularization_term(self):
        """Returns the expected L0 cost (probability of the gate being non-zero)."""
        return torch.sigmoid(self.log_alpha - self.temp * np.log(-self.limit_l / self.limit_r))
    
    def get_prob(self):
        return torch.sigmoid(self.log_alpha).item()


class SparseSearchAgent(nn.Module):
    """
    The Orchestrator for Differentiable Search.
    
    Updates:
    - Adapted 'inspect_gates' to handle the removal of 'coeffs_interact' in the core layer.
    - Now monitors 'factors' magnitude directly for the Interaction Stream.
    """
    def __init__(self, input_dim=10, num_classes=1, rank=8, max_order=5):
        super().__init__()
        self.input_dim = input_dim
        self.max_order = max_order
        # Instantiate the updated Math Core (Parallel Neuron Version)
        self.core = DualStreamInteractionLayer(input_dim, num_classes, rank, max_order)

        self.bias = nn.Parameter(torch.zeros(num_classes))
        
        # Gates are shared across all classes (Structure Sharing)
        self.gates_pure = nn.ModuleList([L0Gate() for _ in range(max_order)])
        self.gates_int  = nn.ModuleList([L0Gate() for _ in range(max_order)])

        # BN maintains statistics for [Batch, Num_Classes]
        self.bn_pure = nn.ModuleList(nn.BatchNorm1d(num_classes, affine=True) for _ in range(max_order))
        self.bn_int = nn.ModuleList(nn.BatchNorm1d(num_classes, affine=True) for _ in range(max_order))

    def forward(self, x, training=True):
        # Start with global bias
        output = self.bias
        
        # --- Stream 1: Pure Power Terms ---
        for i, gate in enumerate(self.gates_pure):
            # core returns [Batch, Num_Classes]
            term = self.core.get_pure_term(x, i)
            # BN expects [Batch, Num_Classes], which is correct
            term_norm = self.bn_pure[i](term)
            output = output + gate(term_norm, training=training)

        # --- Stream 2: Interaction Terms ---
        for i, gate in enumerate(self.gates_int):
            # return [Batch, Num_Classes]
            term = self.core.get_interaction_term(x, i)
            term_norm = self.bn_int[i](term)
            output = output + gate(term_norm, training=training)
            
        return output
    
    def get_structure(self, threshold=0.5):
        """
        Extracts the discovered structure based on gate probabilities.
        Returns indices of active terms (1-based index).
        """
        pure_active = []
        int_active = []
        
        with torch.no_grad():
            for i, gate in enumerate(self.gates_pure):
                if gate.regularization_term() > threshold:
                    pure_active.append(i + 1)
            
            for i, gate in enumerate(self.gates_int):
                if gate.regularization_term() > threshold:
                    int_active.append(i + 1)
                    
        return pure_active, int_active

    def calculate_regularization(self):
            """
            Computes L0 regularization with Order-Weighted penalties.
            Penalty increases linearly or exponentially with the order of the term.
            """
            reg_loss = 0.0
            
            # 1. Pure Terms Penalty
            # Increasing penalty: Order 1 is cheap, Order 5 is expensive.
            for i, gate in enumerate(self.gates_pure):

                # Strategy: Linear increase penalty = 1.0 + (0.5 * order_index)
                order_penalty = 1.0 #+ (0.5 * i)
                reg_loss += order_penalty * gate.regularization_term()

            # 2. Interaction Terms Penalty
            # Interactions are more complex and prone to overfitting in high dims.
            # We apply a slightly harsher penalty slope.
            for i, gate in enumerate(self.gates_int):
                # i=0 -> Order 1 (or 2 depending on definition), ...
                order_penalty = 1.0 #+ (1.0 * i) 
                reg_loss += order_penalty * gate.regularization_term()
                
            return reg_loss

    def inspect_gates(self, threshold=0.5):
        """
        Utility to visualize gate status and weight magnitudes.
        Refactored to handle the new core structure (factors vs coeffs).
        """
        print(f"\n>>> Gate Inspection (Threshold={threshold}) <<<")
        
        def _get_weight_mag(param_or_list):
            # Helper: Handles both Parameter (Pure) and ParameterList (Interaction)
            if isinstance(param_or_list, nn.Parameter):
                return param_or_list.detach().abs().mean().item()
            elif isinstance(param_or_list, (nn.ParameterList, list)):
                # Average magnitude across all factor tensors for this order
                mags = [p.detach().abs().mean() for p in param_or_list]
                return torch.tensor(mags).mean().item()
            return 0.0

        def _fmt(name, gates, params_source):
            info = []
            for i, gate in enumerate(gates):
                prob = gate.regularization_term().item()
                # Determine weight magnitude
                # params_source is a list/ModuleList. Access the i-th element.
                weight = _get_weight_mag(params_source[i])
                
                status = "[ON]" if prob > threshold else " .  "
                info.append(f"Ord{i+1}:{status} P={prob:.4f} W={weight:.4f}")
            return f"{name}:\n  " + " | ".join(info)

        print(_fmt("Pure Stream", self.gates_pure, self.core.coeffs_pure))
        # Note: We pass self.core.factors (ModuleList of ParameterLists) here
        print(_fmt("Int  Stream", self.gates_int, self.core.factors))
        print("-" * 60)