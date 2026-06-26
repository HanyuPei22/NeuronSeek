import torch
import torch.nn as nn
import numpy as np



class DualStreamInteractionLayer(nn.Module):
    """
    Math core for NeuronSeek's dual-stream architecture.

    Stream A (Interaction): CP decomposition for cross-feature interactions
    Stream B (Pure): Explicit polynomial terms x^k
    """

    def __init__(self, input_dim: int, num_classes: int, rank: int, poly_order: int):
        super().__init__()
        self.rank = rank
        self.num_classes = num_classes
        self.poly_order = poly_order

        # Interaction Stream: ModuleList[Order] -> ParameterList[Term] -> Tensor[D, R, C]
        self.factors = nn.ModuleList()
        for i in range(1, poly_order + 1):
            order_params = nn.ParameterList([
                nn.Parameter(torch.empty(input_dim, rank, num_classes))
                for _ in range(i)
            ])
            self.factors.append(order_params)

        # Pure Stream: ParameterList[Order] -> Tensor[D, C]
        self.coeffs_pure = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, num_classes))
            for _ in range(poly_order)
        ])

        # Global bias
        self.beta = nn.Parameter(torch.zeros(num_classes))

        # Masks for gating
        self.register_buffer('mask_interact', torch.ones(poly_order))
        self.register_buffer('mask_pure', torch.ones(poly_order))

        self.reset_parameters()

    def reset_parameters(self):
        for order_params in self.factors:
            for p in order_params:
                nn.init.normal_(p, std=0.05)
        for p in self.coeffs_pure:
            nn.init.normal_(p, std=0.05)

    def forward(self, x: torch.Tensor):
        """Input: [B, D] -> Output: [B, C]"""
        batch_size = x.size(0)
        logits = self.beta.unsqueeze(0).expand(batch_size, -1).clone()

        for i in range(self.poly_order):
            order = i + 1

            # Interaction stream
            if self.mask_interact[i] == 1.0:
                factors = self.factors[i]
                projections = [torch.einsum('bd, drc -> brc', x, u) for u in factors]
                combined = projections[0]
                for p in projections[1:]:
                    combined = combined * p
                logits = logits + torch.sum(combined, dim=1)

            # Pure stream
            if self.mask_pure[i] == 1.0:
                term = x if order == 1 else x.pow(order)
                logits = logits + (term @ self.coeffs_pure[i])

        return logits

    def get_pure_term(self, x: torch.Tensor, order_idx: int) -> torch.Tensor:
        """Compute pure term x^k for specific order. Returns [B, C]."""
        order = order_idx + 1
        term = x if order == 1 else x.pow(order)
        return term @ self.coeffs_pure[order_idx]

    def get_interaction_term(self, x: torch.Tensor, order_idx: int) -> torch.Tensor:
        """Compute CP interaction term for specific order. Returns [B, C]."""
        factors = self.factors[order_idx]
        projections = [torch.einsum('bd, drc -> brc', x, u) for u in factors]
        combined = projections[0]
        for p in projections[1:]:
            combined = combined * p
        return torch.sum(combined, dim=1)


# =============================================================================
# L0 Gate (Differentiable Pruning)
# =============================================================================

class L0Gate(nn.Module):
    """
    Differentiable L0 gate using Hard Concrete distribution.
    Enables gradient-based structure pruning.
    """

    def __init__(self, temperature=0.66, limit_l=-0.1, limit_r=1.1, init_prob=0.9):
        super().__init__()
        self.temp = temperature
        self.limit_l = limit_l
        self.limit_r = limit_r

        init_val = np.log(init_prob / (1 - init_prob))
        self.log_alpha = nn.Parameter(torch.Tensor([init_val]))

    def forward(self, x, training=True):
        if training:
            u = torch.rand_like(self.log_alpha)
            s = torch.sigmoid((torch.log(u + 1e-8) - torch.log(1 - u + 1e-8) + self.log_alpha) / self.temp)
            s = s * (self.limit_r - self.limit_l) + self.limit_l
        else:
            s = torch.sigmoid(self.log_alpha) * (self.limit_r - self.limit_l) + self.limit_l

        z = torch.clamp(s, min=0.0, max=1.0)
        return x * z

    def regularization_term(self):
        """Returns expected L0 cost (probability of gate being non-zero)."""
        return torch.sigmoid(self.log_alpha - self.temp * np.log(-self.limit_l / self.limit_r))

    def get_prob(self):
        return torch.sigmoid(self.log_alpha).item()


# =============================================================================
# Sparse Search Agent (Orchestrator)
# =============================================================================

class SparseSearchAgent(nn.Module):
    """
    Orchestrates differentiable structure search.
    Combines L0 gates, BatchNorm, and DualStreamInteractionLayer.
    """

    def __init__(self, input_dim=10, num_classes=1, rank=8, max_order=5):
        super().__init__()
        self.input_dim = input_dim
        self.max_order = max_order

        self.core = DualStreamInteractionLayer(input_dim, num_classes, rank, max_order)
        self.bias = nn.Parameter(torch.zeros(num_classes))

        # Gates shared across classes
        self.gates_pure = nn.ModuleList([L0Gate() for _ in range(max_order)])
        self.gates_int = nn.ModuleList([L0Gate() for _ in range(max_order)])

        # BatchNorm for each term
        self.bn_pure = nn.ModuleList(nn.BatchNorm1d(num_classes, affine=True) for _ in range(max_order))
        self.bn_int = nn.ModuleList(nn.BatchNorm1d(num_classes, affine=True) for _ in range(max_order))

    def forward(self, x, training=True):
        output = self.bias

        # Pure stream
        for i, gate in enumerate(self.gates_pure):
            term = self.core.get_pure_term(x, i)
            term_norm = self.bn_pure[i](term)
            output = output + gate(term_norm, training=training)

        # Interaction stream
        for i, gate in enumerate(self.gates_int):
            term = self.core.get_interaction_term(x, i)
            term_norm = self.bn_int[i](term)
            output = output + gate(term_norm, training=training)

        return output

    def get_structure(self, threshold=0.5):
        """Extract discovered structure. Returns (pure_indices, interact_indices) 1-based."""
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
        """Compute L0 regularization loss."""
        reg_loss = 0.0

        for i, gate in enumerate(self.gates_pure):
            order_penalty = 1.0
            reg_loss += order_penalty * gate.regularization_term()

        for i, gate in enumerate(self.gates_int):
            order_penalty = 1.0
            reg_loss += order_penalty * gate.regularization_term()

        return reg_loss

    def inspect_gates(self, threshold=0.5):
        """Utility to visualize gate status and weight magnitudes."""
        print(f"\n>>> Gate Inspection (Threshold={threshold}) <<<")

        def _get_weight_mag(param_or_list):
            if isinstance(param_or_list, nn.Parameter):
                return param_or_list.detach().abs().mean().item()
            elif isinstance(param_or_list, (nn.ParameterList, list)):
                mags = [p.detach().abs().mean() for p in param_or_list]
                return torch.tensor(mags).mean().item()
            return 0.0

        def _fmt(name, gates, params_source):
            info = []
            for i, gate in enumerate(gates):
                prob = gate.regularization_term().item()
                weight = _get_weight_mag(params_source[i])
                status = "[ON]" if prob > threshold else " .  "
                info.append(f"Ord{i+1}:{status} P={prob:.4f} W={weight:.4f}")
            return f"{name}:\n  " + " | ".join(info)

        print(_fmt("Pure Stream", self.gates_pure, self.core.coeffs_pure))
        print(_fmt("Int  Stream", self.gates_int, self.core.factors))
        print("-" * 60)
