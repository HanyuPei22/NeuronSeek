"""
Experiment Utilities for Exp2: Tabular Data Benchmark.

This module provides shared utilities for all benchmark notebooks:
- Model statistics (FLOPs, parameters)
- Training loop with differential learning rates
- Experiment runner for model comparison
- Structure loading
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Project Setup
# =============================================================================

def setup_project() -> Tuple[str, torch.device]:
    """
    Setup project path and device.

    Returns:
        (project_root, device)
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return project_root, device


# =============================================================================
# Model Statistics
# =============================================================================

def count_params(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """
    Estimate FLOPs for Linear and KAN layers using forward hooks.

    Args:
        model: PyTorch model
        input_shape: Input shape (batch_size, input_features), e.g., (1, 8)

    Returns:
        Estimated FLOPs count
    """
    # Import KANLayer here to avoid circular imports
    try:
        from neuronseek.layers.kan_layer import KANLayer
        has_kan = True
    except ImportError:
        has_kan = False
        KANLayer = None

    flops = 0

    def linear_hook(module, input, output):
        nonlocal flops
        # Linear: y = xA^T + b. MACs = in_dim * out_dim
        # 1 MAC = 2 FLOPs (Multiply + Add)
        in_feat = input[0].shape[1]
        out_feat = output.shape[1]
        flops += 2 * in_feat * out_feat

    def kan_hook(module, input, output):
        nonlocal flops
        in_feat = module.in_features
        out_feat = module.out_features
        n_coeffs = module.grid_size + module.spline_order
        order = module.spline_order

        # Base path: silu + matmul
        base_flops = in_feat + 2 * in_feat * out_feat
        # Spline path: einsum
        spline_flops = 2 * out_feat * in_feat * n_coeffs
        # B-spline basis computation (approximate)
        bspline_flops = in_feat * n_coeffs * order * 4

        flops += base_flops + spline_flops + bspline_flops

    hooks = []
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(linear_hook))
        elif has_kan and KANLayer is not None and isinstance(layer, KANLayer):
            hooks.append(layer.register_forward_hook(kan_hook))

    # Dummy forward pass
    dummy_input = torch.randn(input_shape).to(next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        model(dummy_input)

    for h in hooks:
        h.remove()

    return flops


# =============================================================================
# Structure Loading
# =============================================================================

def load_structure(dataset_name: str, structure_dir: str) -> Optional[dict]:
    """
    Load structure info from JSON file.

    Args:
        dataset_name: Name of the dataset
        structure_dir: Directory containing structure JSON files

    Returns:
        Structure dictionary or None if not found
    """
    filepath = os.path.join(structure_dir, f"{dataset_name}_structure.json")
    if not os.path.exists(filepath):
        print(f"[Warning] Structure file not found: {filepath}. Using default.")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)


# =============================================================================
# Training
# =============================================================================

def train_and_track(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    task: str,
    device: torch.device,
    loss_fn: nn.Module = None,
    verbose: bool = True
) -> Tuple[Dict[str, List[float]], float]:
    """
    Training loop with differential learning rates and gradient clipping.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training config dict with 'epochs' key
        task: 'regression' or 'classification'
        device: torch device
        loss_fn: Loss function (auto-detected if None)
        verbose: Whether to print parameter grouping info

    Returns:
        (history dict with 'val_loss', best_val_loss)
    """
    model.to(device)

    # Auto-detect loss function if not provided
    if loss_fn is None:
        from neuronseek.models.benchmark_nets import get_loss_fn
        loss_fn = get_loss_fn(model.n_classes)

    # Group parameters for differential learning rates
    params_dict = {
        'pure': [],
        'interact': [],
        'bias': [],
        'bn': [],
        'other': []
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'pure_layers' in name and 'weight' in name:
            params_dict['pure'].append(param)
        elif 'int_params' in name:
            params_dict['interact'].append(param)
        elif 'bias' in name or 'beta' in name:
            params_dict['bias'].append(param)
        elif 'norms' in name or 'bn_' in name:
            params_dict['bn'].append(param)
        else:
            params_dict['other'].append(param)

    if verbose:
        print(f"Params Grouping: Pure={len(params_dict['pure'])}, "
              f"Int={len(params_dict['interact'])}, Bias={len(params_dict['bias'])}, "
              f"BN={len(params_dict['bn'])}, Other={len(params_dict['other'])}")

    optimizer_groups = [
        {'params': params_dict['pure'], 'lr': 0.01, 'weight_decay': 1e-4},
        {'params': params_dict['interact'], 'lr': 0.01, 'weight_decay': 1e-4},
        {'params': params_dict['bias'], 'lr': 0.01, 'weight_decay': 0.0},
        {'params': params_dict['bn'], 'lr': 0.01, 'weight_decay': 0.0},
        {'params': params_dict['other'], 'lr': 0.01, 'weight_decay': 1e-4},
    ]

    optimizer = torch.optim.Adam(optimizer_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-5
    )

    history = {'val_loss': []}
    best_val_loss = float('inf')

    for epoch in range(config['epochs']):
        # --- Train ---
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # --- Validation ---
        model.eval()
        ep_val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)
                ep_val_loss += loss.item() * len(X)

        avg_val_loss = ep_val_loss / len(val_loader.dataset)
        history['val_loss'].append(avg_val_loss)

        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    return history, best_val_loss


# =============================================================================
# Experiment Runner
# =============================================================================

def run_experiment(
    models_config: Dict[str, dict],
    data: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    train_config: dict,
    device: torch.device,
    plot: bool = True,
    verbose: bool = True
) -> Dict[str, dict]:
    """
    Run a full model comparison experiment.

    Args:
        models_config: Dict mapping model names to their config dicts
        data: Data dict from load_dataset (needs 'n_features', 'n_classes', 'task')
        train_loader, val_loader, test_loader: Data loaders
        train_config: Training configuration
        device: torch device
        plot: Whether to plot validation curves
        verbose: Whether to print progress

    Returns:
        Results dict with model stats and metrics
    """
    from neuronseek.models.benchmark_nets import build_network, get_loss_fn

    results = {}

    if verbose:
        print(f"{'Model':<20} | {'Params':<8} | {'FLOPs':<10} | {'Test Loss':<10}")
        print("-" * 60)

    if plot:
        plt.figure(figsize=(12, 6))

    for idx, (name, cfg) in enumerate(models_config.items()):
        # 1. Build model
        try:
            model = build_network(
                in_features=data['n_features'],
                n_classes=data['n_classes'],
                **cfg
            )
        except Exception as e:
            print(f"Skipping {name}: {e}")
            continue

        # 2. Compute statistics
        n_params = count_params(model)
        flops = count_flops(model, (1, data['n_features']))

        # 3. Train
        hist, best_val = train_and_track(
            model, train_loader, val_loader,
            train_config, data['task'], device,
            verbose=verbose
        )

        # 4. Test evaluation
        loss_fn = get_loss_fn(model.n_classes)
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in test_loader:
                test_loss += loss_fn(model(X.to(device)), y.to(device)).item() * len(X)
        test_loss /= len(test_loader.dataset)

        # 5. Store results
        results[name] = {
            'params': n_params,
            'flops': flops,
            'test_loss': test_loss,
            'best_val_loss': best_val,
            'history': hist
        }

        if verbose:
            print(f"{name:<20} | {n_params:<8} | {flops:<10.0f} | {test_loss:<10.4f}")

        if plot:
            plt.plot(hist['val_loss'], label=f"{name} (Best: {best_val:.4f})",
                     linewidth=2, alpha=0.8)

    if plot:
        plt.title(f"Model Comparison on {data.get('dataset_name', 'Dataset')} (Validation Loss)")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE/CE)")
        plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.show()

    return results


# =============================================================================
# Data Loader Helper
# =============================================================================

def prepare_dataloaders(
    data: dict,
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders from load_dataset output.

    Args:
        data: Dict from load_dataset with X_train, y_train, etc.
        batch_size: Batch size for loaders

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = TensorDataset(data['X_train'], data['y_train'])
    val_dataset = TensorDataset(data['X_val'], data['y_val'])
    test_dataset = TensorDataset(data['X_test'], data['y_test'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
