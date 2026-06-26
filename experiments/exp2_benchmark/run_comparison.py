"""
Stage 2: Model Comparison

Trains and evaluates MLP, KAN, NS-Net, TNSR-Net on each dataset.
Requires structure JSON files from run_search.py.

Usage:
    python -m experiments.exp2_benchmark.run_comparison --datasets california_housing
    python -m experiments.exp2_benchmark.run_comparison --all
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.exp2_benchmark.config import (
    ALL_DATASETS, NETWORK_CONFIG, TRAIN_CONFIG, PATHS, EXPERIMENT_CONFIG
)
from neuronseek.utils.tabular_data_loader import load_dataset
from neuronseek.models.benchmark_nets import build_network, get_loss_fn


def load_structure(dataset_name: str, structure_dir: str) -> dict:
    """Load structure info from JSON."""
    filepath = os.path.join(structure_dir, f"{dataset_name}_structure.json")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Structure file not found: {filepath}. Run run_search.py first.")
    with open(filepath, 'r') as f:
        return json.load(f)


def train_epoch(model, loader, optimizer, loss_fn, device):
    """Single training epoch."""
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, task: str):
    """Evaluate model on given loader."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item() * len(X)

        if task == 'classification':
            pred_class = pred.argmax(dim=1)
            correct += (pred_class == y).sum().item()
        total += len(X)

    avg_loss = total_loss / total
    acc = correct / total if task == 'classification' else None
    return avg_loss, acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    task: str,
    device: torch.device,
) -> dict:
    """Full training loop with early stopping."""
    model.to(device)
    loss_fn = get_loss_fn(model.n_classes)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    if config['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device, task)
        scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config['early_stop_patience']:
                break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    return {'best_val_loss': best_val_loss, 'epochs_trained': epoch + 1}


def run_comparison(dataset_name: str, structure_info: dict, seed: int = 42) -> dict:
    """Run comparison for all architectures on one dataset."""
    print(f"\n{'='*60}")
    print(f"Comparison: {dataset_name} | Seed: {seed}")
    print('='*60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    data = load_dataset(
        dataset_name,
        test_size=EXPERIMENT_CONFIG['test_size'],
        val_size=EXPERIMENT_CONFIG['val_size'],
        random_state=seed,
        return_torch=True,
    )

    train_dataset = TensorDataset(data['X_train'], data['y_train'])
    val_dataset = TensorDataset(data['X_val'], data['y_val'])
    test_dataset = TensorDataset(data['X_test'], data['y_test'])

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TRAIN_CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=TRAIN_CONFIG['batch_size'])

    n_features = data['n_features']
    n_classes = data['n_classes']
    task = data['task']

    # Architectures to compare
    architectures = {
        'MLP': {'arch': 'mlp'},
        'KAN': {'arch': 'kan', 'grid_size': NETWORK_CONFIG['kan_grid_size']},
        'NS-Net': {'arch': 'tasknet', 'structure_info': structure_info.get('NeuronSeek', {})},
        'TNSR-Net': {'arch': 'tasknet', 'structure_info': structure_info.get('TNSR', {})},
    }

    results = {
        'dataset': dataset_name,
        'task': task,
        'n_features': n_features,
        'n_classes': n_classes,
        'seed': seed,
    }

    loss_fn = get_loss_fn(n_classes)

    for arch_name, arch_config in architectures.items():
        print(f"\n  [{arch_name}] Training...")

        try:
            # Build kwargs based on architecture type
            build_kwargs = {
                'in_features': n_features,
                'n_classes': n_classes,
                'hidden_dims': NETWORK_CONFIG['hidden_dims'],
                **arch_config,
            }
            # Only pass dropout to architectures that support it (MLP, TaskNet)
            if arch_config['arch'] != 'kan':
                build_kwargs['dropout'] = NETWORK_CONFIG['dropout']

            model = build_network(**build_kwargs)

            train_info = train_model(
                model, train_loader, val_loader,
                TRAIN_CONFIG, task, device
            )

            # Final test evaluation
            test_loss, test_acc = evaluate(model, test_loader, loss_fn, device, task)

            if task == 'regression':
                results[arch_name] = {'mse': test_loss}
                print(f"    Test MSE: {test_loss:.4f}")
            else:
                results[arch_name] = {'acc': test_acc, 'loss': test_loss}
                print(f"    Test Acc: {test_acc:.4f}")

        except Exception as e:
            print(f"    [Error] {e}")
            results[arch_name] = {'error': str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(description='Compare networks on datasets')
    parser.add_argument('--datasets', nargs='+', default=['california_housing'])
    parser.add_argument('--all', action='store_true', help='Run on all datasets')
    parser.add_argument('--n_trials', type=int, default=EXPERIMENT_CONFIG['n_trials'])
    parser.add_argument('--seed_base', type=int, default=EXPERIMENT_CONFIG['seed_base'])
    parser.add_argument('--structure_dir', type=str, default=PATHS['structure_dir'])
    parser.add_argument('--output_csv', type=str, default=PATHS['result_csv'])
    args = parser.parse_args()

    datasets = ALL_DATASETS if args.all else args.datasets

    print(f"Model Comparison | Datasets: {len(datasets)} | Trials: {args.n_trials}")

    all_results = []

    for ds_name in datasets:
        try:
            structure_info = load_structure(ds_name, args.structure_dir)
        except FileNotFoundError as e:
            print(f"\n[Skip] {ds_name}: {e}")
            continue

        for trial in range(args.n_trials):
            seed = args.seed_base + trial
            try:
                results = run_comparison(ds_name, structure_info, seed)
                results['trial'] = trial
                all_results.append(results)
            except Exception as e:
                print(f"\n[Error] {ds_name} trial {trial}: {e}")

    # Save results
    if all_results:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

        # Flatten results for CSV
        rows = []
        for r in all_results:
            base = {
                'dataset': r['dataset'],
                'task': r['task'],
                'trial': r['trial'],
                'seed': r['seed'],
            }
            for arch in ['MLP', 'KAN', 'NS-Net', 'TNSR-Net']:
                if arch in r:
                    arch_result = r[arch]
                    if 'mse' in arch_result:
                        base[f'{arch}_mse'] = arch_result['mse']
                    if 'acc' in arch_result:
                        base[f'{arch}_acc'] = arch_result['acc']
            rows.append(base)

        df = pd.DataFrame(rows)
        df.to_csv(args.output_csv, index=False)
        print(f"\n[Saved] {args.output_csv}")

        # Print summary
        print(f"\n{'='*60}")
        print("Summary (Mean ± Std)")
        print('='*60)
        for ds in df['dataset'].unique():
            ds_df = df[df['dataset'] == ds]
            task = ds_df['task'].iloc[0]
            print(f"\n{ds} ({task}):")
            metric_cols = [c for c in df.columns if '_mse' in c or '_acc' in c]
            for col in metric_cols:
                if col in ds_df.columns:
                    mean = ds_df[col].mean()
                    std = ds_df[col].std()
                    print(f"  {col}: {mean:.4f} ± {std:.4f}")

    print(f"\n{'='*60}")
    print("Comparison complete!")
    print('='*60)


if __name__ == '__main__':
    main()
