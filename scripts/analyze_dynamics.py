
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Adjust path to project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Import your existing modules
from neuronseek.utils.data_utils import SyntheticGenerator
from experiments.common.structure_evaluator import StructuralProbe

def parse_arguments():
    parser = argparse.ArgumentParser(description="Dynamics Analyzer: Pure vs Interaction")
    
    # Environment
    parser.add_argument('--dim', type=int, default=30, help='Input dimension')
    parser.add_argument('--rank', type=int, default=8, help='Rank for Interaction')
    parser.add_argument('--samples', type=int, default=2000, help='Number of training samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Training
    parser.add_argument('--epochs', type=int, default=300, help='Training epochs')
    parser.add_argument('--lr_pure', type=float, default=0.01, help='Learning rate for Pure components')
    parser.add_argument('--lr_int', type=float, default=0.01, help='Learning rate for Interaction components')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 Regularization')
    
    # Structure (Controls what is physically in the model)
    parser.add_argument('--pure_orders', type=int, nargs='+', default=[1, 2], help='Active pure orders')
    parser.add_argument('--interact_orders', type=int, nargs='+', default=[2], help='Active interact orders')
    
    return parser.parse_args()

def get_component_mse(probe, X, y, component='pure'):
    """
    Helper to calculate MSE if ONLY one component was active.
    """
    probe.eval()
    with torch.no_grad():
        batch_size = X.size(0)
        output = probe.bias.unsqueeze(0).expand(batch_size, -1).clone()
        
        if component == 'pure':
            for order_str, weight in probe.pure_modules.items():
                order = int(order_str)
                term = X if order == 1 else X.pow(order)
                output = output + (term @ weight)
                
        elif component == 'interact':
            for order_str, factors in probe.interact_modules.items():
                projections = [torch.einsum('bd, drc -> brc', X, u) for u in factors]
                combined = projections[0]
                for p in projections[1:]:
                    combined = combined * p
                output = output + torch.sum(combined, dim=1)
        
        loss = nn.MSELoss()(output, y)
    probe.train()
    return loss.item()

def main():
    args = parse_arguments()
    
    # 1. Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"DYNAMICS ANALYSIS | Dim={args.dim} | Rank={args.rank}")
    print(f"LR_Pure={args.lr_pure} | LR_Int={args.lr_int}")
    print(f"{'='*60}")

    # 2. Data Generation (Ground Truth: Strictly Interaction Order 2)
    print("Generating Synthetic Data (Ground Truth: Interaction [2] ONLY)...")
    gen = SyntheticGenerator(n_samples=args.samples + 500, input_dim=args.dim, noise_level=0.01)
    dataset, truth = gen.get_data(mode='interact', variant=0)
    
    X, y = dataset.tensors
    X = X.to(device)
    y = y.to(device)
    
    X_train, y_train = X[:args.samples], y[:args.samples]
    
    baseline_mse = torch.var(y).item()
    print(f"Data Variance (Baseline): {baseline_mse:.5f}\n")

    # 3. Model Setup
    structure_info = {
        'type': 'neuronseek',
        'rank': args.rank,
        'pure_indices': args.pure_orders,
        'interact_indices': args.interact_orders
    }
    
    probe = StructuralProbe(args.dim, structure_info, num_classes=1).to(device)
    
    # 4. Optimizer with Differential Learning Rates
    optimizer = optim.Adam([
        {'params': probe.pure_modules.parameters(), 'lr': args.lr_pure},
        {'params': probe.bias, 'lr': args.lr_pure},
        {'params': probe.interact_modules.parameters(), 'lr': args.lr_int}
    ], weight_decay=args.weight_decay)
    
    # 5. Training Loop
    history = {
        'epoch': [],
        'total_loss': [],
        'pure_solo_mse': [],
        'int_solo_mse': []
    }
    
    print("Starting Training...")
    for epoch in range(1, args.epochs + 1):
        probe.train()
        optimizer.zero_grad()
        
        pred = probe(X_train)
        loss = nn.MSELoss()(pred, y_train)
        loss.backward()
        optimizer.step()
        
        mse_pure_solo = get_component_mse(probe, X_train, y_train, 'pure')
        mse_int_solo = get_component_mse(probe, X_train, y_train, 'interact')
        
        history['epoch'].append(epoch)
        history['total_loss'].append(loss.item())
        history['pure_solo_mse'].append(mse_pure_solo)
        history['int_solo_mse'].append(mse_int_solo)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Total: {loss.item():.4f} | Pure_Solo: {mse_pure_solo:.4f} | Int_Solo: {mse_int_solo:.4f}")

    # 6. Visualization
    save_path = f"dynamics_D{args.dim}_R{args.rank}_LRp{args.lr_pure}_LRi{args.lr_int}.png"
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['total_loss'], label='Total Model Loss', color='black', linewidth=2)
    plt.plot(history['epoch'], history['pure_solo_mse'], label='Pure Stream Solo MSE', color='blue', linestyle='--')
    plt.plot(history['epoch'], history['int_solo_mse'], label='Interact Stream Solo MSE', color='red', linestyle='--')
    
    plt.axhline(y=baseline_mse, color='gray', linestyle=':', label='Baseline (Variance)')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('MSE (Log Scale)')
    plt.title(f'Training Dynamics: Pure vs Interact (Dim={args.dim})\nLR_Pure={args.lr_pure}, LR_Int={args.lr_int}')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.savefig(save_path)
    print(f"\nPlot saved to: {save_path}")

if __name__ == "__main__":
    main()
