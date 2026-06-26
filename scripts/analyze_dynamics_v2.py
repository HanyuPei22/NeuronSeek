import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Adjust path to project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from neuronseek.utils.data_utils import SyntheticGenerator
from experiments.common.structure_evaluator import StructuralProbe

def parse_arguments():
    parser = argparse.ArgumentParser(description="Dynamics Analyzer V2: The Three Horses")
    parser.add_argument('--dim', type=int, default=100, help='Input dimension')
    parser.add_argument('--rank', type=int, default=8, help='Rank for Interaction')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs') # 100 is enough for start
    parser.add_argument('--lr_pure', type=float, default=0.01, help='LR for Pure')
    parser.add_argument('--lr_int', type=float, default=0.01, help='LR for Interact')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def train_one_model(X, y, structure_info, lr_pure, lr_int, epochs, label):
    """
    Trains a single model structure and returns its loss history.
    """
    input_dim = X.shape[1]
    device = X.device
    
    # Init Model
    model = StructuralProbe(input_dim, structure_info, num_classes=1).to(device)
    
    # Init Optimizer (Differential LR support)
    params = []
    if len(model.pure_modules) > 0:
        params.append({'params': model.pure_modules.parameters(), 'lr': lr_pure})
    if len(model.interact_modules) > 0:
        params.append({'params': model.interact_modules.parameters(), 'lr': lr_int})
    params.append({'params': [model.bias], 'lr': lr_pure}) # Bias follows Pure LR usually
    
    optimizer = optim.Adam(params, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    
    history = []
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        
    return history

def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"RACE TRACK V2 | Dim={args.dim} | Rank={args.rank}")
    print(f"LR Scheme: Pure={args.lr_pure}, Int={args.lr_int}")
    print(f"{'='*60}")

    # 1. Data Gen (Ground Truth: Interact [2])
    # Adding a tiny bit more noise to help Pure terms 'cheat' if they want to
    gen = SyntheticGenerator(n_samples=2500, input_dim=args.dim, noise_level=0.1)
    dataset, _ = gen.get_data(mode='interact', variant=0)
    X, y = dataset.tensors
    X, y = X.to(device), y.to(device)
    
    baseline = torch.var(y).item()
    print(f"Baseline MSE (Variance): {baseline:.4f}")

    # 2. Define The Three Horses
    
    # Horse A: Pure Only
    struct_pure = {'type': 'neuronseek', 'rank': args.rank, 'pure_indices': [2], 'interact_indices': []}
    
    # Horse B: Interact Only
    struct_int = {'type': 'neuronseek', 'rank': args.rank, 'pure_indices': [], 'interact_indices': [2]}
    
    # Horse C: Mixed (Your Model)
    struct_mixed = {'type': 'neuronseek', 'rank': args.rank, 'pure_indices': [2], 'interact_indices': [2]}

    # 3. Race!
    print("Running Horse A (Pure Only)...")
    loss_pure = train_one_model(X, y, struct_pure, args.lr_pure, args.lr_int, args.epochs, "Pure")
    
    print("Running Horse B (Interact Only)...")
    loss_int = train_one_model(X, y, struct_int, args.lr_pure, args.lr_int, args.epochs, "Interact")
    
    print("Running Horse C (Mixed Model)...")
    loss_mixed = train_one_model(X, y, struct_mixed, args.lr_pure, args.lr_int, args.epochs, "Mixed")

    # 4. Visualization (The Key Part)
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Linear X (Standard View)
    plt.subplot(1, 2, 1)
    plt.plot(loss_pure, label='Pure Only', color='blue', linestyle='--')
    plt.plot(loss_int, label='Interact Only', color='red')
    plt.plot(loss_mixed, label='Mixed Model', color='black', alpha=0.6)
    plt.axhline(y=baseline, color='gray', linestyle=':', label='Baseline')
    plt.yscale('log')
    plt.xlabel('Epochs (Linear)')
    plt.ylabel('MSE (Log)')
    plt.title('Standard View')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    # Subplot 2: Log X (Zoom into the start!)
    plt.subplot(1, 2, 2)
    plt.plot(loss_pure, label='Pure Only', color='blue', linestyle='--')
    plt.plot(loss_int, label='Interact Only', color='red')
    plt.plot(loss_mixed, label='Mixed Model', color='black', alpha=0.6)
    plt.axhline(y=baseline, color='gray', linestyle=':', label='Baseline')
    plt.xscale('log') # <--- THIS IS THE MAGIC
    plt.yscale('log')
    plt.xlabel('Epochs (Log Scale)')
    plt.title('Zoomed Start (Log-X)')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    save_path = f"race_v2_LRp{args.lr_pure}_LRi{args.lr_int}.png"
    plt.savefig(save_path)
    print(f"\nResult saved to {save_path}")

if __name__ == "__main__":
    main()