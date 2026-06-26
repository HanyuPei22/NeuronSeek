import sys
import os
import torch
import numpy as np
import argparse
import pandas as pd

# Adjust path to project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from neuronseek.utils.synthetic_data import SyntheticGenerator
from experiments.common.structure_evaluator import retrain_and_evaluate

class DummySearcher:
    """
    A hollow shell to satisfy retrain_and_evaluate's interface requirement.
    """
    def __init__(self):
        pass

def parse_arguments():
    parser = argparse.ArgumentParser(description="NeuronSeek Evaluator Diagnostic Tool (Multi-Rank)")
    
    # --- Experiment Settings ---
    parser.add_argument('--dim', type=int, default=100, help='Input dimension (D)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=100, help='Retraining epochs')
    
    # --- Ground Truth Settings ---
    parser.add_argument('--data_mode', type=str, default='pure', choices=['pure', 'interact'], 
                        help='Type of ground truth data')
    parser.add_argument('--data_variant', type=int, default=0, 
                        help='Variant index')
    
    # --- Model Structure Settings ---
    # [IMPROVED] Accept multiple ranks
    parser.add_argument('--ranks', type=int, nargs='+', default=[1, 2, 4, 8], 
                        help='List of ranks to test (e.g. --ranks 1 8)')
    
    parser.add_argument('--pure', type=int, nargs='*', default=[], 
                        help='List of Pure orders to enable (e.g. --pure 1 2)')
    parser.add_argument('--interact', type=int, nargs='*', default=[], 
                        help='List of Interaction orders to enable (e.g. --interact 2)')

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # 1. Setup Environment & Data (Fixed for all ranks to ensure fairness)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"\n{'='*60}")
    print(f"EVALUATOR DIAGNOSTIC | Dim={args.dim} | Ranks={args.ranks}")
    print(f"Target Structure: Pure={args.pure}, Interact={args.interact}")
    print(f"{'='*60}")

    gen = SyntheticGenerator(n_samples=3000, input_dim=args.dim, noise_level=0.01)
    dataset, truth = gen.get_data(mode=args.data_mode, variant=args.data_variant)
    
    X, y = dataset.tensors
    X = X.numpy()
    y = y.numpy()
    
    split = 2000
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    baseline_mse = np.var(y_test)
    print(f"\n[Data] Ground Truth: {truth}")
    print(f"[Data] Baseline MSE (Variance): {baseline_mse:.5f}\n")
    
    results = []

    # 2. Iterate over Ranks
    for r in args.ranks:
        print(f">>> Testing Rank = {r}...")
        
        structure_info = {
            'type': 'neuronseek',
            'rank': r,
            'pure_indices': args.pure,
            'interact_indices': args.interact
        }
        
        # Calculate Parameter Count Estimation for CP Decomposition
        # Pure: D * #Orders
        # Interact: Rank * D * #Orders (Simplification, technically Rank*D + Rank*D... per order)
        # Actually in Probe: 
        #   Pure: D * 1 (per order)
        #   Interact: (D * Rank * 1) * Order_Value (Because we have 'Order' factors)
        
        num_params_pure = len(args.pure) * args.dim
        num_params_int = 0
        for order in args.interact:
            # Interaction layer creates 'order' number of factors, each is [D, Rank]
            num_params_int += order * args.dim * r
            
        total_params = num_params_pure + num_params_int
        
        print(f"    Params: ~{total_params} (Pure: {num_params_pure}, Int: {num_params_int})")
        
        # Run Retrain
        try:
            # Re-instantiate dummy model to clear any state if needed (though structure_info drives it)
            dummy = DummySearcher()
            mse = retrain_and_evaluate(
                dummy,
                structure_info,
                X_train, y_train,
                X_test, y_test,
                epochs=args.epochs
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            mse = 999.0
            
        print(f"    -> MSE: {mse:.5f}")
        
        # Verdict
        if mse < 0.05 * baseline_mse:
            verdict = "PERFECT"
        elif mse < baseline_mse:
            verdict = "OK (Learning)"
        else:
            verdict = "FAIL (Noise)"
            
        results.append({
            'Rank': r, 
            'Params': total_params,
            'MSE': mse,
            'vs_Baseline': f"{mse/baseline_mse:.2f}x",
            'Verdict': verdict
        })
        print(f"    Status: {verdict}\n")

    # 3. Final Report
    print(f"{'='*60}")
    print("FINAL SUMMARY REPORT")
    print(f"{'='*60}")
    df = pd.DataFrame(results)
    print(df)
    print(f"\nBaseline MSE: {baseline_mse:.5f}")

if __name__ == "__main__":
    main()