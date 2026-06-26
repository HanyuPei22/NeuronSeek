import sys
import os
import torch
import numpy as np
from tabulate import tabulate

# [Setup] Add project root to sys.path to import src modules
# Assuming structure: /home/ET/hypei/NeuronSeek/experiments/synthetic_data/this_script.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from neuronseek.models.search_agent import SearchAgent
from neuronseek.utils.synthetic_data import SyntheticGenerator

# --- Configuration & Hyperparameters ---
# Tweak these if the model fails to capture weak signals or keeps noise.
HYPERPARAMS = {
    'input_dim': 10,       # Dimension of X
    'num_classes': 1,      # Regression task
    'rank': 8,             # CP-Rank for Interaction Stream
    'poly_order': 5,       # Challenge: Search up to order 5
    'reg_lambda': 0.01,    # Strength of Group Lasso (Increase if too much noise)
    'threshold': 0.1,     # Pruning threshold relative to max energy (Lower if weak signals are lost)
    'epochs': 50,          # Training duration
    'lr': 1e-3,            # Learning rate
    'noise_level': 0.01    # Observation noise in y
}

def run_benchmark():
    print(f"\n{'='*80}")
    print(f"STRidge Synthetic Benchmark | Project: NeuronSeek")
    print(f"Config: Order={HYPERPARAMS['poly_order']}, Reg={HYPERPARAMS['reg_lambda']}, Threshold={HYPERPARAMS['threshold']}")
    print(f"{'='*80}\n")

    modes = ['pure', 'interact', 'hybrid']
    variants = 5 # 0 to 4
    results_log = []

    # Iterate through all scenarios
    for mode in modes:
        for v in range(variants):
            case_id = f"{mode.upper()}-V{v}"
            
            # 1. Data Generation
            gen = SyntheticGenerator(n_samples=2500, input_dim=HYPERPARAMS['input_dim'], 
                                     noise_level=HYPERPARAMS['noise_level'])
            dataset, truth = gen.get_data(mode=mode, variant=v)
            
            # 2. Model Initialization
            agent = SearchAgent(
                input_dim=HYPERPARAMS['input_dim'],
                num_classes=HYPERPARAMS['num_classes'],
                rank=HYPERPARAMS['rank'],
                poly_order=HYPERPARAMS['poly_order'],
                reg_lambda=HYPERPARAMS['reg_lambda'],
                task_type='regression'
            )
            
            # 3. Training
            # Suppress per-epoch printing to keep output clean, only print final result
            agent.fit_stridge(dataset, epochs=HYPERPARAMS['epochs'], 
                              batch_size=64, device='cuda' if torch.cuda.is_available() else 'cpu')
            
            # 4. Evaluation (Discovery)
            # Use the explicit threshold defined in HYPERPARAMS
            # Note: You might need to expose threshold in get_discovered_orders or ensure fit_stridge uses it.
            # Assuming fit_stridge performs pruning at epoch//2 using internal logic.
            found = agent.get_discovered_orders()
            
            # 5. Comparison Logic
            truth_pure = sorted(truth['pure'])
            found_pure = sorted(found['pure'])
            truth_int = sorted(truth['interact'])
            found_int = sorted(found['interact'])

            # Strict Exact Match Check
            is_pure_match = (truth_pure == found_pure)
            is_int_match = (truth_int == found_int)
            
            # Detailed Status
            if is_pure_match and is_int_match:
                status = "PERFECT"
            elif set(truth_pure).issubset(set(found_pure)) and set(truth_int).issubset(set(found_int)):
                status = "NOISY" # Found truth but failed to prune extra terms
            else:
                status = "MISSING" # Failed to find truth
            
            # Console Log for Real-time Monitoring
            print(f"[{case_id}] Status: {status}")
            print(f"  GT:   Pure{truth_pure} | Int{truth_int}")
            print(f"  Pred: Pure{found_pure} | Int{found_int}")
            print(f"  {'-'*40}")

            results_log.append([
                case_id, 
                str(truth_pure), str(found_pure), 
                str(truth_int), str(found_int), 
                status
            ])

    # --- Final Report Table ---
    headers = ["Case ID", "Truth(P)", "Found(P)", "Truth(I)", "Found(I)", "Status"]
    print(f"\n{'='*80}")
    print(f"FINAL BENCHMARK REPORT")
    print(f"{'='*80}")
    print(tabulate(results_log, headers=headers, tablefmt="grid"))
    
    # Summary Statistics
    perfect = sum(1 for r in results_log if r[-1] == "PERFECT")
    noisy = sum(1 for r in results_log if r[-1] == "NOISY")
    missing = sum(1 for r in results_log if r[-1] == "MISSING")
    
    print(f"\nSummary:")
    print(f"  PERFECT: {perfect}/{len(results_log)}")
    print(f"  NOISY:   {noisy}/{len(results_log)} (Try increasing reg_lambda)")
    print(f"  MISSING: {missing}/{len(results_log)} (Try decreasing reg_lambda or threshold)")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    run_benchmark()