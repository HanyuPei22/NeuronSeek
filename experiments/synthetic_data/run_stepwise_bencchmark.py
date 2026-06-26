import sys
import os
import torch
import numpy as np
from tabulate import tabulate

# [Setup] Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from neuronseek.models.stepwise_search_agent import StepwiseSearchAgent
from neuronseek.utils.synthetic_data import SyntheticGenerator

# --- Configuration ---
# Note: reg_lambda and threshold are NOT needed here because 
# the Stepwise Agent uses BIC for automatic selection.
HYPERPARAMS = {
    'input_dim': 10,       
    'num_classes': 1,      
    'rank': 8,             # CP-Rank for Interaction Stream
    'max_order': 5,        # Search up to Order 5
    'n_samples': 2500,     # Important for BIC calculation
    'noise_level': 0.01    
}

def run_benchmark():
    print(f"\n{'='*80}")
    print(f"Dual-Stream Stepwise Selection Benchmark | Project: NeuronSeek")
    print(f"Config: Rank={HYPERPARAMS['rank']}, MaxOrder={HYPERPARAMS['max_order']}, Samples={HYPERPARAMS['n_samples']}")
    print(f"{'='*80}\n")

    modes = ['pure', 'interact', 'hybrid']
    variants = 5 # 0 to 4
    results_log = []

    # Iterate through all scenarios
    for mode in modes:
        for v in range(variants):
            case_id = f"{mode.upper()}-V{v}"
            
            # 1. Data Generation
            # Using the Correct Generator (Diagonal Removed + Standardized)
            gen = SyntheticGenerator(n_samples=HYPERPARAMS['n_samples'], 
                                     input_dim=HYPERPARAMS['input_dim'], 
                                     noise_level=HYPERPARAMS['noise_level'])
            
            # Get separate Train/Val sets for fair BIC evaluation
            train_dataset, truth = gen.get_data(mode=mode, variant=v)
            val_dataset, _ = gen.get_data(mode=mode, variant=v)
            
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
            
            # 2. Model Initialization
            agent = StepwiseSearchAgent(
                input_dim=HYPERPARAMS['input_dim'],
                num_classes=HYPERPARAMS['num_classes'],
                rank=HYPERPARAMS['rank'],
                max_order=HYPERPARAMS['max_order'],
                n_samples=HYPERPARAMS['n_samples']
            )
            
            if torch.cuda.is_available():
                agent.cuda()
            
            # 3. Stepwise Search Process
            # This handles Forward Probing -> Decision -> Backward Pruning internally
            found_pure_set, found_interact_set = agent.fit_stepwise(train_loader, val_loader)
            
            # 4. Result Parsing
            truth_pure = sorted(truth['pure'])
            found_pure = sorted(list(found_pure_set))
            truth_int = sorted(truth['interact'])
            found_int = sorted(list(found_interact_set))

            # 5. Status Determination
            is_pure_match = (truth_pure == found_pure)
            is_int_match = (truth_int == found_int)
            
            if is_pure_match and is_int_match:
                status = "PERFECT"
            elif set(truth_pure).issubset(set(found_pure)) and set(truth_int).issubset(set(found_int)):
                status = "NOISY" # Found truth but includes extras
            elif not set(truth_pure).issubset(set(found_pure)) and not set(truth_int).issubset(set(found_int)):
                 status = "TOTAL_MISS" # Missed both
            else:
                status = "MISSING" # Missed some truth terms
            
            # Console Log
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
    missing = sum(1 for r in results_log if r[-1] == "MISSING") or sum(1 for r in results_log if r[-1] == "TOTAL_MISS")
    
    print(f"\nSummary:")
    print(f"  PERFECT: {perfect}/{len(results_log)}")
    print(f"  NOISY:   {noisy}/{len(results_log)} (BIC penalty might be too weak)")
    print(f"  MISSING: {missing}/{len(results_log)} (BIC penalty might be too strong)")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    run_benchmark()