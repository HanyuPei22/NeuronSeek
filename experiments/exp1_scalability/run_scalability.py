import sys
import os
import time
import argparse
import pandas as pd
import numpy as np
import torch
from func_timeout import func_timeout, FunctionTimedOut

# --- Path Setup ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# --- Imports ---
from neuronseek.searchers.neuronseek_searcher_expand import NeuronSeekSearcherExpand as DiagnosticNeuronSeekSearcher 
from neuronseek.searchers.neuronseek_searcher import NeuronSeekSearcher
from neuronseek.searchers.tnsr_searcher import TNSRSearcher
from neuronseek.searchers.sr_searcher import SRSearcher
from neuronseek.searchers.eql_searcher import EQLSearcher
from neuronseek.searchers.metasymnet_searcher import MetaSymNetSearcher
from neuronseek.utils.data_utils import get_synthetic_data, SyntheticGenerator 
from experiments.common.structure_evaluator import retrain_and_evaluate

def run_experiment(searcher_cls, params, X_train, y_train, X_test, y_test, timeout=60, seed=42, run_info=None):
    """
    Executes experiment. Includes Diagnostic plotting and Searcher Sanity Check.
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # --- 1. Initialization ---
    try:
        model = searcher_cls(**params)
    except Exception as e:
        print(f"    [Init Error] {searcher_cls.__name__}: {e}")
        return 0.0, "Init_Fail", 999.0

    # --- 2. Search Phase (Fit) ---
    start_time = time.time()
    status = "Success"
    
    # Baseline for plotting
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    baseline_mse = torch.var(y_train_t).item() if y_train_t.numel() > 1 else 1.0

    try:
        if isinstance(model, DiagnosticNeuronSeekSearcher):
            # Pass baseline for plots
            func_timeout(timeout, model.fit, args=(X_train, y_train), kwargs={'baseline_mse': baseline_mse})
        else:
            func_timeout(timeout, model.fit, args=(X_train, y_train))
            
    except FunctionTimedOut:
        status = "Timeout"
    except Exception as e:
        print(f"    [Fit Error] {e}")
        status = "Failed"
    
    search_time = time.time() - start_time

    # --- 3. Structure Extraction & Diagnostics ---
    try:
        structure = model.get_structure_info()
        
        # Plot diagnostics if available
        if isinstance(model, DiagnosticNeuronSeekSearcher) and run_info:
            os.makedirs("result/synthetic_data_result/diagnostic_plots", exist_ok=True)
            fname = f"result/synthetic_data_result/diagnostic_plots/diag_D{run_info['dim']}_Trial{run_info['trial']}_{status}.png"
            model.plot_diagnostics(
                title=f"Dim={run_info['dim']} | Trial={run_info['trial']} | {status}",
                filename=fname,
                baseline_mse=baseline_mse
            )
            print(f"    [Diag] Plot saved to {fname}")

    except Exception as e:
        print(f"    [Structure Extraction Error] {e}")
        structure = {'type': 'failed'}

    # --- 4. Retrain Phase (Strict Probe) ---
    mse = 999.0
    
    if status != "Init_Fail":
        try:
            # [Sanity Check] Check Searcher's direct performance on Test Set
            # This helps distinguish if the failure is in Search (bad structure) or Eval (bad retraining)
            if hasattr(model, 'agent'):
                model.agent.eval() # Use learned BN stats
                with torch.no_grad():
                    xt = torch.tensor(X_test, dtype=torch.float32).to(model.device)
                    yt = torch.tensor(y_test, dtype=torch.float32).reshape(-1,1).to(model.device)
                    pred = model.agent(xt, training=False)
                    direct_mse = torch.nn.functional.mse_loss(pred, yt).item()
                print(f"    [Sanity Check] Searcher Direct Test MSE: {direct_mse:.4f}")

            if structure.get('type') == 'neuronseek':
                print(f"    [Structure] Pure: {structure.get('pure_indices')} | Int: {structure.get('interact_indices')}")

            mse = retrain_and_evaluate(
                model, 
                structure, 
                X_train, y_train, 
                X_test, y_test, 
                epochs=150 
            )
            
            # Clip extreme values
            if np.isnan(mse) or np.isinf(mse) or mse > 1e6:
                mse = 1e6
                
        except Exception as e:
            print(f"    [Retrain Error] {e}")

    return search_time, status, mse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Scalability Experiments")
    parser.add_argument('--dims', type=int, nargs='+', default=[10, 30, 50, 100], help='Dimensions')
    parser.add_argument('--methods', type=str, nargs='+', 
                        default=['NeuronSeek', 'TN-SR', 'Standard-SR', 'EQL', 'MetaSymNet'],
                        help='Methods')
    parser.add_argument('--n_trials', type=int, default=3, help='Trials')
    parser.add_argument('--start_seed', type=int, default=42, help='Seed')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout')
    parser.add_argument('--output', type=str, default='result/synthetic_data_result/scalability_strict_results.csv', help='CSV Output')
    return parser.parse_args()

def main():
    args = parse_arguments()
    results = []
    
    print(f"Starting Experiment: Trials={args.n_trials}, Dims={args.dims}")

    for D in args.dims:
        print(f"\n{'='*60}\nTesting Dimension: {D}\n{'='*60}")
        
        # [Config] Use 'interact' mode for fair penalty comparison
        gen = SyntheticGenerator(n_samples=2500, input_dim=D)
        dataset, _ = gen.get_data('hybrid', 1) 
        X, y = dataset.tensors
        X = X.numpy()
        y = y.numpy()
        
        split = 2000
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
        
        all_methods = {
            "NeuronSeek": (DiagnosticNeuronSeekSearcher, {
                'input_dim': D, 
                'epochs': 200, 
                'rank': 8, 
                'reg_lambda': 0.05
            }),
            "TN-SR": (TNSRSearcher, {'input_dim': D, 'population_size': 1000, 'generations': 10}),
            "Standard-SR": (SRSearcher, {'input_dim': D, 'population_size': 1000, 'generations': 10}),
            "EQL": (EQLSearcher, {'input_dim': D, 'epochs': 500}),
            "MetaSymNet": (MetaSymNetSearcher, {'input_dim': D, 'time_limit': args.timeout})
        }
        
        protocol = [(name, all_methods[name][0], all_methods[name][1]) 
                    for name in args.methods if name in all_methods]

        for name, cls, params in protocol:
            print(f">>> Method: {name}")
            
            trial_mses = []
            trial_times = []
            success_count = 0
            
            for i in range(args.n_trials):
                current_seed = args.start_seed + i
                run_info = {'dim': D, 'trial': i+1}
                
                t, stat, mse = run_experiment(
                    cls, params, 
                    X_train, y_train, 
                    X_test, y_test, 
                    timeout=args.timeout, 
                    seed=current_seed,
                    run_info=run_info 
                )
                
                trial_mses.append(mse)
                trial_times.append(t)
                
                if stat == "Success" or (stat == "Timeout" and mse < 1e5):
                    success_count += 1
                
                print(f"    Trial {i+1}: Time={t:.1f}s | MSE={mse:.4f} | ({stat})")
            
            mean_mse = np.mean(trial_mses)
            std_mse = np.std(trial_mses)
            mean_time = np.mean(trial_times)
            
            print(f"  [AVG] {name}: MSE = {mean_mse:.4f} ± {std_mse:.4f}")
            
            row = {
                'Dim': D, 'Method': name,
                'MSE_Mean': mean_mse, 'MSE_Std': std_mse,
                'Time_Mean': mean_time, 'Success_Rate': success_count / args.n_trials
            }
            results.append(row)

    df = pd.DataFrame(results)
    # df.to_csv(args.output, index=False)
    print(df)

if __name__ == "__main__":
    main()