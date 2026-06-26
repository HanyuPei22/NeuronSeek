"""
Convergence Experiment (Step-Based Trigger)
===========================================
Logic:
1. Search Phase: Run until Timeout.
   - Trigger Checkpoint every N steps (epochs/generations).
   - Record Wall-clock Time at that moment.
2. Eval Phase: Retrain extracted structures from checkpoints.
3. Save: Incremental CSV update (safe for running methods one by one).
"""

import sys
import os
import time
import argparse
import pandas as pd
import numpy as np
import torch
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from neuronseek.utils.data_utils import SyntheticGenerator
from neuronseek.searchers.neuronseek_searcher_expand import NeuronSeekSearcherExpand as DiagnosticNeuronSeekSearcher
from neuronseek.searchers.tnsr_searcher import TNSRSearcher
from neuronseek.searchers.sr_searcher import SRSearcher
from neuronseek.searchers.eql_searcher import EQLSearcher
from neuronseek.searchers.official_metasymnet_searcher import OfficialMetaSymNetSearcher
from experiments.common.structure_evaluator import retrain_and_evaluate

# ==============================================================================
# 1. Step-Based Checkpoint Manager
# ==============================================================================

class StepCheckpointManager:
    def __init__(self, method_name, timeout, step_interval, model_wrapper):
        self.method = method_name
        self.timeout = timeout
        self.step_interval = step_interval
        self.wrapper = model_wrapper
        
        self.start_time = time.time()
        self.checkpoints = [] # Stores: {'Time': t, 'Step': s, 'Snapshot': state}

    def __call__(self, **kwargs):
        """
        Called by the searcher every step/epoch.
        kwargs should contain 'epoch', 'generation', or 'iteration'.
        """
        current_time = time.time() - self.start_time

        # 1. Timeout Check (Force Stop)
        if current_time > self.timeout:
            return True

        # 2. Step Check
        # Get current step count (handle different naming conventions)
        current_step = kwargs.get('epoch', kwargs.get('generation', kwargs.get('iteration', None)))

        if current_step is None:
            return False # Should not happen if searchers are instrumented correctly

        # Save logic: Save at step 0, and then every interval
        if current_step == 0 or (current_step % self.step_interval == 0):
            try:
                # Capture Snapshot based on method type
                if self.method == "NeuronSeek":
                    snapshot = deepcopy(self.wrapper.agent.state_dict())
                elif self.method == "EQL":
                    snapshot = deepcopy(self.wrapper.model.state_dict())
                elif self.method == "TN-SR":
                    # TN-SR: capture current best formula directly from engine
                    snapshot = {
                        'best_prog': self.wrapper.engine.best_prog,
                        'global_best': self.wrapper.engine.global_best
                    }
                else:
                    # MetaSymNet, SR: snapshot is the structure dict
                    snapshot = self.wrapper.get_structure_info()

                self.checkpoints.append({
                    'Time': current_time,
                    'Step': current_step,
                    'Snapshot': snapshot
                })
                
                loss_val = kwargs.get('loss', 0.0)
                print(f"  [{self.method}] Snapshot @ Step {current_step} ({current_time:.2f}s) | Train Loss: {loss_val:.4f}")
                
            except Exception as e:
                print(f"  [{self.method}] Snapshot failed: {e}")
                
        return False # Continue training

# ==============================================================================
# 2. Restoration & Eval Logic
# ==============================================================================

def restore_and_get_structure(method, model_cls, params, snapshot):
    """
    Restores a model from a snapshot and returns its structure info.
    """
    model = model_cls(**params)
    
    if method == "NeuronSeek":
        model.agent.load_state_dict(snapshot)
        model.agent.eval()
        return model.get_structure_info()
        
    elif method == "EQL":
        from neuronseek.searchers.eql_searcher import EQLNetwork
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.model = EQLNetwork(params['input_dim'], params.get('hidden_dim', 20)).to(dev)
        model.model.load_state_dict(snapshot)
        return model.get_structure_info()

    elif method == "TN-SR":
        # TN-SR: restore engine state and parse structure
        model.engine.best_prog = snapshot['best_prog']
        model.engine.global_best = snapshot['global_best']
        model.engine.neuron = snapshot['best_prog']  # Set neuron for get_structure_info()
        return model.get_structure_info()

    else:
        # For SR, MetaSymNet, the snapshot IS the structure info
        return snapshot

# ==============================================================================
# 3. Main Experiment Loop
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--timeout', type=int, default=100, help='Max run time in seconds')
    parser.add_argument('--methods', nargs='+', default=['NeuronSeek'], 
                        help='Methods to run: NeuronSeek, TN-SR, EQL, MetaSymNet, SR')
    parser.add_argument('--output_csv', default='result/synthetic_data_result/convergence_step_results.csv')
    parser.add_argument('--eval_epochs', type=int, default=100, help='Epochs for Stage 2 Retraining')
    args = parser.parse_args()

    print(f"=== Convergence Experiment (Step-Based) | Dim={args.dim} | Timeout={args.timeout}s ===\n")

    # 1. Data Generation
    gen = SyntheticGenerator(n_samples=3000, input_dim=args.dim)
    dataset, _ = gen.get_data('hybrid', 3) 
    X, y = dataset.tensors
    X, y = X.numpy(), y.numpy()
    
    split = 2000
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:] # Used for Stage 2 Eval

    # 2. Method Configuration
    # step_interval: How often (in epochs/gens) to save a checkpoint
    # Note: epochs are set high so Timeout controls the end.
    
    configs = {
        "NeuronSeek": {
            'cls': DiagnosticNeuronSeekSearcher,
            'params': {'input_dim': args.dim, 'epochs': 50, 'rank': 8, 'reg_lambda': 0.05},
            'step_interval': 5  # Save every 5 epochs
        },
        "EQL": {
            'cls': EQLSearcher,
            'params': {'input_dim': args.dim, 'epochs': 50},
            'step_interval': 20 # EQL is fast, save every 20 epochs
        },
        "TN-SR": {
            'cls': TNSRSearcher,
            'params': {'input_dim': args.dim, 'population_size': 1000, 'generations': 99999},
            'step_interval': 1  # Genetic algs change every gen, so save every 1
        },
        "MetaSymNet": {
            'cls': OfficialMetaSymNetSearcher,
            # Internal time limit slightly higher to let Manager kill it
            'params': {'input_dim': args.dim, 'iterations': 9999, 'time_limit': args.timeout + 10},
            'step_interval': 1  # Each iteration is a significant step
        },
        "SR": {
            'cls': SRSearcher,
            'params': {'input_dim': args.dim, 'population_size': 1000, 'generations': 99999},
            'step_interval': 1
        }
    }

    results = []
    
    # Filter methods
    selected_methods = {k: v for k, v in configs.items() if k in args.methods}

    for name, cfg in selected_methods.items():
        print(f"\n>>> [Phase 1] Searching: {name}...")
        
        # --- PHASE 1: SEARCH ---
        try:
            model = cfg['cls'](**cfg['params'])
            
            # Step-Based Manager
            manager = StepCheckpointManager(
                method_name=name,
                timeout=args.timeout,
                step_interval=cfg['step_interval'],
                model_wrapper=model
            )
            
            # Start Training
            model.fit(X_train, y_train, callback=manager)
            
        except Exception as e:
            print(f"  CRITICAL: Search crashed for {name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # --- PHASE 2: EVALUATE ---
        num_ckpts = len(manager.checkpoints)
        print(f">>> [Phase 2] Evaluating {num_ckpts} snapshots for {name}...")
        
        for i, ckpt in enumerate(manager.checkpoints):
            try:
                t = ckpt['Time']
                step = ckpt['Step']
                
                # A. Restore Structure
                struct = restore_and_get_structure(name, cfg['cls'], cfg['params'], ckpt['Snapshot'])
                
                # B. Strict Evaluation (Retrain)
                mse = retrain_and_evaluate(
                    model, 
                    struct, 
                    X_train, y_train, 
                    X_test, y_test,
                    epochs=args.eval_epochs
                )
                
                # Sanitize
                if np.isnan(mse) or np.isinf(mse) or mse > 1e6: mse = 1e6
                
                results.append({
                    'Method': name, 
                    'Dim': args.dim, 
                    'Step': step,
                    'Time': t, 
                    'Eval_MSE': mse
                })
                print(f"    [{i+1}/{num_ckpts}] Step {step} (t={t:.1f}s) -> Eval MSE: {mse:.4f}")
                
            except Exception as e:
                print(f"    Eval failed for snapshot {i}: {e}")

    # 3. Save Logic (Incremental)
    if results:
        df_new = pd.DataFrame(results)

        # Load existing results if file exists
        if os.path.exists(args.output_csv):
            try:
                df_old = pd.read_csv(args.output_csv)
                
                # Remove old results for the *current methods* and *current dim*
                # This allows re-running one method without deleting others
                mask = ~((df_old['Method'].isin(args.methods)) & (df_old['Dim'] == args.dim))
                df_old = df_old[mask]
                
                # Concatenate
                df_final = pd.concat([df_old, df_new], ignore_index=True)
            except Exception as e:
                print(f"Warning: Could not read existing CSV ({e}), creating new one.")
                df_final = df_new
        else:
            df_final = df_new

        # Sort and Save
        if 'Step' in df_final.columns:
            df_final = df_final.sort_values(['Method', 'Dim', 'Time']).reset_index(drop=True)
        
        # Save columns
        cols = ['Method', 'Dim', 'Step', 'Time', 'Eval_MSE']
        # Ensure cols exist
        final_cols = [c for c in cols if c in df_final.columns]
        
        df_final[final_cols].to_csv(args.output_csv, index=False)
        print(f"\n✓ Results saved to {args.output_csv}")
        print(f"  Methods updated: {args.methods}")

if __name__ == "__main__":
    main()