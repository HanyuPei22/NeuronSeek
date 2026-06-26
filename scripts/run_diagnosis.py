import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import torch
import numpy as np
from neuronseek.utils.data_utils import SyntheticGenerator
from neuronseek.searchers.neuronseek_searcher_expand import NeuronSeekSearcherExpand as DiagnosticNeuronSeekSearcher

def main():
    # Settings
    DIM = 100
    N_SAMPLES = 2500
    MODE = 'interact' # Use 'interact' for pure Interaction validation
    VARIANT = 0       # 0 = Order 2 Only
    
    print(f"{'='*60}")
    print(f"DIAGNOSTIC RUN: Dim={DIM}, Mode={MODE}, Variant={VARIANT}")
    print(f"{'='*60}")

    # 1. Generate Data & Print Formula
    gen = SyntheticGenerator(N_SAMPLES, DIM, noise_level=0.01)
    dataset, truth = gen.get_data(MODE, VARIANT)
    X, y = dataset.tensors
    X = X.numpy()
    y = y.numpy()
    
    print(f"\n>>> Data Generation Logic:")
    print(f"    {gen.current_formula}")
    print(f"    Ground Truth Indices: {truth}")

    # 2. Calculate Baseline MSE
    y_tensor = torch.tensor(y)
    baseline_var = torch.var(y_tensor).item()
    print(f"\n>>> Baseline MSE (Constant Prediction): {baseline_var:.6f}")
    print(f"    (Failure Threshold: Loss > {baseline_var:.4f})")

    # 3. Initialize Diagnostic Searcher
    print(f"\n>>> Initializing Diagnostic Searcher...")
    searcher = DiagnosticNeuronSeekSearcher(
        input_dim=DIM,
        num_classes=1,
        rank=8,
        epochs=200,      
        batch_size=64,
        reg_lambda=0.05  
    )

    # 4. Run Search
    print(f"\n>>> Fitting Model...")
    searcher.fit(X, y, baseline_mse=baseline_var)

    # 5. Report Structure
    struct = searcher.get_structure_info()
    print(f"\n>>> Final Structure Discovered:")
    print(f"    Pure Indices:     {struct['pure_indices']}")
    print(f"    Interact Indices: {struct['interact_indices']}")

    # 6. Generate Plot
    print(f"\n>>> Generating Diagnostic Plot...")
    searcher.plot_diagnostics(
        title=f"NeuronSeek Diag: {gen.current_formula}",
        filename="neuronseek_diagnostic_report.png",
        baseline_mse=baseline_var
    )

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()