from sched import scheduler
import sys
import os
import torch
import torch.optim as optim
from tabulate import tabulate

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from src import core
from neuronseek.models.sparse_search_agent import SparseSearchAgent
from neuronseek.utils.synthetic_data import SyntheticGenerator

def run_benchmark():
    print(f"{'='*60}")
    print(f"Sparse Differentiable Search (Warm-up + Standard Init)")
    print(f"{'='*60}\n")
    
    # Configuration
    LAMBDA_L0 = 0.6
    EPOCHS = 100          
    WARMUP_EPOCHS = 40    
    LR = 0.02             
    
    modes = ['pure', 'interact', 'hybrid']
    variants = 5
    results = []

    for mode in modes:
        for v in range(variants):
            case_id = f"{mode.upper()}-V{v}"
            
            # Data Generation
            gen = SyntheticGenerator(n_samples=2500, input_dim=10, noise_level=0.01)
            train_d, truth = gen.get_data(mode, v)
            train_loader = torch.utils.data.DataLoader(train_d, batch_size=64, shuffle=True)
            
            # Agent Initialization
            agent = SparseSearchAgent(input_dim=10)
            if torch.cuda.is_available(): agent.cuda()

            gate_params = [p for n, p in agent.named_parameters() if 'gates' in n]
            core_params = [p for n, p in agent.named_parameters() if 'gates' not in n]
            
            # Initialize optimizer with groups
            optimizer = optim.Adam([
                {'params': core_params, 'lr': 0.01},
                {'params': gate_params, 'lr': 0.05} 
            ], lr=LR)
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
            
            # Freeze gates initially using requires_grad
            for param in gate_params:
                param.requires_grad = False

            agent.train()
            for epoch in range(EPOCHS):
                
                # [State Switch] Unfreeze gates exactly when warmup ends
                if epoch == WARMUP_EPOCHS:
                    print(f"[{case_id}] Warmup finished at Epoch {epoch}. Unfreezing Gates.")
                    for param in gate_params:
                        param.requires_grad = True

                for X, y in train_loader:
                    if torch.cuda.is_available(): X, y = X.cuda(), y.cuda()
                    optimizer.zero_grad()
                    
                    # Forward
                    preds = agent(X, training=True)
                    mse_loss = torch.nn.functional.mse_loss(preds, y)
                    
                    # Loss Calculation
                    if epoch < WARMUP_EPOCHS:
                        loss = mse_loss
                    else:
                        reg_loss = agent.calculate_regularization()
                        loss = mse_loss + LAMBDA_L0 * reg_loss
                    
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
            
            # Inspection and Evaluation
            agent.inspect_gates()
            found_p, found_i = agent.get_structure()
            truth_p = sorted(truth['pure'])
            truth_i = sorted(truth['interact'])
            
            if found_p == truth_p and found_i == truth_i:
                status = "PERFECT"
            elif set(truth_p).issubset(set(found_p)) and set(truth_i).issubset(set(found_i)):
                status = "NOISY"
            else:
                status = "MISSING"
                
            print(f"[{case_id}] {status} | GT: P{truth_p} I{truth_i} | Found: P{found_p} I{found_i}")
            results.append([case_id, status, str(truth_p), str(found_p), str(truth_i), str(found_i)])

    # Summary
    print(f"\n{'='*80}")
    print(tabulate(results, headers=['Case', 'Status', 'GT(P)', 'Found(P)', 'GT(I)', 'Found(I)'], tablefmt='grid'))
    
    perfect_count = sum(1 for r in results if r[1] == "PERFECT")
    print(f"\nPERFECT: {perfect_count}/{len(results)}")

if __name__ == "__main__":
    run_benchmark()
