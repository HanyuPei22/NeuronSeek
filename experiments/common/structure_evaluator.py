import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class StructuralProbe(nn.Module):
    """
    Strict Structural Probe (Fixed Rank, Active Orders Only).
    """
    def __init__(self, input_dim, structure_info, num_classes=1):
        super().__init__()
        # Force integer types
        input_dim = int(input_dim)
        num_classes = int(num_classes)
        
        self.structure = structure_info
        self.num_classes = num_classes
        self.mode = 'linear_subset' # Default fallback
        
        type_ = structure_info.get('type', 'explicit_terms')
        
        # ======================================================================
        # Mode A: NeuronSeek (Strict CP, No Head)
        # ======================================================================
        if type_ == 'neuronseek':
            self.mode = 'neuronseek'

            self.rank = int(structure_info.get('rank', 8)) 
            # Sort orders to ensure alignment
            self.p_ords = sorted(structure_info.get('pure_indices', []))
            self.i_ords = sorted(structure_info.get('interact_indices', []))

            self.bns_pure = nn.ModuleList()
            self.bns_int = nn.ModuleList()

            # BN layers track the output dimension (num_classes)
            for _ in self.i_ords:
                self.bns_int.append(nn.BatchNorm1d(num_classes, affine=True))
            for _ in self.p_ords:
                self.bns_pure.append(nn.BatchNorm1d(num_classes, affine=True))
            
            # Interaction Parameters
            self.interact_modules = nn.ModuleDict()
            std_dev = 0.05
            for order in self.i_ords:
                factors = nn.ParameterList([
                    nn.Parameter(torch.randn(input_dim, self.rank, num_classes) * std_dev) 
                    for _ in range(order)
                ])
                self.interact_modules[str(order)] = factors

            # Pure Parameters
            self.pure_modules = nn.ParameterDict()
            for order in self.p_ords:
                self.pure_modules[str(order)] = nn.Parameter(
                    torch.randn(input_dim, num_classes) * std_dev
                )
            
            # Global bias
            self.bias = nn.Parameter(torch.zeros(num_classes))

        # ======================================================================
        # Mode B: Explicit Terms (EQL / Baseline / SR)
        # ======================================================================
        elif type_ == 'explicit_terms':
            terms = structure_info.get('terms', [])
            if terms and ('gplearn_raw' in terms[0].get('type', '') or 'raw_formula' in structure_info):
                 self.mode = 'symbolic_scaling'
                 self.scale = nn.Linear(1, 1) 
                 return

            self.active_indices = []
            for term in terms:
                self.active_indices.extend(term.get('indices', []))
            
            self.active_indices = sorted(list(set(self.active_indices)))
            if not self.active_indices: self.active_indices = [0]
            
            self.register_buffer('selected_idxs', torch.tensor(self.active_indices, dtype=torch.long))
            self.linear = nn.Linear(len(self.active_indices), num_classes) 
            self.mode = 'linear_subset'

    def forward(self, x, formula_pred=None):
        # 1. NeuronSeek Logic
        if self.mode == 'neuronseek':
            batch_size = x.size(0)
            output = self.bias.unsqueeze(0).expand(batch_size, -1).clone()
            
            # --- Active Pure Orders ---
            # [FIXED LOGIC] Apply BN *AFTER* projection to match Searcher
            for i, order in enumerate(self.p_ords):
                order_str = str(order)
                weight = self.pure_modules[order_str]
                
                term = x if order == 1 else x.pow(order)
                
                # 1. Projection: [Batch, D] @ [D, C] -> [Batch, C]
                term_proj = term @ weight
                
                # 2. BatchNorm: Applied on [Batch, C]
                term_norm = self.bns_pure[i](term_proj)

                output = output + term_norm

            # --- Active Interaction Orders ---
            for i, order in enumerate(self.i_ords):
                order_str = str(order)
                factors = self.interact_modules[order_str]
                
                projections = [torch.einsum('bd, drc -> brc', x, u) for u in factors]
                combined = projections[0]
                for p in projections[1:]:
                    combined = combined * p
                
                # 1. Summation: [Batch, R, C] -> [Batch, C]
                term_sum = torch.sum(combined, dim=1)
                
                # 2. BatchNorm: Applied on [Batch, C]
                term_norm = self.bns_int[i](term_sum)
                
                output = output + term_norm
            
            return output
            
        # 2. Linear Subset Logic
        elif self.mode == 'linear_subset':
            x_sel = torch.index_select(x, 1, self.selected_idxs)
            return self.linear(x_sel)
            
        # 3. Symbolic Scaling Logic
        elif self.mode == 'symbolic_scaling':
            if formula_pred is None: 
                batch_size = x.size(0) if x is not None else 1
                return torch.zeros(batch_size, 1).to(self.scale.weight.device)
            return self.scale(formula_pred)
            
        return torch.zeros(x.shape[0], 1).to(x.device)

def retrain_and_evaluate(searcher, structure_info, X_train, y_train, X_test, y_test, epochs=100):
    """
    Retrains the discovered structure.
    Separates logic for standard mini-batch training vs symbolic full-batch scaling.
    """
    input_dim = X_train.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Determine Num Classes
    if y_train.ndim > 1 and y_train.shape[1] > 1:
         num_classes = y_train.shape[1]
         loss_fn = nn.MSELoss() 
    else:
         num_classes = 1
         yt_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
         yt_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(device)
         loss_fn = nn.MSELoss()

    Xt_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    Xt_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # 2. Build Probe
    try:
        probe = StructuralProbe(
            input_dim, 
            structure_info, 
            num_classes=num_classes
        ).to(device)
    except Exception as e:
        print(f"    [Probe Error] {e}")
        return 999.0

    # 3. Symbolic Pre-calculation (if needed)
    sym_train = None
    sym_test = None
    
    if probe.mode == 'symbolic_scaling':
        try:
            # Use the searcher to predict raw values
            raw_pred_train = searcher.predict(X_train)
            raw_pred_test = searcher.predict(X_test)
            
            sym_train = torch.tensor(raw_pred_train, dtype=torch.float32).reshape(-1, 1).to(device)
            sym_test = torch.tensor(raw_pred_test, dtype=torch.float32).reshape(-1, 1).to(device)
        except Exception as e:
            print(f"    [Symbolic Predict Error] {e}")
            return 999.0

    # 4. Training Loop
    optimizer = optim.Adam(probe.parameters(), lr=0.01, weight_decay=1e-5)
    probe.train()

    # --- Branch A: Symbolic Scaling (Full Batch) ---
    if probe.mode == 'symbolic_scaling':
        for _ in range(epochs):
            optimizer.zero_grad()
            # Pass None for x, use pre-calculated formula_pred
            pred = probe(None, formula_pred=sym_train)
            loss = loss_fn(pred, yt_train)
            loss.backward()
            optimizer.step()

    # --- Branch B: Standard Mini-batch Training ---
    else:
        # Prepare Dataset
        if num_classes == 1:
            dataset = TensorDataset(Xt_train, yt_train)
        else:
            target_t = torch.tensor(y_train, dtype=torch.float32).to(device)
            dataset = TensorDataset(Xt_train, target_t)
            
        loader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        for _ in range(epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                pred = probe(bx)
                loss = loss_fn(pred, by)
                loss.backward()
                optimizer.step()

    # 5. Final Evaluation
    probe.eval()
    with torch.no_grad():
        if probe.mode == 'symbolic_scaling':
            pred = probe(None, formula_pred=sym_test)
        else:
            pred = probe(Xt_test)
        
        # Calculate MSE
        if num_classes == 1:
            mse = loss_fn(pred, yt_test).item()
        else:
            target_test = torch.tensor(y_test, dtype=torch.float32).to(device)
            mse = loss_fn(pred, target_test).item()
            
    return mse