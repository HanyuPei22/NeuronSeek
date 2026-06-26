import numpy as np
import time
from scipy.optimize import minimize
from .base import BaseStructureSearcher
from neuronseek.utils.seeds import setup_seed

# --- Safe Math Operations ---
MAX_VAL = 1e9

def safe_clip(x):
    x = np.nan_to_num(x, nan=0.0, posinf=MAX_VAL, neginf=-MAX_VAL)
    return np.clip(x, -MAX_VAL, MAX_VAL)

def safe_div(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore'):
        res = x1 / (x2 + np.sign(x2 + 1e-9) * 1e-6)
    return safe_clip(res)

def safe_log(x):
    return np.log(np.abs(x) + 1e-6)

def safe_sqrt(x):
    return np.sqrt(np.abs(x) + 1e-6)

def safe_exp(x):
    # Clip to avoid overflow
    return np.exp(np.clip(x, -20, 20)) 

def safe_mul(x1, x2):
    return safe_clip(x1 * x2)

class OfficialMetaSymNetEngine:
    def __init__(self, input_dim, iterations=20, time_limit=60):
        self.input_dim = input_dim
        self.iterations = iterations
        self.time_limit = time_limit
        self.ops = ['+', '*', '/', 'sin', 'cos', 'exp', 'log', 'sqrt']
        # Initial Structure: s-s-x-x-s-x-x (Post-order/RPN compliant)
        self.layout = ['s', 's', 'x', 'x', 's', 'x', 'x'] 
        self.best_formula_str = "0"
        self.global_best_r2 = -float('inf')
        self.start_time = None
        self.SX = [f'x{i}' for i in range(input_dim)]

    def _softmax(self, x, c=2):
        x_shift = x - np.max(x)
        exp_x = np.exp(np.clip(c * x_shift, -20, 20))
        return exp_x / np.sum(exp_x)

    def _get_op_val(self, op, z, x):
        if op == '+': return z + x
        if op == '-': return z - x
        if op == '*': return safe_mul(z, x)
        if op == '/': return safe_div(z, x)
        if op == 'sin': return np.sin(z)
        if op == 'cos': return np.cos(z)
        if op == 'exp': return safe_exp(z)
        if op == 'log': return safe_log(z)
        if op == 'sqrt': return safe_sqrt(z)
        return z

    def _sy(self, x, x_1):
        res = [self._get_op_val(op, x, x_1) for op in self.ops]
        return np.array(res)

    def _pro_t(self, layout, X, Params, capture_values=False):
        """
        Forward Pass. Returns prediction.
        """
        stack = []
        node_values = {} 
        
        n_s = layout.count('s')
        n_x = layout.count('x')
        
        len_s = len(self.ops) + 2
        len_x = self.input_dim + 2
        
        split_idx = n_s * len_s
        Params_S = Params[:split_idx]
        Params_X = Params[split_idx:]
        
        ptr_s = n_s
        ptr_x = n_x
        
        # Reverse traversal (Post-order evaluation)
        for i in range(len(layout) - 1, -1, -1):
            node_type = layout[i]
            
            if node_type == 'x':
                p_chunk = Params_X[(ptr_x - 1) * len_x : ptr_x * len_x]
                ptr_x -= 1
                
                scale, bias = p_chunk[0], p_chunk[-1]
                w_feats = p_chunk[1:-1]
                
                probs = self._softmax(w_feats, c=5)
                selected_x = np.dot(probs, X)
                res = safe_clip(scale * selected_x + bias)
                
                stack.append(res)
                if capture_values: node_values[i] = res
                
            elif node_type == 's':
                p_chunk = Params_S[(ptr_s - 1) * len_s : ptr_s * len_s]
                ptr_s -= 1
                
                scale, bias = p_chunk[0], p_chunk[-1]
                w_ops = p_chunk[1:-1]
                
                probs = self._softmax(w_ops, c=5)
                
                if len(stack) < 2: return np.zeros(X.shape[1])
                op2 = stack.pop()
                op1 = stack.pop()
                
                candidates = self._sy(op1, op2)
                res_op = np.dot(probs, candidates)
                res = safe_clip(scale * res_op + bias)
                
                stack.append(res)
                if capture_values: node_values[i] = res
        
        return stack[0] if stack else np.zeros(X.shape[1])

    def _extract_expression(self, params, layout):
        """
        Discrete structure extraction based on max weights.
        """
        L_discrete = []
        
        # [FIX] Define counts locally
        n_s = layout.count('s')
        n_x = layout.count('x')
        
        len_s = len(self.ops) + 2
        len_x = self.input_dim + 2
        
        split = n_s * len_s
        Params_S = params[:split]
        Params_X = params[split:]
        
        # Pointers start at end to match _pro_t traversal order
        ptr_s = n_s
        ptr_x = n_x
        
        stack = []
        
        # Traverse in reverse to rebuild the expression stack
        for i in range(len(layout) - 1, -1, -1):
            node_type = layout[i]
            
            if node_type == 'x':
                p_chunk = Params_X[(ptr_x - 1) * len_x : ptr_x * len_x]
                ptr_x -= 1
                w_feats = p_chunk[1:-1]
                
                # Select best feature
                best_idx = np.argmax(w_feats)
                stack.append(self.SX[best_idx])
                
            elif node_type == 's':
                p_chunk = Params_S[(ptr_s - 1) * len_s : ptr_s * len_s]
                ptr_s -= 1
                w_ops = p_chunk[1:-1]
                
                # Select best op
                best_op = self.ops[np.argmax(w_ops)]
                
                if len(stack) >= 2:
                    r = stack.pop()
                    l = stack.pop()
                    if best_op in ['sin', 'cos', 'exp', 'log', 'sqrt']:
                        stack.append(f"{best_op}({l})")
                    else:
                        stack.append(f"({l} {best_op} {r})")
                else:
                    stack.append("0")
                    
        return stack[0] if stack else "0"

    def _e2n(self, formula_str):
        # Placeholder: Reset layout if optimization fails to improve
        return ['s', 's', 'x', 'x', 's', 'x', 'x']

    def fit(self, X, y, callback=None):
            """
            Main optimization loop with structure evolution.
            X shape: (Features, Samples)
            """
            X_T = X 
            y_flat = y.ravel()
            self.start_time = time.time()
            
            # Standardization (Crucial for Gradient Descent)
            y_mean, y_std = np.mean(y_flat), np.std(y_flat)
            if y_std < 1e-6: y_std = 1.0
            y_norm = (y_flat - y_mean) / y_std

            # Optimization Helper: Loss on Normalized Data
            def loss_func(params):
                try:
                    pred = self._pro_t(self.layout, X_T, params)
                    return np.mean((y_norm - pred)**2)
                except: return 1e9

            # Callback Helper: Handles Internal Timeout Check
            def optim_callback(xk):
                # Internal Timeout Check
                if time.time() - self.start_time > self.time_limit:
                    raise StopIteration("Time limit reached")

            # --- Evolution Loop ---
            for iteration in range(self.iterations):
                if time.time() - self.start_time > self.time_limit: break
                
                # 1. Prepare Parameters
                n_s = self.layout.count('s')
                n_x = self.layout.count('x')
                num_params = n_s * (len(self.ops) + 2) + n_x * (self.input_dim + 2)
                
                try:
                    # 2. Optimization (Continuous Relaxation)
                    x0 = np.random.randn(num_params) * 0.1 # Small random init
                    
                    res = minimize(loss_func, x0, method='L-BFGS-B', 
                                callback=optim_callback, 
                                options={'maxiter': 50}) # Short burst optimization
                    
                    # 3. Evaluation & Structure Extraction
                    pred_norm = self._pro_t(self.layout, X_T, res.x)
                    pred_real = pred_norm * y_std + y_mean
                    mse = np.mean((y_flat - pred_real)**2)
                    r2 = 1 - mse / (y_std**2)
                    
                    # Update Global Best
                    if r2 > self.global_best_r2:
                        self.global_best_r2 = r2
                        # Extract discrete structure from continuous params
                        self.best_formula_str = self._extract_expression(res.x, self.layout)
                    
                    # 4. External Callback (Checkpoint Manager)
                    if callback:
                        should_stop = callback(iteration=iteration, loss=mse)
                        if should_stop:
                            break

                    # 5. Evolution Strategy
                    # If not perfect, evolve structure for next iteration
                    if r2 < 0.999:

                        current_formula_str = self._extract_expression(res.x, self.layout)
                        new_layout = self._e2n(current_formula_str)

                        # Safety check: prevent empty or exploding layouts
                        if len(new_layout) < 3:
                            self.layout = ['s', 's', 'x', 'x', 's', 'x', 'x']
                        elif len(new_layout) > 50:
                            self.layout = ['s', 's', 'x', 'x', 's', 'x', 'x']
                        else:
                            self.layout = new_layout

                except StopIteration:
                    # Caught stop signal from callback or timeout
                    break

                except Exception as e:

                    self.layout = ['s', 's', 'x', 'x', 's', 'x', 'x']

class OfficialMetaSymNetSearcher(BaseStructureSearcher):
    def __init__(self, input_dim: int, iterations=20, time_limit=60):
        super().__init__(input_dim)
        self.engine = OfficialMetaSymNetEngine(input_dim, iterations, time_limit)

    def fit(self, X: np.ndarray, y: np.ndarray, callback=None) -> None:
        setup_seed(42)
        if X.shape[0] != self.engine.input_dim:
            X = X.T
        self.engine.fit(X, y, callback=callback)

    def get_structure_info(self):
        return {
            'type': 'explicit_terms', 
            'raw_formula': self.engine.best_formula_str,
            'terms': []
        }