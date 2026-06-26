import numpy as np
import random
import copy
import operator
import math
from .base import BaseStructureSearcher

# ==============================================================================
# 1. Core Data Structures (Tree Nodes)
# ==============================================================================

class Node:
    def evaluate(self, X):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError
    
    def height(self):
        return 0

class BinaryOp(Node):
    def __init__(self, left, right, op, symbol):
        self.left = left
        self.right = right
        self.op = op
        self.symbol = symbol

    def evaluate(self, X):
        # Safe evaluation with protection against div/0 or overflow
        l_val = self.left.evaluate(X)
        r_val = self.right.evaluate(X)
        try:
            return self.op(l_val, r_val)
        except:
            return np.ones_like(l_val) * 1e9 # Penalty for error

    def __str__(self):
        return f"({self.left} {self.symbol} {self.right})"
    
    def height(self):
        return 1 + max(self.left.height(), self.right.height())

class Feature(Node):
    def __init__(self, feature_idx):
        self.feature_idx = feature_idx

    def evaluate(self, X):
        return X[:, self.feature_idx]

    def __str__(self):
        return f"X{self.feature_idx}"

class Constant(Node):
    def __init__(self, value):
        self.value = value

    def evaluate(self, X):
        return np.full(X.shape[0], self.value)

    def __str__(self):
        return f"{self.value:.3f}"

# ==============================================================================
# 2. Genetic Programming Engine
# ==============================================================================

class SimpleSymbolicRegressor:
    def __init__(self, input_dim, pop_size=1000, generations=20, 
                 tournament_size=20, p_crossover=0.7, p_mutation=0.2, 
                 max_depth=5, const_range=(-1, 1), seed=None):
        self.input_dim = input_dim
        self.pop_size = pop_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.max_depth = max_depth
        self.const_range = const_range
        self.rng = random.Random(seed)
        
        # Operations Map
        self.ops = [
            (operator.add, '+'),
            (operator.sub, '-'),
            (operator.mul, '*'),
            # Protected Division
            (lambda a, b: np.divide(a, np.where(np.abs(b) < 1e-6, 1.0, b)), '/') 
        ]
        
        self.best_program = None
        self.best_mse = float('inf')

    def _random_node(self, depth):
        # Terminal node (Feature or Constant)
        if depth >= self.max_depth or (depth > 0 and self.rng.random() < 0.3):
            if self.rng.random() < 0.8:
                return Feature(self.rng.randint(0, self.input_dim - 1))
            else:
                return Constant(self.rng.uniform(*self.const_range))
        
        # Operator node
        op, symbol = self.rng.choice(self.ops)
        left = self._random_node(depth + 1)
        right = self._random_node(depth + 1)
        return BinaryOp(left, right, op, symbol)

    def _get_nodes_list(self, node):
        """Helper to flatten tree for mutation/crossover point selection"""
        nodes = [node]
        if isinstance(node, BinaryOp):
            nodes.extend(self._get_nodes_list(node.left))
            nodes.extend(self._get_nodes_list(node.right))
        return nodes

    def _fitness(self, program, X, y):
        try:
            pred = program.evaluate(X)
            # MSE
            mse = np.mean((pred - y)**2)
            if np.isnan(mse) or np.isinf(mse): return 1e9
            return mse
        except:
            return 1e9

    def _tournament(self, population, fitnesses):
        indices = self.rng.sample(range(len(population)), self.tournament_size)
        best_idx = min(indices, key=lambda i: fitnesses[i])
        return population[best_idx]

    def _crossover(self, parent1, parent2):
        # Swap a subtree
        offspring = copy.deepcopy(parent1)
        nodes1 = self._get_nodes_list(offspring)
        nodes2 = self._get_nodes_list(parent2)
        
        target1 = self.rng.choice(nodes1)
        source2 = copy.deepcopy(self.rng.choice(nodes2))
        
        # Primitive logic to replace attributes (Python makes this easy)
        if isinstance(target1, BinaryOp):
            target1.left = source2.left if isinstance(source2, BinaryOp) else source2
            pass 
        
        return offspring

    def _mutate(self, parent):
        offspring = copy.deepcopy(parent)
        nodes = self._get_nodes_list(offspring)
        target = self.rng.choice(nodes)
        
        # Replace the chosen node with a random new subtree
        new_subtree = self._random_node(depth=0) # Reset depth counter for local growth
        
        # Python hack: change class and dict to morph the object in-place
        target.__class__ = new_subtree.__class__
        target.__dict__ = new_subtree.__dict__
        
        return offspring

    def fit(self, X, y, callback = None):
        # Initialize
        population = [self._random_node(0) for _ in range(self.pop_size)]
        
        for gen in range(self.generations):
            fitnesses = [self._fitness(p, X, y) for p in population]
            
            # Track best
            min_fit = min(fitnesses)
            if min_fit < self.best_mse:
                self.best_mse = min_fit
                self.best_program = copy.deepcopy(population[fitnesses.index(min_fit)])

            if callback:

                if callback(generation=gen, model=self, loss=self.best_mse):
                    print(f"[SR] Stopped by callback at gen {gen}")
                    break

            # Evolve
            new_pop = []
            # Elitism: keep best
            new_pop.append(self.best_program)
            
            while len(new_pop) < self.pop_size:
                parent = self._tournament(population, fitnesses)
                
                if self.rng.random() < self.p_mutation:
                    child = self._mutate(parent)
                else:
                    # Cloning (or Crossover if implemented fully)
                    child = copy.deepcopy(parent)
                
                new_pop.append(child)
            
            population = new_pop

    def predict(self, X):
        if self.best_program:
            return self.best_program.evaluate(X)
        return np.zeros(X.shape[0])

# ==============================================================================
# 3. The Searcher Wrapper (SRSearcher)
# ==============================================================================

class SRSearcher(BaseStructureSearcher):
    """
    Wrapper for our custom SimpleSymbolicRegressor.
    No gplearn dependency. Pure Python.
    """

    def __init__(self, input_dim: int, population_size: int = 1000, generations: int = 10):
        super().__init__(input_dim)
        self.pop_size = population_size
        self.generations = generations
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, callback = None) -> None:
        y_flat = y.ravel() if y.ndim > 1 else y
        
        # Use dynamic seed from numpy global state (controlled by experiment loop)
        current_seed = np.random.randint(0, 100000)
        
        self.model = SimpleSymbolicRegressor(
            input_dim=self.input_dim,
            pop_size=self.pop_size,
            generations=self.generations,
            seed=current_seed
        )
        self.model.fit(X, y_flat, callback=callback)

    def get_structure_info(self) -> dict:
        if not self.model or not self.model.best_program:
            return {'type': 'explicit_terms', 'terms': []}
        
        raw_str = str(self.model.best_program)
        print(f"[Custom SR] Found: {raw_str}")
        
        # Simple parsing to identify used indices
        import re
        indices = [int(x) for x in re.findall(r'X(\d+)', raw_str)]
        indices = sorted(list(set(indices)))
        
        term_type = 'linear'
        if '*' in raw_str: term_type = 'interact'
        
        return {
            'type': 'explicit_terms',
            'raw_formula': raw_str,
            'terms': [{
                'type': 'sr_tree',
                'indices': indices,
                'raw': raw_str
            }]
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Executes prediction using the custom model.
        """
        if self.model:
            return self.model.predict(X)
        return np.zeros(X.shape[0])