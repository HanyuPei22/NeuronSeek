import numpy as np
import operator
from copy import deepcopy
from random import randint, random, shuffle
from sympy import sympify, expand, Symbol
import sympy
from typing import Dict, List, Any

from neuronseek.utils.seeds import setup_seed
from .base import BaseStructureSearcher

# ==============================================================================
# The Original VecSymRegressor Engine (Your Provided Code)
# ==============================================================================

class VecSymRegressor:
    def __init__(self,
                 random_state=100,
                 pop_size=5000,
                 max_generations=20,
                 tournament_size=10,
                 coefficient_range=None,
                 x_pct=0.7,
                 xover_pct=0.3,
                 save=False,
                 operations=None):

        setup_seed(random_state)
        self.random_state = random_state
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size or round(pop_size * 0.03)

        self.x_pct = x_pct
        self.xover_pct = xover_pct
        self.save = save

        self.global_best = float("inf")
        self.best_prog = None
        self.neuron = None

        if coefficient_range is None:
            self.coefficient_range = [-1, 1]
        else:
            self.coefficient_range = coefficient_range

        self.operations = operations or (
            {"func": operator.add, "arg_count": 2, "format_str": "({} + {})"},
            {"func": operator.sub, "arg_count": 2, "format_str": "({} - {})"},
            {"func": operator.mul, "arg_count": 2, "format_str": "({} * {})"},
            {"func": operator.neg, "arg_count": 1, "format_str": "-({})"},
        )

    def render_prog(self, node):
        if "children" not in node:
            return node["feature_name"]
        return node["format_str"].format(*[self.render_prog(c) for c in node["children"]])

    def simp(self, tree):
        return str(expand(sympify(self.render_prog(tree)))).replace("*", "@").replace('@@', '**')

    def evaluate(self, expr, x_data):
        # TN-SR: x represents the entire vector, formula applies element-wise
        # x_data shape: (D, N) where D=dimensions, N=samples
        # Computation flow:
        # 1. Apply formula element-wise to each feature -> (D, N)
        # 2. Sum over all features to get prediction -> (N,)
        # Semantics: y = Σ g(xi) for all features

        x = x_data  # Keep original shape (D, N)

        # Replace @ with * for element-wise multiplication
        # (@ was used as placeholder to distinguish from ** during simplification)
        expr_eval = expr.replace('@@', '**').replace('@', '*')

        try:
            result = eval(expr_eval)  # shape: (D, N)

            # Ensure result is 2D array
            if result.ndim == 1:
                result = result.reshape(-1, 1)

            # Sum over feature dimension to get prediction for each sample
            prediction = np.sum(result, axis=0)  # shape: (N,)

            return expr, prediction
        except Exception:
            # Handle numerical overflow or other errors
            return expr, np.full(x_data.shape[1], np.inf)

    def rand_w(self):
        # Generate floating-point coefficients to handle scale properly
        # For high-dimensional data (D features), predictions scale by ~sqrt(D)
        # So we need coefficients like 0.01, 0.1, 1, etc.
        return str(np.round(np.random.uniform(low=-1.0, high=1.0), 2))

    def random_prog(self, depth=0):
        n_d = depth
        op = self.operations[randint(0, len(self.operations) - 1)]
        if randint(0, 10) >= depth and n_d <= 6:
            n_d += 1
            return {
                "func": op["func"],
                "children": [self.random_prog(depth + 1) for _ in range(op["arg_count"])],
                "format_str": op["format_str"],
            }
        else:
            return {"feature_name": 'x'} if random() < self.x_pct else {"feature_name": self.rand_w()}

    def select_random_node(self, selected, parent, depth):
        if "children" not in selected:
            return parent
        if randint(0, 10) < 2 * depth:
            return selected
        child_count = len(selected["children"])
        return self.select_random_node(
            selected["children"][randint(0, child_count - 1)],
            selected, depth + 1)

    def do_mutate(self, selected):
        offspring = deepcopy(selected)
        mutate_point = self.select_random_node(offspring, None, 0)
        if mutate_point:
            child_count = len(mutate_point.get("children", []))
            if child_count > 0:
                mutate_point["children"][randint(0, child_count - 1)] = self.random_prog(0)
        return offspring

    def do_xover(self, selected1, selected2):
        offspring = deepcopy(selected1)
        xover_point1 = self.select_random_node(offspring, None, 0)
        xover_point2 = self.select_random_node(selected2, None, 0)
        if xover_point1 and xover_point2 and "children" in xover_point1:
            child_count = len(xover_point1["children"])
            if child_count > 0:
                xover_point1["children"][randint(0, child_count - 1)] = deepcopy(xover_point2)
        return offspring

    def get_random_parent(self, popu, fitne):
        tournament_members = [
            randint(0, self.pop_size - 1) for _ in range(self.tournament_size)]
        member_fitness = [(fitne[i], popu[i]) for i in tournament_members]
        return min(member_fitness, key=lambda x: x[0])[1]

    def get_offspring(self, popula, ftns):
        tempt = random()
        parent1 = self.get_random_parent(popula, ftns)
        if tempt < self.xover_pct:
            parent2 = self.get_random_parent(popula, ftns)
            return self.do_xover(parent1, parent2)
        elif self.xover_pct <= tempt < 0.9:
            return self.do_mutate(parent1)
        else:
            return parent1

    def node_count(self, x):
        if "children" not in x:
            return 1
        return sum([self.node_count(c) for c in x["children"]])

    def compute_fitness(self, func, pred, label):
        m = func.count('x')
        # Relaxed condition: reject only if no x at all
        # (Allow formulas with single x like "x**2" or "2*x")
        if m == 0:
            return float("inf")
        else:
            try:
                mse = np.mean(np.square(pred - label))
                return mse
            except:
                return float("inf")

    def fit(self, X, y, callback = None):
        X = X.T
        y = y.T
        self.population = [self.random_prog() for _ in range(self.pop_size)]
        self.box = {}

        for gen in range(self.max_generations):
            fitness = []
            for prog in self.population:
                try:
                    s_prog = self.simp(prog)
                    func_str, prediction = self.evaluate(s_prog, X)
                    score = self.compute_fitness(func_str, prediction, y)
                except Exception:
                    score = float("inf")
                    func_str = ""
                
                fitness.append(score)

                if score < self.global_best:
                    self.global_best = score
                    self.best_prog = func_str

                if len(self.box) < self.pop_size * 0.05:
                    self.box[score] = prog
                else:
                    key_sort = sorted(self.box)
                    if score < key_sort[-1]:
                        self.box.pop(key_sort[-1])
                        self.box[score] = prog

            if callback:
                should_stop = callback(generation=gen, engine=self, loss=self.global_best)
                if should_stop:
                    print(f"[TN-SR] Stopped by callback at gen {gen}")
                    break

            lst = list(self.box.values())
            self.population += lst
            shuffle(self.population)
            population_new = [self.get_offspring(self.population, fitness) for _ in range(self.pop_size)]
            self.population = population_new + lst

        self.neuron = self.best_prog

# ==============================================================================
# WRAPPER (TNSRSearcher)
# ==============================================================================

class TNSRSearcher(BaseStructureSearcher):
    """
    Wrapper for VecSymRegressor (TN-SR).
    Acts as a 'Pure-Stream-Only' Structure Searcher.
    """
    def __init__(self, input_dim: int, population_size=2000, generations=20):
        super().__init__(input_dim)
        # TN-SR Engine (Global Symbolic Regression)
        self.engine = VecSymRegressor(
            pop_size=population_size, 
            max_generations=generations,
            random_state=42
        )
        self.input_dim = input_dim

    def fit(self, X: np.ndarray, y: np.ndarray, callback = None) -> None:
        """
        Fits the genetic algorithm.
        """
        # The engine treats X as a global vector variable (internally transposes it)
        self.engine.fit(X, y, callback=callback)

        print(f"[TN-SR] Search Complete. Best Internal MSE: {self.engine.global_best:.4f}")

    def get_structure_info(self) -> Dict[str, Any]:
        """
        Parses the symbolic formula to extract active polynomial orders.
        Returns a 'neuronseek' type structure with ONLY pure_indices.
        """
        raw_formula = self.engine.neuron
        
        if not raw_formula:
            return {
                'type': 'neuronseek',
                'pure_indices': [],
                'interact_indices': [], # Explicitly empty
                'rank': 1
            }
            
        # 1. Clean format
        clean_formula = raw_formula.replace('@@', '**').replace('@', '*')
        print(f"[TN-SR] Best Formula: {clean_formula}")
        
        # 2. Parse degrees
        active_orders = self._extract_active_degrees(clean_formula)
        print(f"[TN-SR] Mapped to Pure Indices: {active_orders}")

        # 3. Return as NeuronSeek structure for the Evaluator
        return {
            'type': 'neuronseek',       
            'pure_indices': active_orders, 
            'interact_indices': [],     
            'rank': 1                   
        }

    def _extract_active_degrees(self, formula_str: str) -> List[int]:
        """
        Parses a polynomial string of 'x' and returns a list of active degrees.
        """
        try:
            x = Symbol('x')
            expr = expand(sympify(formula_str))
        except Exception as e:
            print(f"[TN-SR Parse Error] {e}")
            return []

        active_degrees = set()
        
        if expr.func == sympy.core.add.Add:
            terms = expr.args
        else:
            terms = [expr]
            
        for term in terms:
            # as_coeff_Mul separates "2.0 * x**2" into (2.0, x**2)
            coeff, var_part = term.as_coeff_Mul()
            
            if var_part == sympy.S.One: continue
                
            degree = sympy.degree(var_part, gen=x)
            if degree > 0:
                active_degrees.add(int(degree))
                
        return sorted(list(active_degrees))
