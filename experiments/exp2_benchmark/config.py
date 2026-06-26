"""
Experiment Configuration for Exp2: Tabular Data Benchmark.
"""

# =============================================================================
# Dataset Configuration (Table III)
# =============================================================================

REGRESSION_DATASETS = [
    'california_housing',
    'house_sales',
    'airfoil',
    'diamonds',
    'abalone',
    'bike_sharing',
    'space_ga',
    'airlines_delay',
]

CLASSIFICATION_DATASETS = [
    'credit',
    'heloc',
    'electricity',
    'phoneme',
    'magic_telescope',
    'vehicle',
    'orange_vs_grapefruit',
    'eye_movements',
]

ALL_DATASETS = REGRESSION_DATASETS + CLASSIFICATION_DATASETS

# =============================================================================
# Searcher Configuration
# =============================================================================

SEARCHER_CONFIG = {
    'NeuronSeek': {
        'epochs': 200,
        'batch_size': 64,
        'rank': 8,
        'reg_lambda': 0.05,
    },
    'TNSR': {
        'population_size': 2000,
        'generations': 20,
    },
}

# =============================================================================
# Network Configuration
# =============================================================================

NETWORK_CONFIG = {
    # Hidden dims matching TNSR paper Table (e.g., 6-3-1 for california_housing)
    # For fair comparison, use small networks
    'hidden_dims': [6, 3],  # Matches TNSR paper structure
    'dropout': 0.0,

    # KAN specific
    'kan_grid_size': 5,
    'kan_spline_order': 3,
}

# =============================================================================
# Training Configuration
# =============================================================================

TRAIN_CONFIG = {
    'epochs': 100,
    'batch_size': 64,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'early_stop_patience': 20,
    'scheduler': 'cosine',  # 'cosine' or 'step'
}

# =============================================================================
# Experiment Settings
# =============================================================================

EXPERIMENT_CONFIG = {
    'n_trials': 3,
    'seed_base': 42,
    'test_size': 0.2,
    'val_size': 0.1,
}

# =============================================================================
# Paths
# =============================================================================

PATHS = {
    'structure_dir': 'result/benchmark_result/structures',
    'result_csv': 'result/benchmark_result/comparison_results.csv',
    'log_dir': 'result/benchmark_result/logs',
}
