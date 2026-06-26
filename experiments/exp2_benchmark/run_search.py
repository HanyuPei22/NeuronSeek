"""
Stage 1: Structure Search

Runs NeuronSeek and TN-SR on each dataset to extract structure information.
Saves results to JSON files for Stage 2.

Usage:
    python -m experiments.exp2_benchmark.run_search --datasets california_housing
    python -m experiments.exp2_benchmark.run_search --all
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.exp2_benchmark.config import (
    ALL_DATASETS, SEARCHER_CONFIG, PATHS, EXPERIMENT_CONFIG
)
from neuronseek.utils.tabular_data_loader import load_dataset
from neuronseek.searchers.neuronseek_searcher import NeuronSeekSearcher
from neuronseek.searchers.tnsr_searcher import TNSRSearcher


def run_search(dataset_name: str, seed: int = 42) -> dict:
    """Run both searchers on a dataset and return structure info."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | Seed: {seed}")
    print('='*60)

    # Load data
    data = load_dataset(
        dataset_name,
        test_size=EXPERIMENT_CONFIG['test_size'],
        val_size=EXPERIMENT_CONFIG['val_size'],
        random_state=seed,
    )

    X_train = data['X_train']
    y_train = data['y_train'].ravel()
    n_features = data['n_features']
    n_classes = data['n_classes']

    results = {
        'dataset': dataset_name,
        'n_features': n_features,
        'n_classes': n_classes,
        'task': data['task'],
        'seed': seed,
    }

    # --- NeuronSeek ---
    print(f"\n[1/2] Running NeuronSeek...")
    try:
        ns_config = SEARCHER_CONFIG['NeuronSeek']
        ns_searcher = NeuronSeekSearcher(
            input_dim=n_features,
            num_classes=n_classes,
            rank=ns_config['rank'],
            epochs=ns_config['epochs'],
            batch_size=ns_config['batch_size'],
            reg_lambda=ns_config['reg_lambda'],
        )
        ns_searcher.fit(X_train, y_train)
        ns_structure = ns_searcher.get_structure_info()
        results['NeuronSeek'] = {
            'pure_indices': ns_structure.get('pure_indices', []),
            'interact_indices': ns_structure.get('interact_indices', []),
            'rank': ns_structure.get('rank', 8),
        }
        print(f"    Structure: Pure={results['NeuronSeek']['pure_indices']}, "
              f"Int={results['NeuronSeek']['interact_indices']}")
    except Exception as e:
        print(f"    [Error] NeuronSeek failed: {e}")
        results['NeuronSeek'] = {'pure_indices': [1, 2], 'interact_indices': [], 'error': str(e)}

    # --- TN-SR ---
    print(f"\n[2/2] Running TN-SR...")
    try:
        tnsr_config = SEARCHER_CONFIG['TNSR']
        tnsr_searcher = TNSRSearcher(
            input_dim=n_features,
            population_size=tnsr_config['population_size'],
            generations=tnsr_config['generations'],
        )
        tnsr_searcher.fit(X_train, y_train)
        tnsr_structure = tnsr_searcher.get_structure_info()
        results['TNSR'] = {
            'pure_indices': tnsr_structure.get('pure_indices', []),
            'interact_indices': tnsr_structure.get('interact_indices', []),
            'rank': tnsr_structure.get('rank', 1),
        }
        print(f"    Structure: Pure={results['TNSR']['pure_indices']}, "
              f"Int={results['TNSR']['interact_indices']}")
    except Exception as e:
        print(f"    [Error] TN-SR failed: {e}")
        results['TNSR'] = {'pure_indices': [1], 'interact_indices': [], 'error': str(e)}

    return results


def save_structure(results: dict, output_dir: str):
    """Save structure info to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{results['dataset']}_structure.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[Saved] {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Run structure search on datasets')
    parser.add_argument('--datasets', nargs='+', default=['california_housing'],
                        help='Datasets to process')
    parser.add_argument('--all', action='store_true', help='Run on all datasets')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default=PATHS['structure_dir'],
                        help='Output directory for JSON files')
    args = parser.parse_args()

    datasets = ALL_DATASETS if args.all else args.datasets

    print(f"Structure Search | Datasets: {len(datasets)} | Seed: {args.seed}")
    print(f"Output: {args.output_dir}")

    for ds_name in datasets:
        try:
            results = run_search(ds_name, seed=args.seed)
            save_structure(results, args.output_dir)
        except Exception as e:
            print(f"\n[FATAL] Dataset {ds_name} failed: {e}")
            continue

    print(f"\n{'='*60}")
    print("Search complete!")
    print('='*60)


if __name__ == '__main__':
    main()
