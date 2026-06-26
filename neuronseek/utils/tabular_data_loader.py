"""
Tabular Data Loader for Benchmark Experiments.

Supports 16 datasets from Table III with mixed loading strategy:
1. sklearn (built-in, most stable)
2. OpenML API (auto-cached to ~/.openml/)
3. Local CSV fallback (data/ directory)

Usage:
    from neuronseek.utils.tabular_data_loader import load_dataset, DATASET_REGISTRY

    # Load a single dataset
    data = load_dataset('california_housing')
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']

    # Get metadata
    print(data['task'])      # 'regression' or 'classification'
    print(data['n_classes']) # 1 for regression, >1 for classification
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ==============================================================================
# Dataset Registry - Table III datasets
# ==============================================================================

DATASET_REGISTRY = {
    # --- Regression (MSE) ---
    'california_housing': {
        'source': 'sklearn',
        'task': 'regression',
        'n_classes': 1,
        'openml_id': None,
    },
    'house_sales': {
        'source': 'openml',
        'task': 'regression',
        'n_classes': 1,
        'openml_id': 42731,  # House Sales in King County
    },
    'airfoil': {
        'source': 'openml',
        'task': 'regression',
        'n_classes': 1,
        'openml_id': 1503,  # Airfoil Self-Noise
    },
    'diamonds': {
        'source': 'openml',
        'task': 'regression',
        'n_classes': 1,
        'openml_id': 42225,
    },
    'abalone': {
        'source': 'openml',
        'task': 'regression',
        'n_classes': 1,
        'openml_id': 183,
    },
    'bike_sharing': {
        'source': 'openml',
        'task': 'regression',
        'n_classes': 1,
        'openml_id': 42712,  # Bike Sharing Demand
    },
    'space_ga': {
        'source': 'openml',
        'task': 'regression',
        'n_classes': 1,
        'openml_id': 507,
    },
    'airlines_delay': {
        'source': 'openml',
        'task': 'regression',
        'n_classes': 1,
        'openml_id': 42721,
    },

    # --- Classification (ACC) ---
    'credit': {
        'source': 'openml',
        'task': 'classification',
        'n_classes': 2,
        'openml_id': 31,  # credit-g
    },
    'heloc': {
        'source': 'openml',
        'task': 'classification',
        'n_classes': 2,
        'openml_id': 45026,
    },
    'electricity': {
        'source': 'openml',
        'task': 'classification',
        'n_classes': 2,
        'openml_id': 151,
    },
    'phoneme': {
        'source': 'openml',
        'task': 'classification',
        'n_classes': 2,
        'openml_id': 1489,
    },
    'magic_telescope': {
        'source': 'openml',
        'task': 'classification',
        'n_classes': 2,
        'openml_id': 1120,
    },
    'vehicle': {
        'source': 'openml',
        'task': 'classification',
        'n_classes': 4,
        'openml_id': 54,
    },
    'orange_vs_grapefruit': {
        'source': 'openml',
        'task': 'classification',
        'n_classes': 2,
        'openml_id': 1493,  # May need adjustment
    },
    'eye_movements': {
        'source': 'openml',
        'task': 'classification',
        'n_classes': 2,
        'openml_id': 1044,
    },
}

# ==============================================================================
# Loading Functions
# ==============================================================================

def _load_from_sklearn(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load datasets from sklearn."""
    if name == 'california_housing':
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        return data.data, data.target
    else:
        raise ValueError(f"Unknown sklearn dataset: {name}")


def _load_from_openml(openml_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset from OpenML with caching."""
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(data_id=openml_id, as_frame=True, parser='auto')

        X = data.data
        y = data.target

        # Handle categorical features - convert to numeric
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        # Handle target encoding for classification
        if y.dtype == 'object' or y.dtype.name == 'category':
            y = LabelEncoder().fit_transform(y.astype(str))

        return X.values.astype(np.float32), np.array(y).astype(np.float32)

    except Exception as e:
        raise RuntimeError(f"Failed to load OpenML dataset {openml_id}: {e}")


def _load_from_csv(name: str, data_dir: str = 'data') -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset from local CSV file."""
    csv_path = Path(data_dir) / f"{name}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset '{name}' not found at {csv_path}. "
            f"Please download it manually or check OpenML connection."
        )

    df = pd.read_csv(csv_path)

    # Assume last column is target
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.float32)

    return X, y


# ==============================================================================
# Main API
# ==============================================================================

def load_dataset(
    name: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    normalize: bool = True,
    data_dir: str = 'data',
    return_torch: bool = False,
) -> Dict[str, Any]:
    """
    Load a tabular dataset with preprocessing.

    Args:
        name: Dataset name (see DATASET_REGISTRY)
        test_size: Fraction for test set (default: 0.2)
        val_size: Fraction for validation set from training (default: 0.1)
        random_state: Random seed for reproducibility
        normalize: Whether to standardize features (default: True)
        data_dir: Directory for local CSV fallback
        return_torch: If True, return torch.Tensor instead of numpy arrays

    Returns:
        Dictionary with keys:
            - X_train, X_val, X_test: Feature arrays
            - y_train, y_val, y_test: Target arrays
            - task: 'regression' or 'classification'
            - n_classes: Number of classes (1 for regression)
            - n_features: Number of input features
            - feature_names: List of feature names (if available)
            - scaler: Fitted StandardScaler (if normalize=True)
    """
    if name not in DATASET_REGISTRY:
        available = list(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")

    info = DATASET_REGISTRY[name]

    # --- 1. Load raw data ---
    X, y = None, None

    # Strategy 1: sklearn
    if info['source'] == 'sklearn':
        try:
            X, y = _load_from_sklearn(name)
            print(f"[DataLoader] Loaded '{name}' from sklearn")
        except Exception as e:
            print(f"[DataLoader] sklearn failed: {e}")

    # Strategy 2: OpenML
    if X is None and info['openml_id'] is not None:
        try:
            X, y = _load_from_openml(info['openml_id'])
            print(f"[DataLoader] Loaded '{name}' from OpenML (id={info['openml_id']})")
        except Exception as e:
            print(f"[DataLoader] OpenML failed: {e}")

    # Strategy 3: Local CSV fallback
    if X is None:
        try:
            X, y = _load_from_csv(name, data_dir)
            print(f"[DataLoader] Loaded '{name}' from local CSV")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset '{name}' from all sources. "
                f"Last error: {e}"
            )

    # --- 2. Handle NaN values ---
    if np.isnan(X).any():
        print(f"[DataLoader] Warning: Found NaN in features, filling with column mean")
        col_means = np.nanmean(X, axis=0)
        nan_indices = np.where(np.isnan(X))
        X[nan_indices] = np.take(col_means, nan_indices[1])

    if np.isnan(y).any():
        print(f"[DataLoader] Warning: Found NaN in target, removing those samples")
        valid_mask = ~np.isnan(y)
        X, y = X[valid_mask], y[valid_mask]

    # --- 3. Train/Val/Test Split ---
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)  # Adjust ratio
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=random_state
    )

    # --- 4. Normalize features ---
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    # --- 5. Reshape target for regression ---
    if info['task'] == 'regression':
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
    else:
        # Ensure integer labels for classification
        y_train = y_train.astype(np.int64)
        y_val = y_val.astype(np.int64)
        y_test = y_test.astype(np.int64)

    # --- 6. Convert to torch if requested ---
    if return_torch:
        import torch
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)

        if info['task'] == 'regression':
            y_train = torch.tensor(y_train, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)
        else:
            y_train = torch.tensor(y_train, dtype=torch.long)
            y_val = torch.tensor(y_val, dtype=torch.long)
            y_test = torch.tensor(y_test, dtype=torch.long)

    # --- 7. Build result ---
    result = {
        # Data
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,

        # Metadata
        'task': info['task'],
        'n_classes': info['n_classes'],
        'n_features': X_train.shape[1],
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'scaler': scaler,
        'dataset_name': name,
    }

    print(f"[DataLoader] {name}: {result['n_features']} features, "
          f"train={result['n_train']}, val={result['n_val']}, test={result['n_test']}, "
          f"task={info['task']}")

    return result


def get_dataloaders(
    name: str,
    batch_size: int = 64,
    **kwargs
) -> Tuple[Any, Any, Any, Dict]:
    """
    Convenience function to get PyTorch DataLoaders directly.

    Returns:
        (train_loader, val_loader, test_loader, metadata)
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    data = load_dataset(name, return_torch=True, **kwargs)

    train_dataset = TensorDataset(data['X_train'], data['y_train'])
    val_dataset = TensorDataset(data['X_val'], data['y_val'])
    test_dataset = TensorDataset(data['X_test'], data['y_test'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    metadata = {
        'task': data['task'],
        'n_classes': data['n_classes'],
        'n_features': data['n_features'],
        'scaler': data['scaler'],
        'dataset_name': name,
    }

    return train_loader, val_loader, test_loader, metadata


def list_datasets(task: Optional[str] = None) -> list:
    """List available datasets, optionally filtered by task type."""
    if task is None:
        return list(DATASET_REGISTRY.keys())
    return [k for k, v in DATASET_REGISTRY.items() if v['task'] == task]


# ==============================================================================
# Quick Test
# ==============================================================================

if __name__ == '__main__':
    # Test loading California Housing (sklearn, always works)
    print("=" * 60)
    print("Testing California Housing (sklearn)")
    print("=" * 60)
    data = load_dataset('california_housing')
    print(f"X_train shape: {data['X_train'].shape}")
    print(f"y_train shape: {data['y_train'].shape}")
    print()

    # List all datasets
    print("=" * 60)
    print("Available Datasets")
    print("=" * 60)
    print(f"Regression: {list_datasets('regression')}")
    print(f"Classification: {list_datasets('classification')}")
