# NeuronSeek-TD 2.0

NeuronSeek is a task-driven neuron discovery framework for learning compact
polynomial neuron structures from data. This release updates the public codebase
to the second-version NeuronSeek-TD implementation used in the revised paper.

## Highlights

- Differentiable dual-stream structure search for pure polynomial terms and
  CP-decomposed interaction terms.
- TN-SR baseline with a vectorized symbolic-regression engine.
- Structural probes and benchmark utilities for synthetic and tabular
  experiments.
- Legacy polynomial tensor regressor kept for compatibility.

## Installation

```bash
git clone git@github.com:HanyuPei22/NeuronSeek.git
cd NeuronSeek
pip install -e .
```

Install optional experiment dependencies when running vision or benchmark
scripts:

```bash
pip install -e ".[vision,experiments]"
```

## Quick Start

```python
import numpy as np
from neuronseek import NeuronSeekSearcher, TNSRSearcher

rng = np.random.default_rng(42)
X = rng.normal(size=(256, 5)).astype("float32")
y = 2.5 * X[:, 0] + 3.0 * X[:, 1] ** 2 - 1.2 * X[:, 2] ** 3

searcher = NeuronSeekSearcher(input_dim=X.shape[1], rank=4, epochs=20)
searcher.fit(X, y)
structure = searcher.get_structure_info()
print(structure)

tnsr = TNSRSearcher(input_dim=X.shape[1], population_size=200, generations=5)
tnsr.fit(X, y)
print(tnsr.get_structure_info())
```

Legacy polynomial tensor regression is still available:

```python
from neuronseek import PolyTensorRegressor

neuron = PolyTensorRegressor(rank=3, poly_order=3, num_epochs=20)
neuron.fit(X, y)
print(neuron.neuron)
```

## Repository Map

```text
neuronseek/
  core/          dual-stream tensor interaction layers and L0 gates
  searchers/     NeuronSeek, TN-SR, SR, EQL, and MetaSymNet search wrappers
  models/        benchmark networks, sparse search agent, ResNet-TN blocks
  layers/        task-based neuron and KAN layers
  utils/         synthetic/tabular data utilities and seed helpers
  deprecated/    first-version STRidge/proxy-model code kept for scripts
scripts/         demos and diagnostic scripts
experiments/     scalability, benchmark, and structural evaluation code
benchmark/       first-version image benchmark scripts
```

## Main APIs

- `NeuronSeekSearcher`: differentiable structure search over pure and
  interaction polynomial orders.
- `TNSRSearcher`: wrapper around the updated `VecSymRegressor` TN-SR engine.
- `VecSymRegressor`: vectorized symbolic regression baseline.
- `PolyTensorRegressor`: compatibility polynomial tensor regressor.

## tnlearn Integration

The lightweight tnlearn integration lives in
[`NewT123-WM/tnlearn`](https://github.com/NewT123-WM/tnlearn). The public class
name is `PolyTensorRegressor`, with `PolyTensorRegression` retained as a
backward-compatible alias.

## Citation

```bibtex
@article{Pei2025,
  author = {Pei, Hanyu and Liao, Jing-Xiao and Zhao, Qibin and Gao, Ting and Zhang, Shijun and Zhang, Xiaoge and Fan, Feng-Lei},
  title = {NeuronSeek: On Stability and Expressivity of Task-Driven Neurons},
  journal = {Preprints},
  year = {2025},
  doi = {10.20944/preprints202506.1586.1},
  url = {https://doi.org/10.20944/preprints202506.1586.1}
}
```
