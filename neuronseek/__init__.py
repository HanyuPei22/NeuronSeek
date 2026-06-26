"""NeuronSeek: task-driven neuron structure search and tensor decomposition tools."""

__version__ = "2.0.0"

from .poly_regressor import (
    PolyTensorRegression,
    PolyTensorRegressor,
    PolynomialTensorRegression,
)
from .searchers import (
    DiagnosticNeuronSeekSearcher,
    NeuronSeekSearcher,
    NeuronSeekSearcherExpand,
    TNSRSearcher,
    VecSymRegressor,
)

__all__ = [
    "DiagnosticNeuronSeekSearcher",
    "NeuronSeekSearcher",
    "NeuronSeekSearcherExpand",
    "PolyTensorRegression",
    "PolyTensorRegressor",
    "PolynomialTensorRegression",
    "TNSRSearcher",
    "VecSymRegressor",
]
