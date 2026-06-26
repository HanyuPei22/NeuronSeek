from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np

class BaseStructureSearcher(ABC):
    """Abstract base class for Stage 1: Formula/Structure Discovery."""

    def __init__(self, input_dim: int, **kwargs):
        self.input_dim = input_dim

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Executes the search process."""
        pass

    @abstractmethod
    def get_structure_info(self) -> Any:
        """Returns the discovered structure (formula string, masks, or config)."""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__