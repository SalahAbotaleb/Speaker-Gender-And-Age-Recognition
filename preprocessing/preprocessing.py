from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Preprocessor(BaseEstimator, TransformerMixin, ABC):
    """
    Preprocessor class to handle missing values and categorical encoding.
    """

    def __init__(self, config: dict = None):
        self.config = config if config is not None else {}

    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def transform(self, audio: np.ndarray) -> np.ndarray:
        pass