from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class Model(ClassifierMixin, BaseEstimator, ABC):
    """
    FeatureExtractor class to handle feature extraction from audio data.
    """

    def __init__(self, config: dict = {}):
        self.config = config if config is not None else {}

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass