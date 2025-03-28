from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureExtractor(BaseEstimator, TransformerMixin, ABC):
    """
    FeatureExtractor class to handle feature extraction from audio data.
    """

    def __init__(self, config: dict = None):
        self.config = config if config is not None else {}

    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def transform(self, audio: np.ndarray) -> np.ndarray:
        pass