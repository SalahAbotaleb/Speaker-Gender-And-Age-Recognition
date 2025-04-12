from .preprocessing import Preprocessor
import librosa
import numpy as np

class SilenceRemover(Preprocessor):

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        clean = [np.array(librosa.effects.trim(x, top_db=100)[0], dtype=np.float64) for x in X]
        return clean