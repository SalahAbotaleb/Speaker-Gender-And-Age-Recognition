from .preprocessing import Preprocessor
import numpy as np

class VolumeNormalizer(Preprocessor):

    def __init__(self, target_rms):
        self.target_rms = target_rms
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        current_rms = np.sqrt(np.mean(audio ** 2))
        if current_rms > 0:
            audio = audio * (self.target_rms / current_rms)