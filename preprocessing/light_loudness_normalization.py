from .preprocessing import Preprocessor
import numpy as np

import sys
sys.path.append('..')
from audio import Audio

class LightLoudnessNormalizer(Preprocessor):

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self
    
    def _rms_normalize(self, signal, target_rms=0.1):
        rms = np.sqrt(np.mean(np.square(np.asarray(signal))))
        return signal * (target_rms / (rms + 1e-8))

    def transform(self, X):
        return [Audio(self._rms_normalize(x.data), x.sampling_rate) for x in X]