from .preprocessing import Preprocessor
import librosa

import sys
sys.path.append('..')
from audio import Audio

class QualityEnhancer(Preprocessor):

    def __init__(self, coef=0.95):
        self.coef = coef
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [Audio(librosa.effects.preemphasis(x.data, coef=self.coef), x.sampling_rate) for x in X]