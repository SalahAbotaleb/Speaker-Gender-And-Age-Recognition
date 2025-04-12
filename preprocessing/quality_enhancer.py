from .preprocessing import Preprocessor
import librosa

class QualityEnhancer(Preprocessor):

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return librosa.effects.preemphasis(X, coef=0.97)