from preprocessing import Preprocessor
import numpy as np
import pyloudnorm as pyln

import sys
sys.path.append('..')
from audio import Audio

class VolumeNormalizer(Preprocessor):

    def __init__(self, target_loudness=-25, max_gain_db=10):
        self.target_loudness = target_loudness
        self.max_gain_db = max_gain_db
        super().__init__()

    def normalize_volume(self, audio: Audio) -> Audio:
        meter = pyln.Meter(audio.sampling_rate)
        loudness = meter.integrated_loudness(audio.data)

        gain_db = self.target_loudness - loudness

        if gain_db > self.max_gain_db:
            adjusted_target = loudness + self.max_gain_db
        else:
            adjusted_target = self.target_loudness

        normalized_audio = pyln.normalize.loudness(audio.data, loudness, adjusted_target)

        peak = np.max(np.abs(normalized_audio))
        if peak > 1.0:
            normalized_audio = normalized_audio / (peak * 1.1)
        
        return Audio(normalized_audio, audio.sampling_rate)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.normalize_volume(x) for x in X]