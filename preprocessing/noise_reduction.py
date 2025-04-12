from .preprocessing import Preprocessor
import noisereduce as nr
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('..')
from audio import Audio

class NoiseReducer(Preprocessor):

    def __init__(self):
        super().__init__()

    def transform(self, X):
        return [Audio(np.array(nr.reduce_noise(x.data, sr=x.sampling_rate, time_mask_smooth_ms=150)), x.sampling_rate) for x in tqdm(X, desc="Noise Reduction")]