from .preprocessing import Preprocessor
import noisereduce as nr
import numpy as np
from tqdm import tqdm

class NoiseReducer(Preprocessor):

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [np.array(nr.reduce_noise(x.data, sr=x.sr, time_mask_smooth_ms=150)) for x in tqdm(X, desc="Noise Reduction")]