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
        return [Audio(np.array(nr.reduce_noise(x.data, sr=x.sampling_rate,  prop_decrease=0.95, n_std_thresh_stationary=1.3)), x.sampling_rate) for x in X]