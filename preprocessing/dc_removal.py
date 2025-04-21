from .preprocessing import Preprocessor
import numpy as np

import sys
sys.path.append('..')
from audio import Audio

class DCRemover(Preprocessor):

    def __init__(self):
        super().__init__()

    def transform(self, X):
        return [Audio(x.data - np.mean(x.data), x.sampling_rate) for x in X]