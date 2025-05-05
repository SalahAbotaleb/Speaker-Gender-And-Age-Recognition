from preprocessing import Preprocessor
import numpy as np
from scipy.signal import butter, lfilter

import sys
sys.path.append('..')
from audio import Audio

class HPF(Preprocessor):

    def __init__(self):
        super().__init__()

    def _butter_highpass_filter(self, data, cutoff=60, fs=16000, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        y = lfilter(b, a, data)
        return y

    def transform(self, X):
        return [Audio(self._butter_highpass_filter(x.data), x.sampling_rate) for x in X]