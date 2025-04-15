from .preprocessing import Preprocessor
import noisereduce as nr
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('..')
from audio import Audio
from scipy.signal import butter, lfilter

class SpeechFilter(Preprocessor):

    def __init__(self, min_amplitude=0.01, low_freq=1, high_freq=3400, noise_reduce=True):
        super().__init__()
        self.min_amplitude = min_amplitude
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.noise_reduce = noise_reduce

    def bandpass_filter(self, data, sampling_rate):
        nyquist = 0.5 * sampling_rate
        low = self.low_freq / nyquist
        high = self.high_freq / nyquist
        b, a = butter(1, [low, high], btype='band')
        return lfilter(b, a, data)

    def reduce_noise(self, data, sampling_rate):
        return nr.reduce_noise(y=data, sr=sampling_rate)


    def transform(self, X):
        processed_audio = []
        for x in X:
            data = x.data
            # Apply bandpass filter
            data = self.bandpass_filter(data, x.sampling_rate)
            # Apply amplitude threshold filtering
            data = np.where(np.abs(data) > self.min_amplitude, data, 0)
            # Optionally reduce noise
            if self.noise_reduce:
                data = self.reduce_noise(data, x.sampling_rate)
            processed_audio.append(Audio(data, x.sampling_rate))
        return processed_audio
