from .feature_extraction import FeatureExtractor
import numpy as np
import librosa
from tqdm import tqdm

class FundamentalFrequency(FeatureExtractor):
    """
    Class to extract Fundamental Frequency (F0) features from audio data.
    """

    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.sr = config.get("sr", 22050)

    def _extract_f0(self, audio, sr):
        # Extract fundamental frequency (F0) using autocorrelation method
        hop_length = 512
        autocorr = librosa.autocorrelate(audio)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=len(autocorr))
        
        # Find the peak in the autocorrelation corresponding to the fundamental frequency
        peak = np.argmax(autocorr[1:]) + 1
        f0 = freqs[peak] if peak < len(freqs) else 0
        
        return f0

    def fit(self, X, y=None):
        return self

    def transform(self, audios: list) -> np.ndarray:
        """
        Extract Fundamental Frequency (F0) features from audio data.

        Parameters:
            audios (list): List of audio data.

        Returns:
            np.ndarray: Fundamental Frequency features.
        """
        return np.array([self._extract_f0(audio, self.sr) for audio in audios]).reshape(-1, 1)  
