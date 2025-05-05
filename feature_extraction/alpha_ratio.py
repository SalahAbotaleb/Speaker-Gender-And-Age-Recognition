from .feature_extraction import FeatureExtractor
import numpy as np
import librosa
from tqdm import tqdm

class AlphaRatio(FeatureExtractor):
    """
    Class to extract Alpha Ratio features from audio data.
    """

    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.sr = config.get("sr", 48000)
        self.low_freq = config.get("low_freq", 50)
        self.high_freq = config.get("high_freq", 1000)

    def _extract_alpha_ratio(self, audio, sr):
        # Compute Short-Time Fourier Transform (STFT)
        stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512, window='hann'))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        # Compute energy in low and high frequency bands
        low_band_energy = np.sum(stft[(freqs >= self.low_freq) & (freqs < self.high_freq)], axis=0)
        high_band_energy = np.sum(stft[freqs >= self.high_freq], axis=0)

        # Compute Alpha Ratio
        alpha_ratio = np.sum(low_band_energy) / (np.sum(high_band_energy) + 1e-6)  # Avoid division by zero
        return alpha_ratio

    def fit(self, X, y=None):
        return self

    def transform(self, audios: list) -> np.ndarray:
        """
        Extract Alpha Ratio features from audio data.

        Parameters:
            audios (list): List of audio data.

        Returns:
            np.ndarray: Alpha Ratio features.
        """
        return np.array([self._extract_alpha_ratio(audio, self.sr) for audio in audios]).reshape(-1, 1)