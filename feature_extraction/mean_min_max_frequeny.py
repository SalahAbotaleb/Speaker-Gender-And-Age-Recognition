from .feature_extraction import FeatureExtractor
import numpy as np
import librosa
from tqdm import tqdm

class MeanMinMaxFrequency(FeatureExtractor):
    """
    Class to extract Mean, Min, and Max Frequency features from audio data.
    """

    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.sr = config.get("sr", 22050)

    def _extract_frequencies(self, audio, sr):
        # Calculate the Short-Time Fourier Transform (STFT)
        stft = np.abs(librosa.stft(audio))
        # Compute the frequencies for each bin
        freqs = librosa.fft_frequencies(sr=sr)
        # Compute the mean frequency weighted by the magnitude of the STFT
        mean_freq = np.sum(freqs * np.sum(stft, axis=1)) / np.sum(stft)
        # Compute the min and max frequencies weighted by the magnitude of the STFT
        # Compute the duration of each frequency bin in seconds
        hop_length = self.config.get("hop_length", 512)
        duration_per_bin = hop_length / sr
        # Compute the total duration each frequency appears
        freq_durations = np.sum(stft > 0, axis=1) * duration_per_bin
        # Apply an amplitude threshold
        amplitude_threshold = self.config.get("amplitude_threshold", 10)
        valid_amplitudes = np.max(stft, axis=1) > amplitude_threshold
        # Filter frequencies that appear for more than 1 second and meet the amplitude threshold
        valid_freqs = freqs[(freq_durations > 0) & valid_amplitudes]
        if len(valid_freqs) > 0:
            min_freq = np.min(valid_freqs)
            max_freq = np.max(valid_freqs)
        else:
            min_freq = 0
            max_freq = 0
        return mean_freq, min_freq, max_freq

    def fit(self, X, y=None):
        return self

    def transform(self, audios: list) -> np.ndarray:
        """
        Extract Mean, Min, and Max Frequency features from audio data.

        Parameters:
            audios (list): List of audio data.

        Returns:
            np.ndarray: Mean, Min, and Max Frequency features.
        """
        return np.array([self._extract_frequencies(audio, self.sr) for audio in audios])
