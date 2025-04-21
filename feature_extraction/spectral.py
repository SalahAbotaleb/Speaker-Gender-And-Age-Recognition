from .feature_extraction import FeatureExtractor
import numpy as np
import librosa
from tqdm import tqdm

class SpectralFeatures(FeatureExtractor):
    """
    Class to extract spectral features from audio data.
    """

    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.sr = config.get("sr", 48000)
        self.n_fft = config.get("n_fft", 2048)
        self.hop_length = config.get("hop_length", 512)

    def _extract_spectral_features(self, audio, sr):
        # Extract spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        # Extract spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        # Extract spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        # Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=audio, hop_length=self.hop_length)

        # Extract zero-crossing rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            y=audio, hop_length=self.hop_length
        )

        # Combine features and take mean across time
        features = np.concatenate(
            [
                np.mean(spectral_centroid, axis=1),
                np.mean(spectral_bandwidth, axis=1),
                np.mean(spectral_rolloff, axis=1),
                np.mean(flatness, axis=1),
                np.mean(zero_crossing_rate, axis=1),
            ]
        )
        return features

    def fit(self, X, y=None):
        return self

    def transform(self, audios: list) -> np.ndarray:
        """
        Extract spectral features from audio data.

        Parameters:
            audios (list): List of audio data.

        Returns:
            np.ndarray: Spectral features.
        """
        return np.array([self._extract_spectral_features(audio, self.sr) for audio in audios])