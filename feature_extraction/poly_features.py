from .feature_extraction import FeatureExtractor
import numpy as np
import librosa
from tqdm import tqdm

class PolyFeatures(FeatureExtractor):
    """
    Class to extract polynomial features from audio data.
    """

    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.sr = config.get("sr", 48000)
        self.order = config.get("order", 2)

    def _extract_poly_features(self, audio, sr, order):
        # Extract Mel-frequency cepstral coefficients (MFCCs) as a base
        S = np.abs(librosa.stft(audio))


        p0 = librosa.feature.poly_features(S=S, order=0, sr=sr)
        p1 = librosa.feature.poly_features(S=S, order=1, sr=sr)
        p2 = librosa.feature.poly_features(S=S, order=2, sr=sr)


        p0_mean = np.mean(p0, axis=1)
        p1_mean = np.mean(p1, axis=1)
        p2_mean = np.mean(p2, axis=1)

        return np.concatenate([p0_mean, p1_mean, p2_mean], axis=0)
    
    def fit(self, X, y=None):
        return self

    def transform(self, audios: list) -> np.ndarray:
        """
        Extract polynomial features from audio data.

        Parameters:
            audios (list): List of audio data.

        Returns:
            np.ndarray: Polynomial features.
        """
        return np.array([self._extract_poly_features(audio, self.sr, self.order) for audio in audios])