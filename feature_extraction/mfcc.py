from .feature_extraction import FeatureExtractor
import numpy as np
import librosa
from tqdm import tqdm

class MFCC(FeatureExtractor):
    """
    Class to extract MFCC features from audio data.
    """

    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.n_mfcc = config.get("n_mfcc", 13)
        self.sr = config.get("sr", 22050)

    def _extract_mfcc(self, audio, sr):
        # Extract MFCCs with paper's parameters
        mfccs = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=40, n_fft=2048,
            hop_length=512, window='hann'
        )
        
        # Take mean across time to get 40 features
        return np.mean(mfccs, axis=1)

    def fit(self, X, y=None):
        return self

    def transform(self, audios: list) -> np.ndarray:
        """
        Extract MFCC features from audio data.

        Parameters:
            audio (np.ndarray): Audio data.

        Returns:
            np.ndarray: MFCC features.
        """
        return np.array([self._extract_mfcc(audio, self.sr) for audio in audios])