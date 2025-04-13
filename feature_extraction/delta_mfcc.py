from .feature_extraction import FeatureExtractor
import numpy as np
import librosa
from tqdm import tqdm

class DeltaMFCC(FeatureExtractor):
    """
    Class to extract Delta MFCC features from audio data.
    """

    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.n_mfcc = config.get("n_mfcc", 13)
        self.sr = config.get("sr", 22050)

    def _extract_delta_mfcc(self, audio, sr):
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=self.n_mfcc, n_fft=2048,
            hop_length=512, window='hann'
        )
        
        delta_mfccs = librosa.feature.delta(mfccs,width=7, order=1)
        
        # Take mean across time to get features
        return np.mean(delta_mfccs, axis=1)

    def fit(self, X, y=None):
        return self

    def transform(self, audios: list) -> np.ndarray:
        """
        Extract Delta MFCC features from audio data.

        Parameters:
            audios (list): List of audio data.

        Returns:
            np.ndarray: Delta MFCC features.
        """
        return np.array([self._extract_delta_mfcc(audio, self.sr) for audio in audios])