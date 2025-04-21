from .feature_extraction import FeatureExtractor
import numpy as np
import librosa
from tqdm import tqdm

class HFCC(FeatureExtractor):
    """
    Class to extract HFCC features from audio data.
    """

    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.n_hfcc = config.get("n_hfcc", 13)
        self.sr = config.get("sr", 48000)

    def _extract_hfcc(self, audio, sr):
        # Extract HFCCs with custom parameters
        stft = np.abs(librosa.stft(
            y=audio, n_fft=2048, hop_length=512, window='hann'
        ))
        mel_filter = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=75)
        hfccs = np.log(np.dot(mel_filter, stft) + 1e-10)
        
        # Take mean across time to get 40 features
        #return np.mean(hfccs, axis=1)
        return np.concatenate([np.mean(hfccs, axis=1), np.var(hfccs, axis=1)])

    def fit(self, X, y=None):
        return self

    def transform(self, audios: list) -> np.ndarray:
        """
        Extract HFCC features from audio data.

        Parameters:
            audio (np.ndarray): Audio data.

        Returns:
            np.ndarray: HFCC features.
        """
        return np.array([self._extract_hfcc(audio, self.sr) for audio in audios])