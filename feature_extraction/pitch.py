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

    def _extract_f0(self, audio, sr=16000, frame_length=2048, hop_length=512):
      f0, voiced_flag, voiced_probs = librosa.pyin(
          audio, 
          fmin=librosa.note_to_hz('C2'),  # ~65 Hz (male)
          fmax=librosa.note_to_hz('C7'),  # ~2093 Hz (female/child)
          frame_length=frame_length,
          hop_length=hop_length
      )
      
      return f0

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
        return np.array([self._extract_f0(audio) for audio in tqdm(audios, desc="Extracting F0")])