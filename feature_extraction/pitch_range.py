from .feature_extraction import FeatureExtractor
import numpy as np
import librosa
from tqdm import tqdm

class PitchRange(FeatureExtractor):
    """
    Class to extract pitch range features from audio data.
    """

    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.sr = config.get("sr", 22050)

    def _extract_pitch_range(self, audio, sr):
        # Extract pitch (fundamental frequency) using librosa
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        
        # Calculate pitch range (max pitch - min pitch) ignoring zeros
        non_zero_pitches = pitches[pitches > 0]
        if len(non_zero_pitches) > 0:
            pitch_range = np.max(non_zero_pitches) - np.min(non_zero_pitches)
        else:
            pitch_range = 0.0
        
        return pitch_range

    def fit(self, X, y=None):
        return self

    def transform(self, audios: list) -> np.ndarray:
        """
        Extract pitch range features from audio data.

        Parameters:
            audios (list): List of audio data.

        Returns:
            np.ndarray: Pitch range features.
        """
        return np.array([self._extract_pitch_range(audio, self.sr) for audio in audios]).reshape(-1, 1)