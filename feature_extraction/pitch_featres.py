from .feature_extraction import FeatureExtractor
import numpy as np
import librosa
from tqdm import tqdm

class PitchFeatures(FeatureExtractor):
    """
    Class to extract pitch range features from audio data.
    """

    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.sr = config.get("sr", 48000)

    def _extract_pitch_range(self, audio, sr):
        # Extract pitch (fundamental frequency) using librosa
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        
        # Calculate pitch range (max pitch - min pitch) ignoring zeros
        non_zero_pitches = pitches[pitches > 0]
        if len(non_zero_pitches) > 0:
            pitch_mean = np.mean(non_zero_pitches)
            pitch_std = np.std(non_zero_pitches)
            pitch_range_max = np.max(non_zero_pitches) 
            pitch_range_min = np.min(non_zero_pitches)
        else:
            pitch_mean = 0.0
            pitch_std = 0.0
            pitch_range_max = 0.0
            pitch_range_min = 0.0
        
        return [pitch_mean, pitch_std, pitch_range_max, pitch_range_min]

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
        return np.array([self._extract_pitch_range(audio, self.sr) for audio in audios])