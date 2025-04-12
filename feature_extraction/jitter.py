from .feature_extraction import FeatureExtractor
import numpy as np
import librosa
from tqdm import tqdm

class Jitter(FeatureExtractor):
    """
    Class to extract Jitter features from audio data.
    """

    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.sr = config.get("sr", 22050)
        self.frame_length = config.get("frame_length", 0.025)  # 25ms
        self.hop_length = config.get("hop_length", 0.01)  # 10ms

    def _extract_jitter(self, audio, sr):
        # Calculate frame and hop lengths in samples
        frame_length_samples = int(self.frame_length * sr)
        hop_length_samples = int(self.hop_length * sr)

        # Extract pitch (F0) using librosa's piptrack
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, n_fft=frame_length_samples, hop_length=hop_length_samples)

        # Get the fundamental frequency (F0) for each frame
        f0 = []
        for i in range(pitches.shape[1]):
            index = magnitudes[:, i].argmax()
            f0.append(pitches[index, i] if magnitudes[index, i] > 0 else 0)

        # Calculate jitter as the average absolute difference between consecutive F0 values
        f0 = np.array(f0)
        nonzero_f0 = f0[f0 > 0]  # Exclude zero values
        if len(nonzero_f0) < 2:
            return 0  # Return 0 if not enough F0 values to calculate jitter

        jitter = np.mean(np.abs(np.diff(nonzero_f0)) / nonzero_f0[:-1])
        return jitter

    def fit(self, X, y=None):
        return self

    def transform(self, audios: list) -> np.ndarray:
        """
        Extract Jitter features from audio data.

        Parameters:
            audios (list): List of audio data.

        Returns:
            np.ndarray: Jitter features.
        """
        return np.array([self._extract_jitter(audio, self.sr) for audio in tqdm(audios, desc="Extracting Jitter")]).reshape(-1, 1)