from .feature_extraction import FeatureExtractor
import numpy as np
import librosa
from tqdm import tqdm

class CPPS(FeatureExtractor):
    """
    Class to extract CPPS (Cepstral Peak Prominence Smoothed) features from audio data.
    """

    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.sr = config.get("sr", 22050)

    def _extract_cpp(self, audio, sr):
        # Compute the cepstrum
        spectrum = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512, window='hann'))**2
        log_spectrum = np.log(spectrum + 1e-10)
        cepstrum = np.fft.irfft(log_spectrum, axis=0)

        # Find the peak prominence in the cepstrum
        cpps = np.max(cepstrum[1:50], axis=0)  # Limit to a reasonable range (e.g., 1:50 quefrency bins)
        
        # Take mean across time to get a single feature
        return np.mean(cpps)

    def fit(self, X, y=None):
        return self

    def transform(self, audios: list) -> np.ndarray:
        """
        Extract CPPS features from audio data.

        Parameters:
            audios (list): List of audio data.

        Returns:
            np.ndarray: CPPS features.
        """
   
        return np.array([self._extract_cpp(audio, self.sr) for audio in audios]).reshape(-1, 1)
