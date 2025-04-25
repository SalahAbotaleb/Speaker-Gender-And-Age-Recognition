from .feature_extraction import FeatureExtractor
import numpy as np
import librosa
from tqdm import tqdm
import scipy.stats

class GenderFeatures(FeatureExtractor):
    """
    Class to extract various acoustic features from audio data.
    """

    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.sr = config.get('sr', 48000)

    def _extract_features(self, audio, sr):
        # Compute the Short-Time Fourier Transform (STFT)
        S = np.abs(librosa.stft(audio))
        S = np.where(S >= np.percentile(S, 5), S, 0)  # Remove very weak frequencies

        # Extract features
        meanfreq = np.mean(S) / 1000  # Mean frequency (kHz)
        sd = np.std(S)  # Standard deviation of frequency
        median = np.median(S) / 1000  # Median frequency (kHz)
        Q25 = np.percentile(S, 25) / 1000  # First quantile (kHz)
        Q75 = np.percentile(S, 75) / 1000  # Third quantile (kHz)
        IQR = Q75 - Q25  # Interquantile range (kHz)
        skew = scipy.stats.skew(S.flatten())  # Skewness
        kurt = scipy.stats.kurtosis(S.flatten())  # Kurtosis
        sp_ent = -np.sum(S * np.log(S + 1e-9))  # Spectral entropy
        sfm = np.mean(S) / (np.mean(S ** 2) ** 0.5)  # Spectral flatness
        mode = scipy.stats.mode(S.flatten(), keepdims=True).mode[0]  # Mode frequency
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr).mean()  # Frequency centroid
        peakf = np.argmax(S) / len(S) * sr  # Peak frequency
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        meanfun = np.mean(pitches[pitches > 0])  # Mean fundamental frequency
        minfun = np.min(pitches[pitches > 0])  # Min fundamental frequency
        maxfun = np.max(pitches[pitches > 0])  # Max fundamental frequency
        meandom = np.mean(magnitudes)  # Mean dominant frequency
        mindom = np.min(magnitudes)  # Min dominant frequency
        maxdom = np.max(magnitudes)  # Max dominant frequency
        dfrange = maxdom - mindom  # Range of dominant frequency
        modindx = np.mean(np.abs(np.diff(pitches[pitches > 0]))) / (maxfun - minfun)  # Modulation index

        # Combine all features into a single vector
        features = np.array([
            meanfreq, sd, median, Q25, Q75, IQR, skew, kurt, sp_ent, sfm, mode,
            centroid, peakf, meanfun, minfun, maxfun, meandom, mindom, maxdom,
            dfrange, modindx
        ])

        return features

    def fit(self, X, y=None):
        return self

    def transform(self, audios: list) -> np.ndarray:
        """
        Extract acoustic features from audio data.

        Parameters:
            audios (list): List of audio data.

        Returns:
            np.ndarray: Extracted features.
        """
        return np.array([self._extract_features(audio, self.sr) for audio in audios])
