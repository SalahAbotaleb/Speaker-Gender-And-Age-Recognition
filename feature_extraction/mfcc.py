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
        self.n_mfcc = config.get("n_mfcc", 75)
        self.n_fft = config.get("n_fft", 2048)
        self.hop_length = config.get("hop_length", 512)
        self.sr = config.get('sr', 48000)
        self.context = config.get("context", 0)

        self.use_spectral_subtraction = config.get("use_spectral_subtraction", False)
        self.use_smoothing = config.get("use_smoothing", True)
        self.use_cmvn = config.get("use_cmvn", False)
        self.use_deltas = config.get("use_deltas", False)

    def spectral_subtraction(self, y: np.ndarray, sr: int, noise_frames: int = 5) -> np.ndarray:
        
        # Compute STFT
        D = librosa.stft(y)
        
        # Estimate noise
        noise_mag = np.mean(np.abs(D[:, :noise_frames]), axis=1, keepdims=True)
        
        # Subtract noise
        D_denoised = np.maximum(np.abs(D) - noise_mag, 0) * np.exp(1j * np.angle(D))
        
        # Inverse STFT
        y_denoised = librosa.istft(D_denoised)
        
        return y_denoised
    
    def smooth_features(self, features, window_size=3):
        # features: (n_features, n_frames)
        smoothed = np.copy(features)
        for i in range(features.shape[0]):  # For each feature dimension
            smoothed[i] = np.convolve(features[i], np.ones(window_size)/window_size, mode='same')
        return smoothed
    
    def cmvn(self, features):
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True) + 1e-9
        normalized = (features - mean) / std
        return normalized

    def _extract_mfcc(self, audio, sr):

        if self.use_spectral_subtraction:
            audio = self.spectral_subtraction(audio, sr)
        # Extract MFCCs with paper's parameters
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft,
            hop_length=self.hop_length, window='hann'
        )

        if self.use_smoothing:
            mfcc = self.smooth_features(mfcc)

        if self.use_cmvn:
            mfcc = self.cmvn(mfcc)

        features = [mfcc]
        if self.use_deltas:
            delta_width = max(3, min(9, mfcc.shape[1]) | 1)  # Ensure width is odd and >= 3
            delta_mfcc = librosa.feature.delta(mfcc, width=delta_width)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2, width=delta_width)
            features.extend([delta_mfcc, delta2_mfcc])

        features = np.vstack(features)

        features = features.T

        pad_width = self.context
        features_padded = np.pad(features, ((pad_width, pad_width), (0, 0)), mode='edge')
        
        stacked_features = []
        for i in range(pad_width, len(features_padded) - pad_width):
            stacked = features_padded[i - self.context:i + self.context + 1].reshape(-1)
            stacked_features.append(stacked)
        
        stacked_features = np.array(stacked_features)  # (frames, 39 * (2*context +1))

        # 7. Summarize
        feature_mean = np.mean(stacked_features, axis=0)
        feature_std = np.std(stacked_features, axis=0)
        # feature_median = np.median(stacked_features, axis=0)
        # mode
        # feature_q25 = np.quantile(stacked_features, 0.25, axis=0)
        # feature_q75 = np.quantile(stacked_features, 0.75, axis=0)
        # feature_min = np.min(stacked_features, axis=0)
        # feature_max = np.max(stacked_features, axis=0)

        
        final_feature_vector = np.concatenate([feature_mean, feature_std]) #, feature_median, feature_q25, feature_q75, feature_min, feature_max

        return final_feature_vector

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