from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
import numpy as np
import librosa
import cloudpickle
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import copy

class Audio:
    def __init__(self, data, sr):
        self.data = data
        self.sampling_rate = sr

class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor class to handle missing values and categorical encoding.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        raise NotImplementedError("Subclasses must implement this method.")
    
class DCRemover(Preprocessor):

    def __init__(self):
        super().__init__()

    def transform(self, X):
        return [Audio(x.data - np.mean(x.data), x.sampling_rate) for x in X]
    
class QualityEnhancer(Preprocessor):

    def __init__(self, coef=0.95):
        self.coef = coef
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [Audio(librosa.effects.preemphasis(x.data, coef=self.coef), x.sampling_rate) for x in X]
    
class LightLoudnessNormalizer(Preprocessor):

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self
    
    def _rms_normalize(self, signal, target_rms=0.1):
        rms = np.sqrt(np.mean(np.square(np.asarray(signal))))
        return signal * (target_rms / (rms + 1e-8))

    def transform(self, X):
        return [Audio(self._rms_normalize(x.data), x.sampling_rate) for x in X]

class SilenceRemover(Preprocessor):

    def __init__(self, amplitude_threshold=0.005, interval_ratio=0.5):
        self.amplitude_threshold = amplitude_threshold
        self.interval_ratio = interval_ratio
        super().__init__()

    def transform(self, X):
        trimmed_audios = []
        for x in X:
            # Find indices where the audio is above the amplitude threshold
            non_silent = np.abs(x.data) > self.amplitude_threshold
            non_silent_indices = np.nonzero(non_silent)[0]

            # If there are no non-silent parts, return an empty array with a warning
            if len(non_silent_indices) == 0:
                print("Warning: No audio above threshold found. Returning original audio.")
                trimmed_audio = np.array(x.data, dtype=np.float64)
            else:
                # Identify continuous regions of sound
                regions = []
                region_start = non_silent_indices[0]
                prev_idx = non_silent_indices[0]
                
                # Define gap tolerance (in samples) - adjust as needed
                gap_tolerance = int(self.interval_ratio * x.sampling_rate)
                
                for idx in non_silent_indices[1:]:
                    # If the gap is too large, end the current region and start a new one
                    if idx - prev_idx > gap_tolerance:
                        regions.append((region_start, prev_idx))
                        region_start = idx
                    prev_idx = idx
                
                # Add the last region
                regions.append((region_start, non_silent_indices[-1]))
                
                # Concatenate all non-silent regions
                trimmed_audio = np.concatenate([x.data[start:end+1] for start, end in regions], dtype=np.float64)
            trimmed_audios.append(trimmed_audio)
        return trimmed_audios
    
class FeatureExtractor(BaseEstimator, TransformerMixin, ABC):
    """
    FeatureExtractor class to handle feature extraction from audio data.
    """

    def __init__(self, config: dict = None):
        self.config = config if config is not None else {}

    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def transform(self, audio: np.ndarray) -> np.ndarray:
        pass

class MFCC(FeatureExtractor):
    """
    Class to extract MFCC features from audio data.
    """

    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.n_mfcc = config.get("n_mfcc", 30)
        self.n_fft = config.get("n_fft", 2048)
        self.hop_length = config.get("hop_length", 400)
        self.sr = config.get('sr', 22050)
        self.context = config.get("context", 1)

        self.use_spectral_subtraction = config.get("use_spectral_subtraction", False)
        self.use_smoothing = config.get("use_smoothing", False)
        self.use_cmvn = config.get("use_cmvn", True)
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
        
        final_feature_vector = np.concatenate([feature_mean, feature_std])  # Final vector

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

class Model(ClassifierMixin, BaseEstimator, ABC):
    """
    FeatureExtractor class to handle feature extraction from audio data.
    """

    def __init__(self, config: dict = {}):
        self.config = config if config is not None else {}

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

class SVM(Model):
    """
    Support Vector Machine (SVM) classifier.
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.model = None  # Placeholder for the SVM model

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the SVM model to the training data.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Labels.
        """

        # Initialize the SVC model
        svm = SVC(kernel='rbf', C=1000, gamma=0.0001, random_state=42, class_weight='balanced')

        if self.config.get('grid_search', False):
        # Define the parameter grid for hyperparameter tuning
          param_grid = {
              'C': [1, 10, 100, 1000],
              'gamma': [0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'linear', 'poly']
          }

          # Perform grid search with cross-validation
          grid = GridSearchCV(svm, param_grid, cv=10, n_jobs=-1)
          grid.fit(X, y)

          self.model = grid.best_estimator_
        else: 
          # Fit the model directly without grid search
          svm.fit(X, y)

          self.model = svm
        
        self.is_fitted_ = True
        return self

    def predict(self, X):
      """
      Predict labels for the given feature matrix.

      Parameters:
          X (np.ndarray): Feature matrix.

      Returns:
          np.ndarray: Predicted labels.
      """
      return self.model.predict(X) if self.model else None

with open(r".\trials\model_21_04_2025_T15_05_46_best\model.pkl", "rb") as f:
    model_pipe = cloudpickle.load(f)

config = {'n_mfcc': 75, 'n_fft': 2048, 'hop_length': 512, 'context': 3, 'use_spectral_subtraction': False, 'use_smoothing': True, 'use_cmvn': False, 'use_deltas': True, 'sr': 40000}

pipeline = Pipeline([
    ('remove_dc', DCRemover()),
    ('enhance_quality', QualityEnhancer()),
    ('loudness_normalizer', LightLoudnessNormalizer()),
    ('silence_remover', SilenceRemover(amplitude_threshold=0.0005)),
    ('feature_extractor', MFCC(config)),
    ('pca', model_pipe.named_steps['pca']),
    ('classifier', copy.deepcopy(model_pipe.named_steps['pipeline-2'].named_steps['svm'].model))
])

with open(r".\trials\model_21_04_2025_T15_05_46_best\export.pkl", "wb") as f:
    cloudpickle.dump(pipeline, f)