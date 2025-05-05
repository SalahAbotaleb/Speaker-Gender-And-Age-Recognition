import numpy as np
from scipy.signal import lfilter
import librosa
from .feature_extraction import FeatureExtractor

class LogMelEnergy(FeatureExtractor):
    def __init__(self, sr=48000, n_mels=75, winlen=0.025, winstep=0.01, preemph=0.97, pooling='std'):
        self.sr = sr
        self.n_mels = n_mels
        self.winlen = winlen
        self.winstep = winstep
        self.preemph = preemph
        self.pooling = pooling  # 'mean', 'mean_std', 'std', or None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._extract(audio) for audio in X])

    def _extract(self, audio_obj):
        signal, sr = audio_obj, 48000
        # signal = lfilter([1, -self.preemph], 1, signal)

        # hop_length = int(self.winstep * sr)
        win_length = int(self.winlen * sr)

        mel_spec = librosa.feature.melspectrogram(
            y=signal,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            win_length=win_length,
            window='hamming',
            n_mels=self.n_mels,
            power=2.0
        )

        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # if self.pooling == 'mean':
        #     return np.mean(log_mel_spec, axis=1)
        # elif self.pooling == 'mean_std':
        #     return np.concatenate([np.mean(log_mel_spec, axis=1), np.std(log_mel_spec, axis=1)])
        # elif self.pooling == 'std':
        return np.std(log_mel_spec, axis=1)
        # else:
        #     return log_mel_spec.T  # return (frames, bands)