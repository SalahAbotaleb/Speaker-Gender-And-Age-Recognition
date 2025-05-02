from .feature_extraction import FeatureExtractor
import numpy as np
from python_speech_features import sigproc
from scipy.signal import lfilter
from scipy.linalg import solve_toeplitz

class PLP(FeatureExtractor):
    def __init__(self, order=13, winlen=0.025, winstep=0.01, nfilters=21, preemph=0.97, NFFT=2048, pooling='mean_std'):
        self.order = order
        self.winlen = winlen
        self.winstep = winstep
        self.nfilters = nfilters
        self.preemph = preemph
        self.NFFT = NFFT
        self.pooling = pooling  # 'mean', 'mean_std', or None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._extract(x) for x in X])
    
    def _lpc(self, signal, order):
        # Compute autocorrelation
        r = np.correlate(signal, signal, mode='full')
        r = r[len(signal)-1:len(signal)+order]
        
        # Use Levinson-Durbin recursion to solve for coefficients
        coeffs = solve_toeplitz(r[:-1], -r[1:])
        return np.concatenate([[1.0], coeffs])  # Add gain term

    def _extract(self, audio_obj):
        sig, rate = audio_obj, 48000
        # sig = lfilter([1, -self.preemph], 1, sig)

        frame_len = int(self.winlen * rate)
        # frame_step = int(self.winstep * rate)
        frame_step = 512

        frames = sigproc.framesig(sig, frame_len, frame_step, winfunc=np.hamming)
        pow_spec = sigproc.powspec(frames, self.NFFT)
        eql_pow = self._apply_equal_loudness(pow_spec, rate)
        bark_energies = np.dot(eql_pow, self._bark_filterbank(rate).T)
        bark_energies = np.power(bark_energies, 0.33)  # intensity-loudness compression

        # Use our own LPC implementation
        lpc_coeffs = np.array([self._lpc(bark_energies[i], self.order) for i in range(len(bark_energies))])
        lpc_coeffs = lpc_coeffs[:, 1:]  # drop gain term

        if self.pooling == 'mean':
            return np.mean(lpc_coeffs, axis=0)
        elif self.pooling == 'mean_std':
            return np.concatenate([np.mean(lpc_coeffs, axis=0), np.std(lpc_coeffs, axis=0)])
        else:
            return lpc_coeffs  # (T, D)

    def _bark_filterbank(self, rate):
        def hz2bark(f): return 6 * np.arcsinh(f / 600)
        def bark2hz(b): return 600 * np.sinh(b / 6)

        low_bark = hz2bark(0)
        high_bark = hz2bark(rate / 2)
        bark_points = np.linspace(low_bark, high_bark, self.nfilters + 2)
        bin_freqs = np.linspace(0, rate / 2, self.NFFT // 2 + 1)
        bin_barks = hz2bark(bin_freqs)

        fb = np.zeros((self.nfilters, len(bin_freqs)))
        for i in range(1, self.nfilters + 1):
            l, c, r = bark_points[i - 1], bark_points[i], bark_points[i + 1]
            fb[i - 1] = np.maximum(0, 1 - np.abs((bin_barks - c) / (r - l)))
        return fb

    def _apply_equal_loudness(self, pow_spec, rate):
        freqs = np.linspace(0, rate / 2, pow_spec.shape[1])
        eql = (freqs**2 + 1.6e5) * freqs**4 / ((freqs**2 + 1.6e5)**2 * (freqs**2 + 1.44e6))
        eql = eql / np.max(eql)
        return pow_spec * eql