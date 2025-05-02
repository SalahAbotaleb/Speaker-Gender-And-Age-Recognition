import numpy as np
from scipy.signal import lfilter
from python_speech_features import sigproc
from .feature_extraction import FeatureExtractor
from scipy.linalg import solve_toeplitz

class LPCC(FeatureExtractor):
    def __init__(self, order=13, winlen=0.025, winstep=0.01, preemph=0.97, NFFT=2048, pooling='mean_std'):
        self.order = order
        self.winlen = winlen
        self.winstep = winstep
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
        lpccs = []

        for frame in frames:
            # Replace sigproc.lpc with your custom _lpc method
            lpc = self._lpc(frame, self.order)
            lpcc = self._lpc_to_lpcc(lpc, self.order)
            lpccs.append(lpcc)

        lpccs = np.stack(lpccs)

        if self.pooling == 'mean':
            return np.mean(lpccs, axis=0)
        elif self.pooling == 'mean_std':
            return np.concatenate([np.mean(lpccs, axis=0), np.std(lpccs, axis=0)])
        else:
            return lpccs

    def _lpc_to_lpcc(self, lpc, num_coeffs):
        """Convert LPC to LPCC using recursion."""
        lpcc = np.zeros(num_coeffs)
        lpcc[0] = -np.log(lpc[0]) if lpc[0] > 0 else 0
        for n in range(1, num_coeffs):
            s = 0
            for k in range(1, n):
                s += (k / n) * lpcc[k] * lpc[n - k]
            lpcc[n] = lpc[n] + s
        return lpcc
