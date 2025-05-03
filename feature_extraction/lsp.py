from scipy.signal import lfilter
from scipy.signal import tf2zpk
from python_speech_features import sigproc
from .feature_extraction import FeatureExtractor
import numpy as np
from scipy.linalg import solve_toeplitz

class LSP(FeatureExtractor):
    def __init__(self, order=13, winlen=0.025, winstep=0.01, preemph=0.97, pooling='mean_std'):
        self.order = order
        self.winlen = winlen
        self.winstep = winstep
        self.preemph = preemph
        self.pooling = pooling

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
        frame_step = int(self.winstep * rate)

        frames = sigproc.framesig(sig, frame_len, frame_step, winfunc=np.hamming)
        lsps = []

        for frame in frames:
            # Use your custom _lpc method instead of sigproc.lpc
            lpc = self._lpc(frame, self.order)
            lsp = self._lpc_to_lsp(lpc)
            if lsp is not None:
                lsps.append(lsp)

        lsps = np.stack(lsps) if len(lsps) else np.zeros((1, self.order))

        if self.pooling == 'mean':
            return np.mean(lsps, axis=0)
        elif self.pooling == 'mean_std':
            return np.concatenate([np.mean(lsps, axis=0), np.std(lsps, axis=0)])
        else:
            return lsps

    def _lpc_to_lsp(self, lpc):
        """Estimate LSP from LPC via root finding on symmetric/antisymmetric polynomials."""
        # No need to import signal.roots since we'll use np.roots
        
        A = lpc
        if len(A) < 2:
            return None

        # The order of the LPC filter
        p = len(A) - 1
        
        # Create symmetric and antisymmetric polynomials
        P = np.zeros(p + 1)
        Q = np.zeros(p + 1)
        
        for i in range(p + 1):
            j = p - i
            P[i] = A[i] + A[j]
            Q[i] = A[i] - A[j]
        
        # Find roots using numpy's roots function
        p_roots = np.roots(P)
        q_roots = np.roots(Q)
        
        # Filter roots
        p_roots = np.array([r for r in p_roots if 0.9 < np.abs(r) < 1.1 and np.imag(r) > 0])
        q_roots = np.array([r for r in q_roots if 0.9 < np.abs(r) < 1.1 and np.imag(r) > 0])
        
        # Calculate angles
        p_angles = np.angle(p_roots)
        q_angles = np.angle(q_roots)
        
        # Combine, sort and handle potential empty arrays
        if len(p_angles) == 0 and len(q_angles) == 0:
            return np.zeros(self.order)
        
        all_angles = np.sort(np.concatenate([p_angles if len(p_angles) > 0 else [], 
                                            q_angles if len(q_angles) > 0 else []]))
        
        # Pad result to required length
        result = np.zeros(self.order)
        length = min(len(all_angles), self.order)
        result[:length] = all_angles[:length]

        return result
