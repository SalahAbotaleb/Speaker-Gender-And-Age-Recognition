from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pyloudnorm as pyln
import webrtcvad

import sys
sys.path.append('..')
from audio import Audio

class VADNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, target_loudness=-23.0, max_gain_db=20.0, vad_aggressiveness=2):
        self.target_loudness = target_loudness
        self.max_gain_db = max_gain_db
        self.vad_aggressiveness = vad_aggressiveness
        self.vad = webrtcvad.Vad(self.vad_aggressiveness)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X should be a list or array of Audio objects
        return [self._normalize_voiced_segments(audio) for audio in X]

    def _normalize_voiced_segments(self, audio: Audio) -> Audio:
        sample_rate = audio.sampling_rate
        audio_data = audio.data
        audio_data = audio_data / np.max(np.abs(audio_data))  # Ensure within [-1, 1]

        # Convert to 16-bit PCM bytes
        pcm = (audio_data * 32767).astype(np.int16).tobytes()
        frame_ms = 30
        frame_len = int(sample_rate * frame_ms / 1000)

        voiced_mask = np.zeros(len(audio_data), dtype=bool)
        for start in range(0, len(audio_data), frame_len):
            end = start + frame_len
            frame_bytes = pcm[start*2:end*2]
            if len(frame_bytes) < frame_len * 2:
                break
            is_voiced = self.vad.is_speech(frame_bytes, sample_rate)
            if is_voiced:
                voiced_mask[start:end] = True

        voiced_audio = audio_data[voiced_mask]
        if len(voiced_audio) == 0:
            return audio  # Return original to avoid crash

        meter = pyln.Meter(sample_rate)
        loudness = meter.integrated_loudness(voiced_audio)
        gain_db = self.target_loudness - loudness
        adjusted_target = loudness + self.max_gain_db if gain_db > self.max_gain_db else self.target_loudness

        normalized_voiced = pyln.normalize.loudness(voiced_audio, loudness, adjusted_target)

        normalized_audio = np.copy(audio_data)
        normalized_audio[voiced_mask] = normalized_voiced

        # Final clipping protection
        peak = np.max(np.abs(normalized_audio))
        if peak > 1.0:
            normalized_audio = normalized_audio / (peak * 1.1)

        return Audio(normalized_audio, sample_rate)
