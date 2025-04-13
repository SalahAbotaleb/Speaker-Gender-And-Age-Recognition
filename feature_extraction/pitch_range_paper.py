from .feature_extraction import FeatureExtractor
import numpy as np
import librosa
import scipy.signal as signal


class PitchRange2(FeatureExtractor):
    """
    Class to extract 3PR pitch range features (6 features) from audio data.
    Based on Barkana & Zhou (2015): max/min and std/mean pitch from 3 filtered bands.
    """

    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.sr = config.get("sr", 22050)

        # Define 3 filters: 1 LPF and 2 HPFs based on the paper
        self.filters = {
            "LPF": signal.butter(4, 840, btype="low", fs=self.sr, output="sos"),
            "HPF_1": signal.butter(4, [1200, 4000], btype="band", fs=self.sr, output="sos"),
            "HPF_2": signal.butter(4, [1600, 4000], btype="band", fs=self.sr, output="sos"),
        }

    def _acf_pitch_track(self, audio):
        """
        Estimate pitch for each frame using short-time autocorrelation.
        """
        frame_size = int(self.sr * 0.032)
        hop_size = int(self.sr * 0.016)
        pitches = []

        for start in range(0, len(audio) - frame_size + 1, hop_size):
            frame = audio[start:start + frame_size]
            frame = frame - np.mean(frame)

            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]

            peaks, _ = signal.find_peaks(autocorr[1:])
            if len(peaks) > 0:
                first_peak = peaks[0] + 1
                pitch = self.sr / first_peak
                pitches.append(pitch)
        
        return np.array(pitches)

    def _pr_features(self, pitches):
        pitches = pitches[pitches > 0]
        if len(pitches) == 0:
            return [0.0, 0.0]
        max_p = np.max(pitches)
        min_p = np.min(pitches)
        mean_p = np.mean(pitches)
        std_p = np.std(pitches)
        par1 = max_p / min_p if min_p > 0 else 0.0
        par2 = std_p / mean_p if mean_p > 0 else 0.0
        return [par1, par2]

    def _extract_pitch_range(self, audio, sr):
        """
        Extract 3PR features by filtering into 3 bands and applying pitch extraction.
        Returns 6 features: 3 bands × (Par1, Par2)
        """
        features = []
        for name, filt in self.filters.items():
            filtered = signal.sosfilt(filt, audio)
            pitch_track = self._acf_pitch_track(filtered)
            features.extend(self._pr_features(pitch_track))
        return features

    def fit(self, X, y=None):
        return self

    def transform(self, audios: list) -> np.ndarray:
        """
        Extract 3PR features from audio data.

        Parameters:
            audios (list): List of 1D np.ndarray audio signals.

        Returns:
            np.ndarray: Shape (n_samples, 6) feature array.
        """
        return np.array([self._extract_pitch_range(audio, self.sr) for audio in audios])
