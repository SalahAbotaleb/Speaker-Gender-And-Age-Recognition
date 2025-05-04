from .preprocessing import Preprocessor
import numpy as np

class SilenceRemover(Preprocessor):

    def __init__(self, amplitude_threshold=0.0005, interval_ratio=0.5):
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