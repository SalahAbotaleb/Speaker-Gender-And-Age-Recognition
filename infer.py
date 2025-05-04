from preprocessing.dc_removal import DCRemover
from preprocessing.quality_enhancer import QualityEnhancer
from preprocessing.light_loudness_normalization import LightLoudnessNormalizer
from preprocessing.silence_removal import SilenceRemover

from feature_extraction.mfcc import MFCC
from feature_extraction.hfcc import HFCC
from feature_extraction.log_mel_energy import LogMelEnergy

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.pipeline import FeatureUnion

from audio import Audio

import sys
import os
import librosa
import time
import joblib
from tqdm import tqdm
import cloudpickle

def load_audio_files(data_folder):
  audio_data = []
  for root, _, files in os.walk(data_folder):
    # Sort files by integer value extracted from filename (before extension)
    def extract_int(filename):
      name, _ = os.path.splitext(filename)
      try:
        return int(name)
      except ValueError:
        return float('inf')  # Non-integer filenames go last
    
    files = sorted(files, key=extract_int)
    for file in tqdm(files, desc=f"Loading audio files from {root}"):
      audio_path = os.path.join(root, file)
      try:
        y, sr = librosa.load(audio_path, sr=None)
        audio_data.append(Audio(y, sr))
      except Exception as e:
        print(f"Error loading {audio_path}: {e}")
  return audio_data

def main():
    if len(sys.argv) < 2:
        print("Usage: infer.exe <path_to_data_folder>")
        sys.exit(1)
    data_folder = sys.argv[1]
    if not (os.path.isdir(data_folder) and os.access(data_folder, os.R_OK | os.X_OK)):
      print(f"Error: The folder '{data_folder}' is not a readable directory.")
      sys.exit(1)

    audio_files = load_audio_files(data_folder)
    if not audio_files:
        print(f"No audio files found in the directory '{data_folder}'.")
        sys.exit(1)
    print(f"Found {len(audio_files)} audio files in the directory '{data_folder}'")

    feature_extractors = FeatureUnion([
      ('mfcc', MFCC({'n_mfcc': 75, 'n_fft': 2048, 'hop_length': 512, 'context': 0, 
        'use_spectral_subtraction': False, 'use_smoothing': True, 
        'use_cmvn': False, 'use_deltas': False, 'sr': 48000})),
      ('hfcc', HFCC()),
      ('logmel', LogMelEnergy(sr=48000, n_mels=75, winlen=0.025, winstep=0.01, preemph=0.97, pooling='std'))
    ])

    model = joblib.load('model.joblib')

    pipeline = Pipeline([
        ('preprocessing', make_pipeline(
          DCRemover(),
          QualityEnhancer(),
          LightLoudnessNormalizer(),
          SilenceRemover()
        )),
        ('features', feature_extractors),
        ('classifier', model)
    ])

    with open('pipeline_cloudpickle.pkl', 'wb') as f:
      cloudpickle.dump(pipeline, f)

    predictions = []
    times = []

    for audio in tqdm(audio_files, desc="Processing audio files"):
      start = time.time()
      features = pipeline.named_steps['features'].transform(
        pipeline.named_steps['preprocessing'].transform([audio])
      )
      pred = pipeline.named_steps['classifier'].predict(features)[0]
      end = time.time()
      predictions.append(pred)
      times.append(end - start)
    
    print(f"Predictions: {predictions}")
    print(f"Processing times: {times}")

    with open('results.txt', 'w') as f:
      for pred in predictions:
        f.write(f"{pred}\n")

    with open('time.txt', 'w') as f:
      for t in times:
        f.write(f"{t}\n")

if __name__ == "__main__":
  main()