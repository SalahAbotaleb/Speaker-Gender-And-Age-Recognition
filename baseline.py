import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import noisereduce as nr

# 1. Load and Preprocess Dataset
# ---------------------------------

metadata_path = ".\\data\\filtered_data_labeled.tsv"
audio_dir = ".\\data\\filtered_clips"

# Load metadata
df = pd.read_csv(metadata_path, sep='\t')

# Filter valid samples with age and gender info
df = df[df['age'].notna() & df['gender'].notna() & df['label'].notna()]

# Filter valid classes and balance to 670 samples per class
samples = df['label'].value_counts().min()
balanced_samples = []
for cls in df['label'].unique():
    cls_df = df[df['label'] == cls]
    sampled = cls_df.sample(n=samples, random_state=42)  # Random sampling
    balanced_samples.append(sampled)

balanced_df = pd.concat(balanced_samples)

# 2. Feature Extraction (MFCCs)
# ---------------------------------
def extract_mfcc(file_path):
    # Load audio, resample to 2050 Hz
    if not os.path.exists(file_path):
      print(f"Warning: File not found: {file_path}")
      return None
    
    try:
      audio, sr = librosa.load(file_path, sr=2050)
    except Exception as e:
      return None
    
    noise_reduced = np.array(nr.reduce_noise(audio, sr=sr, time_mask_smooth_ms=150))
    silence_removed = np.array(librosa.effects.trim(noise_reduced, top_db=100)[0])
    
    # Extract MFCCs with paper's parameters
    mfccs = librosa.feature.mfcc(
        y=silence_removed, sr=sr, n_mfcc=40, n_fft=2048,
        hop_length=512, window='hann'
    )
    
    # Take mean across time to get 40 features
    return np.mean(mfccs, axis=1)

# Extract features and labels

# Print initial stats
num_files = sum(1 for entry in os.scandir(audio_dir) if entry.is_file())

print(f"Number of files in directory: {num_files}")
print(f"Number of records in DataFrame: {len(df)}")

# Filter DataFrame to only include records with existing files
valid_indices = []
for idx, row in tqdm(balanced_df.iterrows(), total=len(balanced_df), desc="Checking files"):
  file_path = os.path.join(audio_dir, row['path'])
  if os.path.exists(file_path):
    valid_indices.append(idx)

# Use only records with existing files
balanced_df = balanced_df.loc[valid_indices]
print(f"Records with existing files: {len(balanced_df)}")

features = []
labels = []
error_file_paths = []
for idx, row in tqdm(balanced_df.iterrows(), total=len(balanced_df), desc="Extracting features"):
  file_path = os.path.join(audio_dir, row['path'])
  mfcc = extract_mfcc(file_path)
  if mfcc is None: 
    print(f"Error: {file_path}")
    error_file_paths.append(file_path)
    continue  # Skip if file not found or error
  features.append(mfcc)
  labels.append(row['label'])

print(f"Number of error files: {len(error_file_paths)}")
with open("error_file_paths.txt", "w") as f:
  for file_path in error_file_paths:
    f.write(file_path + "\n")
X = np.array(features)
y = np.array(labels)

# 3. Train/Test Split (75:25)
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4. SVM Training with Grid Search (or use paper's best params)
# ---------------------------------
# Paper's best params: C=1000, gamma=0.0001, kernel='rbf'
svm = SVC(kernel='rbf', C=1000, gamma=0.0001, random_state=42)

# param_grid = {
#     'C': [1, 10, 100, 1000],
#     'gamma': [0.1, 0.01, 0.001, 0.0001],
#     'kernel': ['rbf', 'linear', 'poly']
# }
# grid = GridSearchCV(SVC(), param_grid, cv=10, n_jobs=-1)
# grid.fit(X_train, y_train)
# svm = grid.best_estimator_

svm.fit(X_train, y_train)

# 5. Evaluation
# ---------------------------------
y_pred = svm.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))