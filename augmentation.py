import random
import soundfile as sf
import librosa
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
audio_dir = os.path.join(".", "data", "filtered_clips")

def augment_audio(audio, sr):
    """
    Augments an audio signal by applying random transformations.

    Args:
        audio (numpy.ndarray): The audio signal.
        sr (int): The sample rate of the audio.

    Returns:
        numpy.ndarray: The augmented audio signal.
    """
    
    # Randomly apply pitch shift
    if random.random() < 0.5:
        steps = random.uniform(-0.1, 0.1)  # Shift pitch by ±10%
        audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=steps * 12)

    # Randomly apply time stretch
    if random.random() < 0.5:
        rate = random.uniform(0.95, 1.05)  # Stretch by ±5%
        audio = librosa.effects.time_stretch(y=audio, rate=rate)

    # Randomly add noise
    if random.random() < 0.5:
        noise = np.random.normal(0, 0.005, audio.shape)
        audio = audio + noise

    # Randomly add mild background noise
    if random.random() < 0.5:
        background_noise = np.random.normal(0, 0.002, audio.shape)
        audio = audio + background_noise
    


    # Clip audio to ensure values are within valid range
    audio = np.clip(audio, -1.0, 1.0)

    return audio


def augment_and_add_data(df, target_class, augment_count):
    """
    Augments audio data for a specific class and adds it to the dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing audio data.
        target_class (int): The class for which data augmentation is performed.
        augment_count (int): The number of augmented samples to generate.

    Returns:
        pd.DataFrame: The updated dataframe with augmented data.
    """
    os.makedirs('./data/augmented_clips', exist_ok=True)
    
    class_df = df[df['label'] == target_class]
    augmented_data = []
    augmented_data_dir = "./data/augmented_clips"
    for _ in tqdm(range(augment_count), desc="Augmenting audio data"):
        sample = class_df.sample(n=1, random_state=random.randint(0, 1000)).iloc[0]
        file_path = os.path.join(audio_dir, sample['path'])
        audio, sr = librosa.load(file_path, sr=None)
        augmented_audio = augment_audio(audio, sr)
        augmented_file_path = f"{augmented_data_dir}/{sample['path'].split('.')[0]}_augmented.mp3"
        sf.write(augmented_file_path, augmented_audio, sr)

        augmented_data.append({
            'path': augmented_file_path,
            'up_votes': sample['up_votes'],
            'down_votes': sample['down_votes'],
            'label': sample['label']
        })

    augmented_df = pd.DataFrame(augmented_data)
    return pd.concat([df, augmented_df], ignore_index=True)

