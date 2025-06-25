# Speaker-Gender-And-Age-Recognition
Speaker Gender Age Recognition Project

This project implements a complete machine learning pipeline to classify a speaker's gender and age from audio recordings. It includes a comprehensive suite of tools for audio preprocessing, a wide array of feature extractors, and a selection of classification models, all orchestrated within a `scikit-learn` framework. The final model uses a specialized `GenderAgePipeline` to first predict gender and then use a gender-specific model to predict age.

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
  - [Audio Preprocessing](#audio-preprocessing)
  - [Feature Extraction](#feature-extraction)
  - [Modeling](#modeling)
- [Workflow](#workflow)
  - [1. Feature Selection and Experimentation](#1-feature-selection-and-experimentation)
  - [2. Final Model Training and Optimization](#2-final-model-training-and-optimization)
- [How to Use](#how-to-use)
- [Dependencies](#dependencies)

## Project Overview

The primary goal of this project is to accurately classify a speaker into one of four categories based on their voice:
1.  **Young Male (20s)**
2.  **Young Female (20s)**
3.  **Old Male (50s)**
4.  **Old Female (50s)**

This is achieved through a modular and extensible pipeline that performs the following steps:
1.  **Cleans and normalizes** raw audio files.
2.  **Extracts** a rich and diverse set of acoustic features.
3.  **Trains** a hierarchical classification model that is optimized for performance using hyperparameter tuning.

## Project Structure

The repository is organized into several key directories:

-   `preprocessing/`: Contains modules for cleaning and preparing audio data.
-   `feature_extraction/`: Contains a wide variety of modules for extracting acoustic features from audio.
-   `models/`: Contains implementations of various classification models and the custom `GenderAgePipeline`.
-   `data/`: Should contain the audio files and the labels.
-   `feature_combinations.ipynb`: A Jupyter Notebook for experimenting with different feature sets to find the most effective combination.
-   `main.ipynb`: The main Jupyter Notebook for building, training, and evaluating the final, optimized model.
-   `trained_model.joblib`: The saved final model after training and hyperparameter tuning.

---

## Core Components

### Audio Preprocessing

The modules in the `preprocessing/` directory are designed to clean and standardize audio signals before feature extraction. Each is a `scikit-learn` compatible transformer.

-   **`DCRemover`**: Removes the DC offset from the audio signal.
-   **`HPF` (High-Pass Filter)**: Attenuates low-frequency noise below 60 Hz.
-   **`VADNormalizer` (Voice Activity Detection Normalizer)**: Normalizes the loudness of only the voiced segments of the audio.
-   **`VolumeNormalizer`**: Adjusts the overall audio volume to a target integrated loudness.
-   **`NoiseReducer`**: Reduces background noise.
-   **`SilenceRemover`**: Trims leading and trailing silences.
-   **`SpeechFilter`**: Applies a band-pass filter to isolate typical speech frequencies.
-   **`QualityEnhancer`**: Applies a pre-emphasis filter to boost high frequencies.

### Feature Extraction

The project includes an extensive library of feature extractors in the `feature_extraction/` directory, allowing for the creation of a rich feature space.

#### Cepstral & Perceptual Features
-   **`MFCC`**: Mel-Frequency Cepstral Coefficients, a standard for speech processing. This implementation is highly customizable with options for spectral subtraction, smoothing, CMVN, and delta features.
-   **`LPCC`**: Linear Predictive Cepstral Coefficients.
-   **`PLP`**: Perceptual Linear Prediction coefficients.
-   **`LSP`**: Line Spectral Pairs, an alternative representation of LPCs.
-   **`HFCC`**: Human Factor Cepstral Coefficients, designed for noise robustness.
-   **`CPPS`**: Cepstral Peak Prominence Smoothed, related to voice periodicity.

#### Pitch-Based Features
-   **`PitchFeatures`**: Mean, standard deviation, max, and min of the fundamental frequency (F0).
-   **`PitchRange`**: The range between the maximum and minimum F0.
-   **`PitchRange2`**: Implements the "3PR" features from the Barkana & Zhou (2015) paper.
-   **`Jitter`**: Measures the short-term variation in F0.
-   **`FundamentalFrequency`**: Extracts the F0 using an autocorrelation method.

#### Spectral Features
-   **`SpectralFeatures`**: A comprehensive set of features including Spectral Centroid, Bandwidth, Contrast, Flatness, and Rolloff.
-   **`AlphaRatio`**: The ratio of low-frequency to high-frequency energy.
-   **`MeanMinMaxFrequency`**: The mean, min, and max frequencies in the signal.
-   **`PolyFeatures`**: Polynomial features fit to the spectrogram.
-   **`GenderFeatures`**: A set of 21 acoustic features commonly used for gender recognition.

### Modeling

The `models/` directory contains a selection of classifiers and the primary `GenderAgePipeline`.

-   **Classification Models**:
    -   `SVM` (Support Vector Machine)
    -   `GradientBoost` (Gradient Boosting)
    -   `NN` (Neural Network using Keras)
    -   `ExtraTree` (Extra Trees Classifier)
    -   `KNN` (K-Nearest Neighbors)
    -   `QDA` (Quadratic Discriminant Analysis)
    -   `LogisticRegression`

-   **`GenderAgePipeline`**:
    This custom pipeline implements a two-step classification strategy:
    1.  A primary model predicts the speaker's gender.
    2.  Based on the predicted gender, a second, dedicated model (one for 'male', one for 'female') predicts the speaker's age. This hierarchical approach breaks down the complex 4-class problem into simpler sub-problems, often leading to improved accuracy.

---

## Workflow

The project is designed to be run in two main stages, each corresponding to a Jupyter Notebook.

### 1. Feature Selection and Experimentation

**Notebook:** `feature_combinations.ipynb`

This notebook is used to determine the most predictive set of features.

-   It defines various combinations of the available feature extractors.
-   It iterates through each combination, building a standardized pipeline with a basic `SVC` model.
-   It trains and evaluates each pipeline, reporting the accuracy for each feature set.
-   The results from this notebook inform the feature selection for the final model.

### 2. Final Model Training and Optimization

**Notebook:** `main.ipynb`

This notebook builds, trains, and saves the definitive model using the best practices and insights gathered from the experimentation phase.

-   It constructs a complete pipeline using the most effective combination of preprocessing steps and features.
-   It uses the custom `GenderAgePipeline` with `SVM` as the base classifier.
-   The entire pipeline is wrapped in `GridSearchCV` to perform an exhaustive search for the optimal hyperparameters (`C` and `gamma` for the SVMs).
-   The best model found is thoroughly evaluated using a classification report and a confusion matrix.
-   Finally, the fully trained and optimized pipeline is saved to `trained_model.joblib` for future use.

---

## How to Use

1.  **Setup Environment**:
    Install all the required dependencies. It is recommended to use a virtual environment.

2.  **Data Setup**:
    -   Create a `data/` directory in the project root.
    -   Inside `data/`, place your audio files (e.g., in a subdirectory like `data/clips/`).
    -   Create a `labels.csv` file in the `data/` directory with columns for the audio file paths and their corresponding labels (`gender` and `age`) or use one of the files provided.

3.  **Run Feature Experimentation (Optional)**:
    -   Open and run the `feature_combinations.ipynb` notebook to experiment with different feature sets and identify the most effective ones for your dataset.

4.  **Train the Final Model**:
    -   Open and run the `main.ipynb` notebook.
    -   This will execute the entire pipeline: loading data, preprocessing, feature extraction, training the `GenderAgePipeline`, and performing the grid search for hyperparameter optimization.
    -   The trained model will be saved as `trained_model.joblib`.

5.  **Make Predictions**:
    -   You can load the saved model to make predictions on new audio files.
    ```python
    import joblib

    # Load the trained model
    model = joblib.load('trained_model.joblib')

    # Load a new audio file (as a numpy array)
    new_audio, sr = librosa.load('path/to/new_audio.wav', sr=48000)

    # Make a prediction
    prediction = model.predict([new_audio])
    ```

## Dependencies
- pandas
- numpy
- librosa
- scikit-learn
- tensorflow (for the NN model)
- noisereduce
- pyloudnorm
- webrtcvad
- python_speech_features
- joblib
- tqdm
- matplotlib
