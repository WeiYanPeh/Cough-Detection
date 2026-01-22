import warnings
warnings.filterwarnings('ignore')

import os
import time
import json
import csv
import librosa
import librosa.display
import scipy.stats
from scipy.stats import skew, kurtosis
from scipy.signal import hilbert

import pandas as pd
import numpy as np

from tqdm.notebook import tqdm
from collections import Counter
from pprint import pprint

#################################################################################
def pad_array(array, target_length):
    return np.pad(array, (0, max(0, target_length - len(array))), 'constant')

#################################################################################
def split_audio(
        y,  # Signal
        sr, # Sample frequency
        segment_length=10.0, # Segment length 1s
        overlap=0 # Overlap 50%
    ):
    # Calculate the number of samples per segment
    segment_samples = int(segment_length * sr)
    
    # Calculate the step size
    step_size = int(segment_samples * (1 - overlap))
    
    # Initialize the start and end points
    start = 0
    end = segment_samples
    
    segments = []
    
    while start < len(y):
        segment = y[start:end]
        segments.append(segment)
        start += step_size
        end = start + segment_samples
    
    return segments

#################################################################################
def mean_variance_normalize(audio_waveform):
    """
    Normalise a 1D NumPy array representing audio waveform using mean-variance scaling.
    Output will have mean 0 and standard deviation 1.
    """
    mean = np.mean(audio_waveform)
    std = np.std(audio_waveform)
    
    # Avoid division by zero
    if std == 0:
        return np.zeros_like(audio_waveform)
    
    normalized_waveform = (audio_waveform - mean) / std
    return normalized_waveform

#################################################################################
def extract_features(segment, sr):
    segment = mean_variance_normalize(segment) # Normalize
    
    # Time domain features
    mean = np.mean(segment)
    variance = np.var(segment)
    std_dev = np.std(segment)
    max_value = np.max(segment)
    min_value = np.min(segment)
    rms = np.sqrt(np.mean(segment**2))
    skewness = skew(segment)
    kurt = kurtosis(segment)
    median = np.median(segment)
    range_val = np.ptp(segment)
    iqr = np.percentile(segment, 75) - np.percentile(segment, 25)
    zcr = np.mean(librosa.feature.zero_crossing_rate(segment))
    energy = np.mean(np.sum(segment ** 2))
    rmse = np.mean(librosa.feature.rms(y=segment))
    
    # Entropy
    entropy = scipy.stats.entropy(np.abs(segment))
    
    # Frequency domain features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr, hop_length=1024))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr, hop_length=1024))
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=segment, sr=sr, hop_length=1024))
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=segment, hop_length=1024))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr, hop_length=1024))
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=segment, sr=sr))
    
    mfcc = librosa.feature.mfcc(y=segment, sr=sr)

    result_row = [
        mean, variance, std_dev, max_value, min_value, rms,
        skewness, kurt, median, range_val, iqr,
        zcr, energy, rmse, entropy,
        spectral_centroid, spectral_bandwidth, spectral_contrast,
        spectral_flatness, spectral_rolloff, chroma_stft,
    ]

    for e in mfcc:
        result_row.append(np.mean(e))
        result_row.append(np.std(e))
    
    return result_row



#################################################################################
# Function to process each row and extract features
#################################################################################
def process_row(i, df_all, segment_length, overlap):
    results = []
    
    filepath = df_all['filepath'][i] # Audio path
    dataset = df_all['dataset'][i] # Dataset name
    filename = df_all['filename'][i]
    
    label = df_all['label'][i]
    age = df_all['age'][i]
    gender = df_all['gender'][i]
    status = df_all['status'][i]
    
    try:
#     if True:
        (y, sr) = librosa.load(filepath, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        if duration == 0:
            return results

        segments = split_audio(y, sr, segment_length=segment_length, overlap=overlap)
        
        counter = 0
        for segment in segments:
            duration_segment = librosa.get_duration(y=segment, sr=sr)
            result_row = extract_features(segment, sr)

            mean = np.mean(np.abs(segment))
            
            result_row = [
                dataset, filepath, filename,
                age, gender, label, status, duration,
                duration_segment, sr, mean] + result_row
    
            results.append(result_row)

    except Exception as error:
        # print(error)
        pass
            
    return results

#################################################################################
def extract_features_CNN(segment, sr, segment_length):
    segment = mean_variance_normalize(segment) # Normalize

    if segment_length < 1:
        log_mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128, hop_length=512)
    elif segment_length == 1:
        log_mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128, hop_length=1024)
    elif segment_length == 5:
        log_mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128, hop_length=4096)
    elif segment_length == 10:
        log_mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128, hop_length=8192)
        
    segment_shape = log_mel_spec.shape
    segment_output = list(log_mel_spec.flatten())
    # print(segment_shape)

    return segment_output, segment_shape

#################################################################################
# Function to process each row and extract features
#################################################################################
def process_CNN_row(i, df_all, segment_length, overlap):
    results = []
    
    filepath = df_all['filepath'][i] # Audio path
    dataset = df_all['dataset'][i] # Dataset name
    filename = df_all['filename'][i]
    
    label = df_all['label'][i]
    age = df_all['age'][i]
    gender = df_all['gender'][i]
    status = df_all['status'][i]
    
    try:
    # if True:
        (y, sr) = librosa.load(filepath, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        if duration == 0:
            return results, None

        segments = split_audio(y, sr, segment_length=segment_length, overlap=overlap)
        for segment in segments:
            segment = pad_array(segment, segment_length*sr)
            duration_segment = librosa.get_duration(y=segment, sr=sr)
            segment_output, segment_shape = extract_features_CNN(segment, sr, segment_length)

            mean = np.mean(np.abs(segment))
            
            result_row = [
                dataset, filepath, filename,
                age, gender, label, status, duration,
                duration_segment,
                sr, mean, segment_shape,
            ] + segment_output

            results.append(result_row)

    except Exception as error:
        # print(error)
        pass
            
    return results, len(segment_output)