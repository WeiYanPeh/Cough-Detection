import warnings
warnings.filterwarnings('ignore')

import librosa
import scipy.stats
from scipy.stats import skew, kurtosis

import numpy as np


#################################################################################
def pad_array(array, target_length):
    """
    Pad a 1D NumPy array with zeros to reach a specified target length.

    Parameters:
        array (np.ndarray): Input 1D array to pad.
        target_length (int): Desired length after padding.

    Returns:
        np.ndarray: Zero-padded array of length `target_length`.
                    If `array` is already longer, it is returned unchanged.
    """
    return np.pad(array, (0, max(0, target_length - len(array))), 'constant')

#################################################################################
def split_audio(
        y,  # Signal
        sr, # Sample frequency
        segment_length=10.0, # Segment length 1s
        overlap=0 # Overlap 50%
    ):
    """
    Split a 1D audio signal into fixed-length segments with optional overlap.

    Parameters:
        y (np.ndarray): Input audio signal (1D array).
        sr (int): Sample rate of the audio signal.
        segment_length (float): Desired segment length in seconds (default 10s).
        overlap (float): Fraction of overlap between consecutive segments (0.0 means no overlap).

    Returns:
        list of np.ndarray: List containing segments of the original audio.
    """
    
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
    Normalize a 1D NumPy array representing an audio waveform using mean-variance scaling.

    This process ensures that the waveform has:
        - Mean = 0
        - Standard deviation = 1

    Parameters:
        audio_waveform (np.ndarray): 1D array of audio samples.

    Returns:
        np.ndarray: Normalized waveform with zero mean and unit variance.
                    If the input has zero variance, returns a zero array.
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
    """
    Extract a comprehensive set of features from a 1D audio segment.

    Features include:
      - Time-domain: mean, variance, std, max, min, RMS, skewness, kurtosis, median, range, IQR, ZCR, energy, RMSE
      - Frequency-domain: spectral centroid, bandwidth, contrast, flatness, rolloff, chroma
      - MFCCs: mean and std for each coefficient
      - Entropy of the segment

    Parameters:
        segment (np.ndarray): 1D audio signal segment
        sr (int): Sampling rate of the segment

    Returns:
        list: Feature vector representing the segment
    """
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
    """
    Process a single row of the DataFrame containing audio file information,
    split the audio into segments, extract features for each segment, and
    return a list of feature vectors.

    Parameters:
        i (int): Index of the row in the DataFrame.
        df_all (pd.DataFrame): DataFrame containing audio metadata and labels.
        segment_length (float): Length of each audio segment in seconds.
        overlap (float): Fractional overlap between consecutive segments.

    Returns:
        list: List of feature vectors (one per segment). Each feature vector
              includes metadata, segment info, and extracted features.
    """
    
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
    """
    Extract log-mel spectrogram features from an audio segment for CNN input.

    Parameters:
        segment (np.ndarray): 1D audio segment
        sr (int): Sampling rate of the segment
        segment_length (float): Length of the segment in seconds, used to adjust hop_length

    Returns:
        tuple:
            - segment_output (list): Flattened log-mel spectrogram (for ML input)
            - segment_shape (tuple): Shape of the original log-mel spectrogram (n_mels, time_frames)
    """
    
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

    return segment_output, segment_shape

#################################################################################
# Function to process each row and extract features
#################################################################################
def process_CNN_row(i, df_all, segment_length, overlap):
    """
    Process a single row of a DataFrame for CNN feature extraction.

    This function:
    - Loads the audio file.
    - Splits it into segments with optional overlap.
    - Pads each segment to a fixed length.
    - Extracts log-mel spectrogram features for CNN input.
    - Returns a list of feature vectors along with the feature length.

    Parameters:
        i (int): Index of the row in df_all.
        df_all (pd.DataFrame): DataFrame containing audio metadata and labels.
        segment_length (float): Length of each segment in seconds.
        overlap (float): Fractional overlap between consecutive segments.

    Returns:
        tuple:
            - results (list): List of feature vectors for each segment.
              Each vector contains metadata, segment info, shape info, and flattened log-mel spectrogram.
            - feature_length (int or None): Length of the flattened feature vector (log-mel spectrogram part).
    """
    
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