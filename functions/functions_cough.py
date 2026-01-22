import warnings
warnings.filterwarnings('ignore')

import numpy as np
from pydub import AudioSegment # sudo apt install ffmpeg

#################################################################################
def get_cough(y, segment_length, fs):
    """
    Detect potential cough events in an audio waveform based on amplitude/energy thresholding.

    Parameters:
        y (np.ndarray): 1D audio waveform (amplitude values).
        segment_length (float): Window length in seconds for short-time analysis.
        fs (int): Sampling frequency (Hz).

    Returns:
        cough_events (list of tuples): Start and end frame indices of detected coughs.
        silent_events (list of tuples): Start and end frame indices of silent frames.
        hop_length (int): Number of samples between consecutive frames.
        energy (np.ndarray): Energy values for each frame.
        threshold_cough (float): Energy threshold used for cough detection.
    """
    # -------------------------------
    # Step 1: Normalize
    # -------------------------------
    # Max scaling - amplify noise
    # y_norm = y / np.max(np.abs(y))  # Normalise amplitude

    # Clip scaling
    # y_clipped = np.clip(y, -1.0, 1.0)
    # y_norm = y_clipped / (np.max(np.abs(y_clipped)) + 1e-6)

    # RMS Scaling - amplify noise
    # rms = np.sqrt(np.mean(y**2))
    # if rms > 0:
    #     y_norm = y / rms
    # else:
    #     y_norm = y  # Leave unchanged if silent

    # Percentile scaling - If you see cough as an anomal event
    scale = np.percentile(np.abs(y), 95)
    # print(f'Scale: {scale}')
    if scale > 0:
        y_norm = y / scale
    else:
        y_norm = y

    # -------------------------------
    # Step 2: Frame the signal
    # ------------------------------
    # Get sliding window frame    
    frame_length = int(fs*segment_length) # 2048
    # hop_length = int(frame_length / 4) # 512
    hop_length = int(frame_length / 4) # 512
    
    # -------------------------------
    # Step 3: Compute frame-level energy
    # -------------------------------
    # energy = np.array([
    #     np.sum(np.abs(y_norm[i:i+frame_length])**2)
    #     for i in range(0, len(y_norm), hop_length)
    # ])

    energy = np.array([
        np.mean(np.abs(y_norm[i:i+frame_length]))
        for i in range(0, len(y_norm), hop_length)
    ])

    # -------------------------------
    # Step 4: Determine cough threshold
    # -------------------------------
    # max_energy = np.max(energy)
    # max_energy = np.median(energy)
    max_energy = np.percentile(energy, 90)
    
    threshold_cough = 0.2 * max_energy
    # threshold_low = 0.05 * max_energy

    # print(f'High Energy: {threshold_cough}')
    # print(f'Low Energy : {threshold_low}')
    
    # -------------------------------
    # Step 5: Identify frames above/below threshold
    # -------------------------------
    cough_frames = np.where(energy > threshold_cough)[0]
    silent_frames = np.where(energy <= threshold_cough)[0]
    # low_frames = np.where((energy > threshold_low) & (energy <= threshold_cough))[0]
    # silent_frames = np.where(energy <= threshold_low)[0]

    # -------------------------------
    # Step 6: Group consecutive frames into events
    # -------------------------------
    def group_frames(frames):
        events = []
        if len(frames) > 0:
            start = frames[0]
            for i in range(1, len(frames)):
                if frames[i] > frames[i-1] + 1:
                    end = frames[i-1]
                    events.append((start, end))
                    start = frames[i]
            events.append((start, frames[-1]))
        return events
    
    return (
        group_frames(cough_frames),
        # group_frames(low_frames),
        group_frames(silent_frames),
        hop_length,
        energy,
        threshold_cough
    )

#################################################################################
def convert_events_to_seconds(events, segment_length, hop_length, sr):
    """
    Convert detected event indices into real-time seconds.

    Parameters:
        events (list of tuples/lists): Each element is [start_index, end_index] of detected event (in frame indices).
        segment_length (float): Length of the segment in seconds used in feature extraction.
        hop_length (int): Hop length used when generating frames (samples per step).
        sr (int): Sampling rate of the audio (samples per second).

    Returns:
        results (list of lists): Each element is [start_time_sec, end_time_sec] of event in seconds.
    """
    
    results = []
    for start, end in events:
        t_start = np.round(start * hop_length / sr, 1)
        t_end = np.round((end + 1) * hop_length / sr, 1)
        if t_start == t_end:
            t_end += segment_length
        results.append([t_start, t_end])
    return results

#################################################################################
def label_generator(time_windows, duration, segment_length):
    """
    Generate a binary label sequence for an audio file based on event time windows.

    Parameters:
        time_windows (list of lists/tuples): Each element is [start_time, end_time] in seconds
                                             representing when an event occurs.
        duration (float): Total duration of the audio in seconds.
        segment_length (float): Time step for labeling (resolution of the binary sequence, e.g., 0.1 s).

    Returns:
        tuple:
            - time_intervals (np.ndarray): Array of time points at which labels are generated.
            - binary_array (list): List of 0/1 labels where 1 indicates the presence of an event.
    """
    
    # Create time intervals of 0.1 from 0 to duration
    time_intervals = np.arange(0, duration, segment_length)
    
    # Generate the binary array
    binary_array = [
        1 if any(start <= t <= end for start, end in time_windows) else 0
        for t in time_intervals
    ]
    return time_intervals, binary_array

#################################################################################
def slice_audio(file_path, segment_length, new_sample_rate):
    """
    Slice a WAV audio file into fixed-length segments after resampling.

    Parameters:
        file_path (str): Path to the input WAV file.
        segment_length (float): Length of each segment in seconds.
        new_sample_rate (int): Desired sample rate in Hz for the audio.

    Returns:
        slices (list of AudioSegment): List of sliced audio segments.
    """
    
    audio = AudioSegment.from_wav(file_path)
    audio = audio.set_frame_rate(new_sample_rate)  # Change sample rate
    duration = len(audio)  # Duration in milliseconds
    slices = []

    for start in range(0, duration, int(segment_length * 1000)):
        end = min(start + segment_length * 1000, duration)
        slice = audio[start:end]
        slices.append(slice)

    return slices

#################################################################################
def audiosegment_to_amplitudes(audio_segment):
    """
    Convert a pydub AudioSegment to a NumPy array of amplitude values.

    Parameters:
        audio_segment (AudioSegment): A pydub AudioSegment object (mono or stereo).

    Returns:
        samples (np.ndarray): 1D NumPy array of float32 amplitudes.
                              Stereo audio is converted to mono by averaging channels.
    """
    
    samples = np.array(audio_segment.get_array_of_samples())
    
    # If stereo, average the two channels
    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2))
        samples = samples.mean(axis=1)
    
    return samples.astype(np.float32)