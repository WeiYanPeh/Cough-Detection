import warnings
warnings.filterwarnings('ignore')

import numpy as np
from pydub import AudioSegment # sudo apt install ffmpeg

#################################################################################
def get_cough(y, segment_length, fs):
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

    # Get sliding window frame    
    frame_length = int(fs*segment_length) # 2048
    # hop_length = int(frame_length / 4) # 512
    hop_length = int(frame_length / 4) # 512
    
    # energy = np.array([
    #     np.sum(np.abs(y_norm[i:i+frame_length])**2)
    #     for i in range(0, len(y_norm), hop_length)
    # ])

    energy = np.array([
        np.mean(np.abs(y_norm[i:i+frame_length]))
        for i in range(0, len(y_norm), hop_length)
    ])

    # max_energy = np.max(energy)
    # max_energy = np.median(energy)
    max_energy = np.percentile(energy, 90)
    
    threshold_cough = 0.2 * max_energy
    # threshold_low = 0.05 * max_energy

    # print(f'High Energy: {threshold_cough}')
    # print(f'Low Energy : {threshold_low}')
    
    cough_frames = np.where(energy > threshold_cough)[0]
    silent_frames = np.where(energy <= threshold_cough)[0]
    # low_frames = np.where((energy > threshold_low) & (energy <= threshold_cough))[0]
    # silent_frames = np.where(energy <= threshold_low)[0]

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
    If stereo, it will be converted to mono by averaging the channels.
    """
    samples = np.array(audio_segment.get_array_of_samples())
    
    # If stereo, average the two channels
    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2))
        samples = samples.mean(axis=1)
    
    return samples.astype(np.float32)