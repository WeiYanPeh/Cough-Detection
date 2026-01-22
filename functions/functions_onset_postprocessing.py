import numpy as np



def split_time_series(series, n):
    """Split a time series into n equal intervals."""
    length = len(series)
    interval_size = length // n
    intervals = [series[i * interval_size : (i + 1) * interval_size] for i in range(n)]

    # Handle any remaining data (if length is not divisible by n)
    remainder = series[n * interval_size:]
    if remainder:
        intervals[-1].extend(remainder)

    return intervals



# Convert [[0.7, 2.3]] 
# into [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
def binary_to_intervals(binary_sequence, time_step=0.1):
    """Convert a binary sequence to intervals of consecutive 1s in seconds."""
    intervals = []
    start = None
    for i, val in enumerate(binary_sequence):
        if val == 1 and start is None:
            start = i
        elif val == 0 and start is not None:
            intervals.append([round(start * time_step, 1), round(i * time_step, 1)])
            start = None
    if start is not None:
        intervals.append([round(start * time_step, 1), round(len(binary_sequence) * time_step, 1)])
    return intervals

# Undo
# Convert [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# into [[0.7, 2.3]] 
def intervals_to_binary(intervals, original_length, time_step=0.1):
    """Convert time intervals back into a binary sequence and pad to original length."""
    binary_sequence = [0] * original_length
    for start, end in intervals:
        start_index = int(start / time_step)
        end_index = int(end / time_step)
        for i in range(start_index, min(end_index, original_length)):
            binary_sequence[i] = 1
    return binary_sequence

# Postprocessing
def merge_close_intervals(intervals, max_gap=0.2):
    """Merge intervals that are within max_gap seconds of each other."""
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for current in intervals[1:]:
        previous = merged[-1]
        if current[0] - previous[1] <= max_gap:
            merged[-1] = [previous[0], max(previous[1], current[1])]
        else:
            merged.append(current)

    return merged

# intervals = [[0.7, 0.8], [1.0, 1.2], [1.3, 2.3]]
# print(merge_close_intervals(intervals))


def inverse_intervals(labels, duration):
    """
    Compute the inverse of intervals within a fixed duration.

    Parameters:
        labels (list of [start, end]): List of intervals.
        duration (float): Total duration.

    Returns:
        list of [start, end]: Intervals not covered by the input labels.
    """
    inverse = []
    current = 0
    for start, end in labels:
        if current < start:
            inverse.append([current, start])
        current = end
    if current < duration:
        inverse.append([current, duration])
    return inverse

# labels = [[1, 3]]
# duration = 10
# print(inverse_intervals(labels, duration))
# Output: [[0, 1], [3, 10]]



#################################################################################
# Postprocessing
#################################################################################
def remove_isolated_detection(sequence, max_length_sequence=1):

    if max_length_sequence == 0:
        return sequence

    cleaned_sequence = sequence.copy()
    i = 0
    while i < len(sequence):
        if sequence[i] == 1:
            start = i
            while i < len(sequence) and sequence[i] == 1:
                i += 1
            end = i
            if start > 0 and end < len(sequence) and (end - start) <= max_length_sequence:
                for j in range(start, end):
                    cleaned_sequence[j] = 0
        else:
            i += 1
    return cleaned_sequence

def fill_short_gaps(sequence, threshold=1):
    '''
    Fill short gaps of 0s between 1s in a binary sequence.

    If a gap of 0s between two 1s is less than or equal to `threshold`, 
    the gap is replaced with 1s.

    Parameters:
        sequence (list): Binary list of 0s and 1s.
        threshold (int): Max gap length to fill (default is 1).

    Returns:
        list: Modified sequence with short gaps filled.

    Example:
        fill_short_gaps([1, 0, 1], threshold=1) → [1, 1, 1]
    '''
    filled_sequence = sequence.copy()
    i = 0
    while i < len(sequence):
        if sequence[i] == 1:
            start = i
            i += 1
            while i < len(sequence) and sequence[i] == 0:
                i += 1
            end = i
            if end < len(sequence) and (end - start - 1) <= threshold:
                for j in range(start + 1, end):
                    filled_sequence[j] = 1
        else:
            i += 1
    return filled_sequence


def remove_amplitude_threshold(label_pred, list_amplitude_mean):
    n = len(label_pred)
    for i in range(n):
        if list_amplitude_mean[i] == 0:
            label_pred[i] = 0
    return label_pred


def mean_filter_same_length(arr, n):
    # Ensure n is odd for symmetric filtering
    if n % 2 == 0:
        raise ValueError("Filter length n must be odd to maintain symmetry.")
    
    pad_width = n // 2
    padded_arr = np.pad(arr, pad_width, mode='edge')
    kernel = np.ones(n) / n
    filtered = np.convolve(padded_arr, kernel, mode='valid')
    
    # Keep original value if the mean is lower
    result = np.where(filtered < arr, arr, filtered)
    
    return result

# # Example usage
# arr = np.array([1, 2, 3, 4, 5])
# n = 3
# result = mean_filter_same_length(arr, n)
# print(result)  # Output: array with same length as `arr


