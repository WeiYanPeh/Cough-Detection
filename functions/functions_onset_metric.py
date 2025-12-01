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

def compute_overlap(interval1, interval2):
    start1, end1 = interval1
    start2, end2 = interval2
    return max(0, min(end1, end2) - max(start1, start2))

def total_overlap_duration(predicted, ground_truth):
    overlap = 0
    for gt in ground_truth:
        for pred in predicted:
            overlap += compute_overlap(gt, pred)
    return overlap

def subtract_intervals(base, subtracting):
    result = []
    for b_start, b_end in base:
        current = [(b_start, b_end)]
        for s_start, s_end in subtracting:
            temp = []
            for c_start, c_end in current:
                if s_end <= c_start or s_start >= c_end:
                    temp.append((c_start, c_end))
                else:
                    if s_start > c_start:
                        temp.append((c_start, s_start))
                    if s_end < c_end:
                        temp.append((s_end, c_end))
            current = temp
        result.extend(current)
    return result

def total_duration(intervals):
    return sum(end - start for start, end in intervals)

def overlap_calculations(predicted_intervals, ground_truth_intervals):
    # Calculations
    overlap_duration = total_overlap_duration(predicted_intervals, ground_truth_intervals)
    gt_not_detected = subtract_intervals(ground_truth_intervals, predicted_intervals)
    gt_not_detected_duration = total_duration(gt_not_detected)
    pred_not_matched = subtract_intervals(predicted_intervals, ground_truth_intervals)
    pred_not_matched_duration = total_duration(pred_not_matched)
    
    # Output as JSON
    output = {
        "Total Overlap Duration (seconds)": round(overlap_duration, 2),
        "Total Cough Not Detected Duration (seconds)": round(gt_not_detected_duration, 2),
        "Total FP Duration (seconds)": round(pred_not_matched_duration, 2)
    }
    
    return output

# Example intervals
# predicted_intervals = [[1, 4], [8, 11], [40, 45], [46, 50]]
# ground_truth_intervals = [[2, 5], [9, 12], [39, 44], [50, 54]]

# metrics_overlap = overlap_calculations(predicted_intervals, ground_truth_intervals)
# for key, value in metrics_overlap.items():
#     print(f"{key}: {value}")


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

def interval_intersection(intervals1, intervals2):
    """
    Compute the intersection of two lists of intervals.

    Parameters:
        intervals1 (list of [start, end]): First list of intervals.
        intervals2 (list of [start, end]): Second list of intervals.

    Returns:
        list of [start, end]: Intervals where the two lists overlap.
    """
    i, j = 0, 0
    result = []

    while i < len(intervals1) and j < len(intervals2):
        a_start, a_end = intervals1[i]
        b_start, b_end = intervals2[j]

        # Find overlap
        start = max(a_start, b_start)
        end = min(a_end, b_end)

        if start < end:
            result.append([start, end])

        # Move to next interval
        if a_end < b_end:
            i += 1
        else:
            j += 1

    return result

# Example usage
# label_onset_inv_interval = [[0.2, 0.6826757369614512]]
# label_pred_inv_interval = [[0, 0.6826757369614512]]

# intersection = interval_intersection(label_onset_inv_interval, label_pred_inv_interval)
# print("Intersection:", intersection)

def total_interval_duration(intervals):
    """
    Calculate the total duration covered by a list of intervals.

    Parameters:
        intervals (list of [start, end]): List of intervals.

    Returns:
        float: Total duration covered by all intervals.
    """
    return sum(end - start for start, end in intervals)

# Example usage
# intervals = [[0.2, 0.6], [1.0, 2.5]]
# total_duration = total_interval_duration(intervals)
# print("Total duration:", total_duration)


def compute_overlap(interval1, interval2):
    """Compute the overlap length between two intervals."""
    start1, end1 = interval1
    start2, end2 = interval2
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    return max(0, intersection_end - intersection_start)

def evaluate_intervals(predicted, ground_truth, overlap_threshold=0.1):
    """Evaluate detection performance with collective overlap logic and detailed reporting."""
    true_positives = 0
    matched_gt = set()
    tp_contributors = []
    fp_intervals = []
    matched_gt_intervals = []
    unmatched_gt_intervals = []

    for i, gt in enumerate(ground_truth):
        gt_length = gt[1] - gt[0]
        overlapping_preds = [pred for pred in predicted if compute_overlap(pred, gt) > 0]
        total_overlap = sum(compute_overlap(pred, gt) for pred in overlapping_preds)
        overlap_ratio = total_overlap / gt_length if gt_length > 0 else 0
        if overlap_ratio >= overlap_threshold:
            true_positives += 1
            matched_gt.add(i)
            tp_contributors.extend(overlapping_preds)
            matched_gt_intervals.append(gt)

    # Identify false positives: predicted intervals not contributing to any TP
    fp_intervals = [pred for pred in predicted if pred not in tp_contributors]

    # Identify unmatched ground truth intervals
    unmatched_gt_intervals = [gt for i, gt in enumerate(ground_truth) if i not in matched_gt]

    false_positives = len(fp_intervals)
    false_negatives = len(unmatched_gt_intervals)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "TP": true_positives,
        "FP": false_positives,
        "FN": false_negatives,
        "PRE": precision,
        "REC": recall,
        "F1": f1_score,
        "TP Contributors": tp_contributors,
        "False Positive Intervals": fp_intervals,
        "Matched Ground Truth Intervals": matched_gt_intervals,
        "Unmatched Ground Truth Intervals": unmatched_gt_intervals
    }
    
def remove_isolated_detection(sequence, max_length_sequence=1):
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


def remove_mean_threshold(label_pred, list_threshold_mean):
    n = len(label_pred)
    for i in range(n):
        if list_threshold_mean[i] == 0:
            label_pred[i] = 0
    return label_pred

def safe_divide(numerator, denominator):
    """
    Safely divide two numbers, returning 0 if denominator is zero or negative.
    """
    if denominator <= 0:
        return 0
    return round(numerator / denominator, 3)

# Example usage
# predicted_intervals = [[1, 4], [8, 11], [40, 45], [46, 50]]
# ground_truth_intervals = [[2, 5], [9, 12], [39, 44], [50, 54]]

# Example usage
# predicted_intervals = [[1, 5], [15, 20]]
# ground_truth_intervals = [[0, 20]]
# threshold_overlap = 0.1  # 10% overlap required

# metrics = evaluate_intervals(predicted_intervals, ground_truth_intervals, overlap_threshold=threshold_overlap)
# for key, value in metrics.items():
#     print(f"{key}: {value}")





