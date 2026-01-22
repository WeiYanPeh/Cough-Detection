import numpy as np

#################################################################################
def compute_overlap(interval1, interval2):
    """
    Compute the overlap length between two intervals.

    Parameters:
        interval1 (list or tuple): [start1, end1] of the first interval.
        interval2 (list or tuple): [start2, end2] of the second interval.

    Returns:
        float: Length of the overlapping region. Returns 0 if no overlap.
    """
    
    start1, end1 = interval1
    start2, end2 = interval2
    return max(0, min(end1, end2) - max(start1, start2))


#################################################################################
def total_overlap_duration(predicted, ground_truth):
    """
    Compute the total overlap duration between predicted and ground truth intervals.

    For each ground truth interval, sums the overlapping durations with all predicted intervals.

    Parameters:
        predicted (list of lists/tuples): List of predicted [start, end] intervals.
        ground_truth (list of lists/tuples): List of ground truth [start, end] intervals.

    Returns:
        float: Total overlapping duration in the same units as the intervals.
    """
    
    overlap = 0
    for gt in ground_truth:
        for pred in predicted:
            overlap += compute_overlap(gt, pred)
    return overlap


#################################################################################
def subtract_intervals(base, subtracting):
    """
    Subtract a list of intervals from a base list of intervals.

    For each interval in `base`, any overlapping portions with intervals
    in `subtracting` are removed. The result is a list of non-overlapping
    intervals.

    Parameters:
        base (list of tuples): List of base intervals [(start, end), ...].
        subtracting (list of tuples): List of intervals to subtract.

    Returns:
        list of tuples: List of intervals after subtraction.
    """
    
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


#################################################################################
def total_duration(intervals):
    """
    Compute the total duration covered by a list of intervals.

    Parameters:
        intervals (list of tuples/lists): List of intervals [(start, end), ...].

    Returns:
        float: Sum of durations of all intervals (end - start).
    """
    return sum(end - start for start, end in intervals)


#################################################################################
def overlap_calculations(predicted_intervals, ground_truth_intervals):
    """
    Compute overlap metrics between predicted and ground truth intervals.

    This function calculates:
      - Total overlapping duration
      - Duration of ground truth events not detected (missed)
      - Duration of predicted events not matched (false positives)

    Parameters:
        predicted_intervals (list of tuples/lists): Predicted [start, end] intervals.
        ground_truth_intervals (list of tuples/lists): Ground truth [start, end] intervals.

    Returns:
        dict: Metrics rounded to 2 decimal places:
            {
                "Total Overlap Duration (seconds)": float,
                "Total Cough Not Detected Duration (seconds)": float,
                "Total FP Duration (seconds)": float
            }
    """
    
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


#################################################################################
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


#################################################################################
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



#################################################################################
# Evaluation
#################################################################################
def evaluate_intervals_event_based(predicted, ground_truth, duration, overlap_threshold=0.1):
    """
    Evaluate event-based detection performance between predicted and ground truth intervals.

    Each ground truth event is considered detected (TP) if the total overlap with predicted
    events exceeds the overlap_threshold fraction of the ground truth duration.

    Parameters:
        predicted (list of [start, end]): Predicted intervals.
        ground_truth (list of [start, end]): Ground truth intervals.
        duration (float): Total duration of the recording (seconds) for FAR calculation.
        overlap_threshold (float): Minimum fraction of GT overlap to count as a true positive.

    Returns:
        dict: Detailed event-based metrics and intervals:
            - TP_e: Number of true positives
            - FP_e: Number of false positives
            - FN_e: Number of false negatives
            - PRE_e: Precision
            - REC_e: Recall
            - F1_e: F1 score
            - FAR_e: False alarm rate (per second)
            - FARh_e: False alarm rate (per hour)
            - Event - TP Contributors: List of predicted intervals contributing to TPs
            - Event - False Positive Intervals: Predicted intervals not contributing to TPs
            - Event - Matched Ground Truth Intervals: GT intervals matched (TP)
            - Event - Unmatched Ground Truth Intervals: GT intervals not matched (FN)
    """
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

    FAR = round(false_positives / duration, 5)
    FARh = round(false_positives / duration * 3600, 5)
    
    return {
        "TP_e": true_positives,
        "FP_e": false_positives,
        "FN_e": false_negatives,
        "PRE_e": round(precision, 3),
        "REC_e": round(recall, 3),
        "F1_e": round(f1_score, 3),

        'FAR_e': FAR,
        'FARh_e': FARh,
        
        "Event - TP Contributors": tp_contributors, # TP from detector
        "Event - False Positive Intervals": fp_intervals, # FP
        "Event - Matched Ground Truth Intervals": matched_gt_intervals, # TP
        "Event - Unmatched Ground Truth Intervals": unmatched_gt_intervals # FN <- Check
    }


#################################################################################
def evaluate_intervals_duration_based(predicted, ground_truth, duration):
    """
    Evaluate interval-based detection performance using duration-based metrics.

    This method computes how much of the total duration is correctly or incorrectly
    predicted as events (e.g., coughs) and non-events, giving TP, TN, FP, FN, and
    derived metrics such as sensitivity, specificity, precision, F1 score, and
    false alarm rates.

    Parameters:
        predicted (list of [start, end]): Predicted event intervals.
        ground_truth (list of [start, end]): Ground truth event intervals.
        duration (float): Total recording duration in seconds.

    Returns:
        dict: Duration-based metrics and interval information, including:
            - TP_d, FP_d, FN_d, TN_d: True/False Positive/Negative durations
            - SEN_d: Sensitivity (Recall)
            - SPE_d: Specificity
            - PRE_d: Precision
            - F1_d: F1 score
            - FAR_d: False Alarm Rate (per second)
            - FARh_d: False Alarm Rate per hour
            - Detailed interval intersections for analysis
    """
    
    label_pred_interval = predicted
    label_onset_interval = ground_truth
    
    # Get non-cough intervals
    label_onset_inv_interval = inverse_intervals(label_onset_interval, duration)
    label_pred_inv_interval = inverse_intervals(label_pred_interval, duration)

    # Get cough and non-cough intersects
    label_cough_interval_intersect = interval_intersection(label_onset_interval, label_pred_interval)
    label_non_cough_interval_intersect = interval_intersection(label_onset_inv_interval, label_pred_inv_interval)

    # Get intersection duration
    total_cough_intersect_duration = round(total_interval_duration(label_cough_interval_intersect), 1) # TP
    total_non_cough_intersect_duration = round(total_interval_duration(label_non_cough_interval_intersect), 1) # TN

    total_non_cough_duration = round(total_interval_duration(label_onset_inv_interval), 1)
    total_cough_duration = round(total_interval_duration(label_onset_interval), 1)
    total_pred_duration = round(total_interval_duration(label_pred_interval), 1)

    FP = total_pred_duration - total_cough_intersect_duration
    FN = total_cough_duration - total_cough_intersect_duration
    
    SENd = round(safe_divide(total_cough_intersect_duration, total_cough_duration), 3)
    SPEd = round(safe_divide(total_non_cough_intersect_duration, total_non_cough_duration), 3)
    PREd = round(safe_divide(total_cough_intersect_duration, total_pred_duration), 3)
    F1d = round(2 * safe_divide((PREd * SENd), (PREd + SENd)), 3)

    FAR = round(FP / duration, 5)
    FARh = round(FP / duration * 3600, 5)
    
    
    return {        
        'label_onset_inv_interval': label_onset_inv_interval,
        'label_pred_inv_interval': label_pred_inv_interval,
        'label_cough_interval_intersect': label_cough_interval_intersect,
        'label_non_cough_interval_intersect': label_non_cough_interval_intersect,
        'total_cough_intersect_duration': total_cough_intersect_duration,
        'total_non_cough_intersect_duration': total_non_cough_intersect_duration,
        'total_cough_duration': total_cough_duration,
        'total_non_cough_duration': total_non_cough_duration,
        'total_pred_duration': total_pred_duration,

        'TP_d': total_cough_intersect_duration,
        'FP_d': FP,
        'FN_d': FN,
        'TN_d': total_non_cough_intersect_duration,
        
        'SEN_d': SENd,
        'SPE_d': SPEd,
        'PRE_d': PREd,
        'F1_d': F1d,

        'FAR_d': FAR,
        'FARh_d': FARh,
    }

#################################################################################
# Compute average and sum metrics
#################################################################################
def get_average_metrics(results_all, model_name, segment_length):
    """
    Compute average performance metrics for a model across multiple recordings or segments.

    This function filters out outlier results with excessively high false alarm rates,
    then calculates mean metrics for both duration-based (d) and event-based (e) evaluation.

    Parameters:
        results_all (pd.DataFrame): DataFrame containing evaluation results with columns such as
                                    'SEN_d', 'SPE_d', 'PRE_d', 'FAR_d', 'FARh_d',
                                    'PRE_e', 'REC_e', 'FAR_e', 'FARh_e', and 'label'.
        model_name (str): Name of the model being evaluated.
        segment_length (float/int): Length of the segment/window for evaluation.

    Returns:
        dict: Dictionary with averaged metrics and metadata:
              - 'model': model name
              - 'window_length': segment length
              - 'type': 'avg'
              - Duration-based metrics: SEN_d, SPE_d, PRE_d, F1_d, FAR_d, FARh_d
              - Event-based metrics: PRE_e, REC_e, F1_e, FAR_e, FARh_e
    """

    results_all = results_all[results_all['FAR_d'] <= 0.2].reset_index(drop=True)
    results_all = results_all[results_all['FAR_e'] <= 0.2].reset_index(drop=True)
    
    results_dict_avg = {   
        'model': model_name,
        'window_length': segment_length,
        'type': 'avg'
    }
    
    for item in [
        'SEN_d', 'SPE_d', 'PRE_d', 'FAR_d', 'FARh_d', 
        'PRE_e', 'REC_e', 'FAR_e', 'FARh_e'
        ]:
        if item in ['SEN_d', 'PRE_d', 'PRE_e', 'REC_e']:
            results_dict_avg[item] = round(np.mean(results_all[results_all['label'] == 1][item]), 3)
        elif item in ['FAR_e', 'FAR_e', 'FAR_e', 'FARh_e']:
            results_dict_avg[item] = round(np.mean(results_all[item]), 5)
        else: #  'SPE_d',
            results_dict_avg[item] = round(np.mean(results_all[item]), 3)
    
    results_dict_avg['F1_d'] = round(2 * results_dict_avg['SEN_d'] * results_dict_avg['PRE_d'] / (results_dict_avg['SEN_d'] + results_dict_avg['PRE_d']), 3)
    results_dict_avg['F1_e'] = round(2 * results_dict_avg['REC_e'] * results_dict_avg['PRE_e'] / (results_dict_avg['REC_e'] + results_dict_avg['PRE_e']), 3)
    return results_dict_avg


#################################################################################
def get_sum_metrics(results_all, model_name, segment_length):
    """
    Compute summed/aggregated performance metrics across multiple recordings or segments.

    Unlike average metrics, this method aggregates totals first and then computes
    derived metrics from the summed values. This is useful for weighted evaluation
    over recordings of different durations.

    Parameters:
        results_all (pd.DataFrame): DataFrame containing evaluation results with columns such as
                                    'TP_d', 'FP_e', 'total_cough_duration', etc.
        model_name (str): Name of the model being evaluated.
        segment_length (float/int): Length of the segment/window for evaluation.

    Returns:
        dict: Dictionary with summed metrics and metadata:
              - 'model': model name
              - 'window_length': segment length
              - 'type': 'sum'
              - Duration-based metrics: SEN_d, SPE_d, PRE_d, F1_d, FAR_d, FARh_d
              - Event-based metrics: PRE_e, REC_e, F1_e, FAR_e, FARh_e
    """
    
    results_all = results_all[results_all['FAR_d'] <= 0.2].reset_index(drop=True)
    results_all = results_all[results_all['FAR_e'] <= 0.2].reset_index(drop=True)
    
    results_dict_sum = {
        'model': model_name,
        'window_length': segment_length,
        'type': 'sum'
    }
    
    duration_total = np.sum(results_all['duration'])
    
    # Duration based
    TP_d = np.sum(results_all['TP_d'])
    
    results_dict_sum['SEN_d'] = round(np.sum(results_all['total_cough_intersect_duration']) / np.sum(results_all['total_cough_duration']), 3)
    results_dict_sum['SPE_d'] = round(np.sum(results_all['total_non_cough_intersect_duration']) / np.sum(results_all['total_non_cough_duration']), 3)
    results_dict_sum['PRE_d'] = round(np.sum(results_all['total_cough_intersect_duration']) / np.sum(results_all['total_pred_duration']), 3)
    results_dict_sum['F1_d'] = round(2 * (results_dict_sum['PRE_d'] * results_dict_sum['SEN_d']) / (results_dict_sum['PRE_d'] + results_dict_sum['SEN_d']), 3)
    
    results_dict_sum['FAR_d'] = round(TP_d / duration_total, 5)
    results_dict_sum['FARh_d'] = round(TP_d / duration_total * 3600, 5)
    
    # Event based
    TP_e = np.sum(results_all['TP_e'])
    FP_e = np.sum(results_all['FP_e'])
    FN_e = np.sum(results_all['FN_e'])
    
    results_dict_sum['PRE_e'] = round(TP_e / (TP_e + FP_e), 3)
    results_dict_sum['REC_e'] = round(TP_e / (TP_e + FN_e), 3)
    results_dict_sum['F1_e'] = round(2 * (results_dict_sum['PRE_e'] * results_dict_sum['REC_e']) / (results_dict_sum['PRE_e'] + results_dict_sum['REC_e']), 3)
    
    results_dict_sum['FAR_e'] = round(TP_e / duration_total, 5)
    results_dict_sum['FARh_e'] = round(TP_e / duration_total * 3600, 5)
    return results_dict_sum


#################################################################################
def safe_divide(numerator, denominator):
    """
    Safely divide two numbers, avoiding division by zero or negative denominators.

    This is useful for metric calculations like precision, recall, or F1 score,
    where the denominator can sometimes be zero due to no predicted or actual events.

    Parameters:
        numerator (float): The numerator of the division.
        denominator (float): The denominator of the division.

    Returns:
        float: The division result rounded to 3 decimal places.
               Returns 0 if the denominator is zero or negative.
    """
    
    if denominator <= 0:
        return 0
    
    return round(numerator / denominator, 3)



