import numpy as np
from sklearn.linear_model import Ridge
from kneed import KneeLocator

def detect_cycle_duration_anomalies(cycle_info):
    if not cycle_info:
        return cycle_info
        
    durations = np.array([c['duration'] for c in cycle_info])
    sorted_durations = np.sort(durations)
    
    kneedle_dur = KneeLocator(
        range(len(sorted_durations)), 
        sorted_durations, 
        curve='concave', 
        direction='increasing'
    )
    
    if kneedle_dur.knee_y is not None:
        dur_threshold = kneedle_dur.knee_y
    else:
        dur_threshold = np.median(durations) * 0.5
        
    for info in cycle_info:
        info['is_freq_makeup'] = info['duration'] < dur_threshold
        
    return cycle_info

def detect_shape_anomalies_ridge(data, cycle_info, train_n=500, window_k=10):
    if len(data) <= train_n:
        train_n = len(data) // 2
        
    X_train = []
    y_train = []
    for i in range(train_n - window_k):
        X_train.append(data[i:i+window_k])
        y_train.append(data[i+window_k])
        
    model = Ridge()
    model.fit(X_train, y_train)
    
    X_all = []
    for i in range(len(data) - window_k):
        X_all.append(data[i:i+window_k])
        
    if not X_all:
        for info in cycle_info:
            info['shape_anomaly'] = False
            info['anomaly_score'] = 0.0
        return cycle_info
        
    preds = model.predict(X_all)
    errors = (data[window_k:] - preds) ** 2
    
    full_errors = np.zeros(len(data))
    full_errors[window_k:] = errors
    
    scores = []
    for info in cycle_info:
        start = info['start_idx']
        end = info['end_idx']
        
        if end > start and start >= window_k:
            score = np.max(full_errors[start:end])
        else:
            score = 0.0
            
        info['anomaly_score'] = float(score)
        scores.append(score)
        
    sorted_scores = np.sort(scores)
    kneedle_score = KneeLocator(
        range(len(sorted_scores)), 
        sorted_scores, 
        curve='convex', 
        direction='increasing'
    )
    
    if kneedle_score.knee_y is not None:
        score_threshold = max(kneedle_score.knee_y, 0.0)
    else:
        score_threshold = 0.0
        
    for info in cycle_info:
        info['shape_anomaly'] = info['anomaly_score'] > score_threshold
        
    return cycle_info

def detect_pcw_anomalies_kneed(data, rise_slope_threshold=1.0, fixed_length=100, train_n=500, window_k=10):
    diffs = np.diff(data)
    rise_idx = np.where(diffs > rise_slope_threshold)[0]
    
    if len(rise_idx) == 0:
        return [], np.array([])
        
    breaks = np.where(np.diff(rise_idx) > 1)[0]
    start_of_rises = rise_idx[np.r_[0, breaks + 1]]
    end_of_rises = rise_idx[np.r_[breaks, len(rise_idx) - 1]] + 1
    
    cycles = []
    cycle_info = []
    
    for i in range(len(end_of_rises) - 1):
        start_drop = end_of_rises[i]
        end_drop = start_of_rises[i+1]
        cycle_data = data[start_drop:end_drop]
        duration = end_drop - start_drop
        
        if duration < 5:
            continue
            
        x_old = np.linspace(0, 1, duration)
        x_new = np.linspace(0, 1, fixed_length)
        cycle_norm = np.interp(x_new, x_old, cycle_data)
        
        cycles.append(cycle_norm)
        cycle_info.append({
            'cycle_index': i + 1,
            'start_idx': start_drop,
            'end_idx': end_drop,
            'duration': duration
        })
        
    if not cycles:
        return cycle_info, np.array([])
        
    cycles_matrix = np.array(cycles)
    
    cycle_info = detect_cycle_duration_anomalies(cycle_info)
    cycle_info = detect_shape_anomalies_ridge(data, cycle_info, train_n=train_n, window_k=window_k)
    
    for info in cycle_info:
        if info['is_freq_makeup']:
            info['case'] = 'Case 2 (Frequent Makeup / Fast Leak)'
        elif info['shape_anomaly']:
            info['case'] = 'Case 1 or 3 (Shape Distortion / Sudden Drop)'
        else:
            info['case'] = 'Normal'
            
    return cycle_info, cycles_matrix
