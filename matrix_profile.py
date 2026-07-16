import stumpy
import numpy as np
from kneed import KneeLocator
from scipy.signal import find_peaks, resample

def detect_anomalies_via_matrix_profile(time_series, target_cycle_length):
    peaks, _ = find_peaks(time_series)
    
    resampled_cycles = []
    mapped_indices = []
    
    for i in range(len(peaks) - 1):
        start_idx = peaks[i]
        end_idx = peaks[i + 1]
        
        cycle_data = time_series[start_idx:end_idx]
        resampled_cycles.append(resample(cycle_data, target_cycle_length))
        
        cycle_indices = np.linspace(start_idx, end_idx, target_cycle_length, endpoint=False)
        mapped_indices.append(cycle_indices)
        
    aligned_series = np.concatenate(resampled_cycles)
    aligned_indices = np.concatenate(mapped_indices)
    
    matrix_profile = stumpy.stump(aligned_series, m=target_cycle_length)
    
    distances = matrix_profile[:, 0].astype(float)
    original_indices = np.round(aligned_indices[:len(distances)]).astype(int)

    return matrix_profile, original_indices, distances

def generate_anomalous_reverse_sawtooth(total_cycles=20, min_cycle_length=100, max_cycle_length=150):
    time_series_segments = []
    
    anomaly_fast_index = total_cycles // 4
    anomaly_drop_index = total_cycles // 2
    anomaly_diff_index = (total_cycles * 3) // 4
    
    for cycle_index in range(total_cycles):
        current_length = np.random.randint(min_cycle_length, max_cycle_length)
        
        if cycle_index == anomaly_fast_index:
            fast_length = current_length // 4
            for _ in range(4):
                time_series_segments.append(np.linspace(1, -1, fast_length))
                
        elif cycle_index == anomaly_drop_index:
            segment = np.linspace(1, -1, current_length)
            drop_start = current_length // 2
            drop_end = drop_start + int(current_length * 0.1)
            segment[drop_start:drop_end] = -2
            time_series_segments.append(segment)
            
        elif cycle_index == anomaly_diff_index:
            sine_wave = np.sin(np.linspace(0, 2 * np.pi, current_length))
            time_series_segments.append(sine_wave)
            
        else:
            normal_segment = np.linspace(1, -1, current_length)
            time_series_segments.append(normal_segment)
            
    return np.concatenate(time_series_segments)

if __name__ == "__main__":
    arr = generate_anomalous_reverse_sawtooth()
    _, idx, dist = detect_anomalies_via_matrix_profile(arr, 100)
    sorted_idx = np.argsort(dist)[::-1]
    sotred_dist = dist[sorted_idx]
    kl = KneeLocator(np.arange(len(sotred_dist)), sotred_dist, curve="convex", direction="decreasing")
    cutoff_idx = kl.knee
    plt.figure(figsize=(24, 3))
    plt.plot(sotred_dist)
    plt.axvline(cutoff_idx, color="black")
    plt.show()

    fig, ax = plt.subplots(2, 1, figsize=(24, 6))
    ax[0].plot(arr)
    for i in sorted_idx[:cutoff_idx]:
        i = idx[i]
        ax[0].axvspan(i, i + 100, alpha=0.2, color='red')
    # ax[0].scatter(idx, arr[idx
    ax[1].plot(idx, dist)
    plt.show()
