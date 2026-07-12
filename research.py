import numpy as np
def generate_variant_signal(period_range, seq_len):
    """產生反向鋸齒波，並且有 1/2 的機率會有上跟下的shfit在該波型內發生"""
    signal = []
    while len(signal) < seq_len:
        period = np.random.randint(period_range[0], period_range[1] + 1)
        wave = np.linspace(1, 0, period, endpoint=False)

        num_shifts = np.random.randint(0, 2)
        for _ in range(num_shifts):
            start_idx = np.random.randint(1, period)
            magnitude = np.random.uniform(0.15, 0.4)
            direction = np.random.choice([-1, 1])
            wave[start_idx :] += (magnitude * direction)
        signal.extend(wave)

    result = np.array(signal[:seq_len]) + np.random.normal(0, 0.05, seq_len)
    return result.reshape(seq_len, )


arr = generate_variant_signal((80, 100), 1000)

import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
diff_arr = np.diff(arr)
peaks, _ = find_peaks(diff_arr, height=0.5, distance=50)
plt.figure(figsize=(24, 3))
plt.plot(arr,color="blue")
for i in peaks:
    plt.axvline(x=i,color="red")
plt.grid(color="gray", linestyle="--", alpha=.4)
plt.show()

import numpy as np
from fastdtw import fastdtw

def keogh_derivative(q):
    """
    依據 Keogh & Pazzani (2001) 論文實現的 DDTW 專用導數估算
    """
    q = np.array(q)
    n = len(q)
    if n < 3:
        return np.zeros_like(q)

    deriv = np.zeros(n)
    # 向量化計算中間點的導數
    deriv[1:-1] = ((q[1:-1] - q[:-2]) + ((q[2:] - q[:-2]) / 2.0)) / 2.0

    # 邊界條件處理：端點通常直接設為與相鄰點相同
    deriv[0] = deriv[1]
    deriv[-1] = deriv[-2]
    return deriv

def ddtw_distance(x, y, radius=5):
    """
    DDTW 封裝函數
    """
    # 1. 轉為導數序列
    dx = keogh_derivative(x)
    dy = keogh_derivative(y)

    # 2. 丟入現有的標準 fastdtw 計算
    distance, path = fastdtw(dx, dy, radius=radius, dist=lambda a, b: np.abs(a - b))
    return distance, path

# !pip install fastdtw
from fastdtw import fastdtw
from scipy.stats import linregress, pearsonr
cycle_intervals = []
if peaks[0] > 0:
    cycle_intervals.append((0, peaks[0]))
for i in range(len(peaks) - 1):
    cycle_intervals.append((peaks[i], peaks[i+1]))
if peaks[-1] < len(arr):
    cycle_intervals.append((peaks[-1], len(arr)))

scores = []
midpoints = []
for (start, end) in cycle_intervals:
  cycle_signal = arr[start:end]
  cycle_signal -= np.min(cycle_signal)
  cycle_signal /= np.max(cycle_signal)
  midpoint = (start + end) / 2.0
  midpoints.append(midpoint)
  ideal_template = np.linspace(1, 0, end-start+1)
  # distance, path = fastdtw(cycle_signal, ideal_template, dist=lambda x, y: np.abs(x - y))
  distance, path = ddtw_distance(cycle_signal, ideal_template)
  _path = np.array(path)
  diff = np.mean(np.abs(_path[:, 0] - _path[:, 1]))
  coef = np.polyfit(_path[:, 0], _path[:, 1], 1)
  slope, intercept, r_value, _, _ = linregress(_path[:, 0], _path[:, 1])
  r_value = 1 - r_value
  scores.append((distance / (end-start+1))*r_value + diff) 

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 6), sharex=True)
ax1.plot(arr, color="blue")
for p in peaks:
    ax1.axvline(x=p, color="red")
ax1.set_title("Original Variant Signal with Peak Boundaries", fontsize=14)
ax1.grid(color="gray", linestyle="--", alpha=0.4)
ax2.plot(midpoints[:-1], scores[:-1], color="orange", marker="o", linestyle="-", linewidth=2, markersize=8)
for p in peaks:
    ax2.axvline(x=p, color="red", alpha=0.2)
ax2.set_title("DTW Scores per Segment (Higher Score = More Distorted / Abnormal)", fontsize=14)
ax2.set_ylabel("Normalized DTW Distance", fontsize=12)
ax2.set_xlabel("Time Step (Index)", fontsize=12)
ax2.grid(color="gray", linestyle="--", alpha=0.4)
