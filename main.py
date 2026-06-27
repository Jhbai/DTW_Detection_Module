# !pip install dtaidistance
# !pip install json
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtaidistance.subsequence.dtw import subsequence_alignment
from scipy.signal import savgol_filter

def dtw_anomaly(query, pattern):
  # 套件處理
  from dtaidistance.subsequence.dtw import subsequence_alignment
  from sklearn.linear_model import LinearRegression
  from scipy.signal import savgol_filter
  import numpy as np

  # 時間序列整理
  query = np.asarray(query, dtype=np.double)
  pattern = np.asarray(pattern, dtype=np.double)

  # 序列平滑化
  query = savgol_filter(query, window_length=101, polyorder=3)
  pattern = savgol_filter(pattern, window_length=101, polyorder=3)

  # DTW 計算
  sa = subsequence_alignment(query, pattern, use_c = True)
  match = sa.best_match()
  st = match.segment[0]
  ed = st + query.shape[0]

  # 計算異常值
  dist1 = np.sum(np.abs(query - pattern[st:ed])) # 逐點差異
  dist2 = np.abs(np.max(query) - np.max(pattern[st:ed])) + np.abs(np.min(query) - np.min(pattern[st:ed])) # 極值差異
  dist3 = np.abs(np.abs(np.max(query) - np.min(query)) - np.abs(np.max(pattern[st:ed]) - np.min(pattern[st:ed]))) # 全距差異

  # 計算弦差異
  _model = LinearRegression()
  y = query
  X = np.arange(len(y)).reshape(-1, 1)
  _model.fit(X, y)
  yhat = _model.predict(X)
  q_chord = y - yhat

  _model = LinearRegression()
  y = pattern[st:ed]
  X = np.arange(len(y)).reshape(-1, 1)
  _model.fit(X, y)
  yhat = _model.predict(X)
  p_chord = y - yhat
  dist4 = np.sum((q_chord - p_chord)**2)

  # 總差異
  dist = dist1 + dist2 + dist3 + dist4

  # 結果
  return dist


# 建立長度 1000 的歷史 pattern
# 使用 sawtooth 建立直角三角波 (width=0.9 代表長緩爬升後垂直掉落)
t_pattern = np.linspace(0, 20 * np.pi, 10000)
pattern = sawtooth(t_pattern, width=0.9) + np.random.normal(0, 0.1, 10000)

# 狀況 1：正常 query (從歷史片段第 300 點切取片段並加上一點新雜訊)
query_normal = pattern[300:1300] + np.random.normal(0, 0.05, 1000)

# 狀況 2：頻率變快 (Anomaly)
# 頻率提升 1.5 倍的三角波
t_fast = np.linspace(0, 20 * np.pi, 10000) * 2  
pattern_fast = sawtooth(t_fast, width=0.9)
query_fast = pattern_fast[100:1100] + np.random.normal(0, 0.05, 1000)

# 狀況 3：形狀完全不同 (Anomaly) - 例如出現反向三角波或完全沒看過的特徵
query_diff = sawtooth(np.linspace(0, 4 * np.pi, 1000), width=0.1) + np.random.normal(0, 0.05, 1000)

dtw_anomaly(query_normal, pattern), dtw_anomaly(query_fast, pattern), dtw_anomaly(query_diff, pattern)
