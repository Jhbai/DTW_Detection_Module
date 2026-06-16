import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import warnings

class TimeSeriesPatternMatcher:
    def __init__(self, noise_filter_window=11, filter_polyorder=3, dtw_radius_ratio=0.05):
        """
        參數設定：
        noise_filter_window: Savitzky-Golay filter 的視窗長度 (必須為奇數)
        filter_polyorder: 多項式擬合階數
        dtw_radius_ratio: Sakoe-Chiba band 的寬度比例。越小對頻率變化越敏感。
        """
        self.noise_window = noise_filter_window
        self.polyorder = filter_polyorder
        self.dtw_radius_ratio = dtw_radius_ratio

    def _preprocess_and_znorm(self, ts):
        """套用 Savitzky-Golay 濾波器降噪，並進行 Z-Normalization"""
        # 1. 降噪
        if len(ts) >= self.noise_window:
            smoothed = savgol_filter(ts, self.noise_window, self.polyorder)
        else:
            smoothed = ts
            
        # 2. Z-Normalization (避免除以零)
        std = np.std(smoothed)
        if std == 0:
            return np.zeros_like(smoothed)
        return (smoothed - np.mean(smoothed)) / std

    def _constrained_dtw_distance(self, s1, s2, radius):
        """
        計算帶有 Sakoe-Chiba Band 限制的 DTW 距離。
        如果頻率變化導致對齊路徑超出 radius，將返回 np.inf。
        """
        n, m = len(s1), len(s2)
        # 初始化 DP 矩陣為無限大
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            # 實作 Sakoe-Chiba Band
            start = max(1, i - radius)
            end = min(m, i + radius)
            
            for j in range(start, end + 1):
                cost = (s1[i - 1] - s2[j - 1]) ** 2
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],    # insertion
                    dtw_matrix[i, j - 1],    # deletion
                    dtw_matrix[i - 1, j - 1] # match
                )
        
        return np.sqrt(dtw_matrix[n, m])

    def detect_anomalies(self, pattern, query, corr_threshold=0.8, dtw_threshold=1.5):
        """
        pattern: shape (seq_len1, ) 歷史長序列
        query: shape (seq_len2, ) 當前觀測序列
        corr_threshold: 相關係數閾值 (低於此值視為不相似或頻率改變)
        dtw_threshold: DTW距離閾值 (大於此值視為異常)
        """
        seq_len1 = len(pattern)
        seq_len2 = len(query)
        
        if seq_len1 < seq_len2:
            raise ValueError("pattern 長度必須大於或等於 query 長度 (seq_len1 >>> seq_len2)")

        # 1. 訊號預處理：降噪與形狀萃取 (Z-Norm)
        q_norm = self._preprocess_and_znorm(query)
        p_smooth = savgol_filter(pattern, self.noise_window, self.polyorder) if seq_len1 >= self.noise_window else pattern
        
        radius = max(1, int(seq_len2 * self.dtw_radius_ratio))
        
        best_corr = -1.0
        best_dtw = np.inf
        best_idx = -1
        
        # 儲存每個滑動視窗的結果矩陣，供後續分析 (可選)
        distances = []

        # 2. 執行 Sliding Window 搜尋
        for i in range(seq_len1 - seq_len2 + 1):
            sub_pattern = p_smooth[i : i + seq_len2]
            
            # 子序列的 Z-Norm (確保是在局部上下文中的形狀對比)
            sub_std = np.std(sub_pattern)
            if sub_std == 0:
                continue
            sub_norm = (sub_pattern - np.mean(sub_pattern)) / sub_std
            
            # Metric A: Pearson Correlation (對頻率變化極度敏感)
            correlation = np.corrcoef(q_norm, sub_norm)[0, 1]
            if np.isnan(correlation):
                correlation = -1.0
                
            # Metric B: Constrained DTW (容忍微小雜訊偏移，但不容忍明顯頻率變化)
            dtw_dist = self._constrained_dtw_distance(q_norm, sub_norm, radius)
            
            distances.append({
                'index': i,
                'correlation': correlation,
                'dtw_dist': dtw_dist
            })
            
            # 尋找最相似的子序列 (以 correlation 為基準)
            if correlation > best_corr:
                best_corr = correlation
                best_dtw = dtw_dist
                best_idx = i

        # 3. 異常判定邏輯
        # 只要最相符的片段無法跨越 thresholds，就判定為異常 (包含形狀不似或頻率改變)
        is_anomaly = (best_corr < corr_threshold) or (best_dtw > dtw_threshold)
        
        result = {
            "is_anomaly": is_anomaly,
            "best_match_start_idx": best_idx,
            "best_correlation": best_corr,
            "best_dtw_distance": best_dtw,
            "reason": []
        }
        
        if is_anomaly:
            if best_corr < corr_threshold:
                result["reason"].append(f"形狀相似度不足或頻率改變 (Max Correlation {best_corr:.3f} < {corr_threshold})")
            if best_dtw > dtw_threshold:
                result["reason"].append(f"DTW 距離過大 (Min c-DTW {best_dtw:.3f} > {dtw_threshold})")
                
        return result

# ==========================================
# 測試與驗證邏輯
# ==========================================
if __name__ == "__main__":
    # 模擬長度 1000 的歷史 pattern (正弦波 + 雜訊)
    t_pattern = np.linspace(0, 50, 1000)
    pattern = np.sin(t_pattern) + np.random.normal(0, 0.2, 1000)
    
    # 狀況 1：正常 query (切取片段並加上新雜訊)
    query_normal = pattern[200:250] + np.random.normal(0, 0.1, 50)
    
    # 狀況 2：頻率變快 (Anomaly)
    t_fast = np.linspace(0, 50, 1000)  # 頻率提升 1.5 倍
    pattern_fast = np.sin(t_fast*1.5)
    query_fast = pattern_fast[100:250] + np.random.normal(0, 0.1, 150)
    
    # 狀況 3：形狀完全不同 (Anomaly)
    query_diff = np.sign(np.sin(np.linspace(0, 10, 50))) + np.random.normal(0, 0.1, 50) # 方波

    matcher = TimeSeriesPatternMatcher(noise_filter_window=11, dtw_radius_ratio=0.05)

    # 觀察原本的 pattern 是什麼樣子
    plt.plot(pattern, color = "black", label = "original")
    plt.legend()
    plt.grid(color="gray", linestyle="--", alpha=.4)
    plt.show()
    
    print("--- 測試 1: 正常序列 ---")
    res1 = matcher.detect_anomalies(pattern, query_normal)
    plt.plot(query_normal, color = "blue", label = "normal")
    plt.legend()
    plt.grid(color="gray", linestyle="--", alpha=.4)
    plt.show()
    print(res1)
    
    print("\n--- 測試 2: 頻率變快 (Anomaly) ---")
    res2 = matcher.detect_anomalies(pattern, query_fast)
    plt.plot(query_fast, color = "orange", label = "freq faster")
    plt.legend()
    plt.grid(color="gray", linestyle="--", alpha=.4)
    plt.show()
    print(res2)

    print("\n--- 測試 3: 形狀不同 (Anomaly) ---")
    res3 = matcher.detect_anomalies(pattern, query_diff)
    plt.plot(query_diff, color = "green", label = "different")
    plt.legend()
    plt.grid(color="gray", linestyle="--", alpha=.4)
    plt.show()
    print(res3)
