import numpy as np
from scipy.signal import savgol_filter, sawtooth
import matplotlib.pyplot as plt
import warnings

# 導入快速 DTW 套件 (需先 pip install dtaidistance)
try:
    from dtaidistance import dtw
except ImportError:
    raise ImportError("請先安裝 dtaidistance 套件: pip install dtaidistance")


class TimeSeriesPatternMatcher:
    def __init__(self, noise_filter_window=11, filter_polyorder=3, dtw_radius_ratio=0.05):
        """
        參數設定：
        noise_filter_window: Savitzky-Golay filter 的視窗長度 (必須為奇數)
        filter_polyorder: 多項式擬合階數
        dtw_radius_ratio: DTW 視窗約束的寬度比例。越小對頻率變化越敏感。
        """
        self.noise_window = noise_filter_window
        self.polyorder = filter_polyorder
        self.dtw_radius_ratio = dtw_radius_ratio

    def _preprocess_and_znorm(self, ts):
        """套用 Savitzky-Golay 濾波器降噪，並進行 Z-Normalization"""
        if len(ts) >= self.noise_window:
            smoothed = savgol_filter(ts, self.noise_window, self.polyorder)
        else:
            smoothed = ts
            
        std = np.std(smoothed)
        if std == 0:
            return np.zeros_like(smoothed)
        return (smoothed - np.mean(smoothed)) / std

    def detect_anomalies(self, pattern, query, dtw_threshold=0.6):
        """
        pattern: shape (seq_len1, ) 歷史長序列
        query: shape (seq_len2, ) 當前觀測序列
        dtw_threshold: 標準化的 DTW 距離閾值。數值代表「每個點的平均標準差誤差」。
                       大於此值視為歷史中找不到相似片段 (異常)。
        """
        seq_len1 = len(pattern)
        seq_len2 = len(query)
        
        if seq_len1 < seq_len2:
            raise ValueError("pattern 長度必須大於或等於 query 長度 (seq_len1 >= seq_len2)")

        # 1. 預處理 Query (必須轉型為 float64 給 dtaidistance 使用)
        q_norm = self._preprocess_and_znorm(query).astype(np.float64)
        
        # 預先對 Pattern 平滑化
        p_smooth = savgol_filter(pattern, self.noise_window, self.polyorder) if seq_len1 >= self.noise_window else pattern
        
        # 設定 DTW 視窗步數限制 (用來捕捉頻率改變)
        radius = max(1, int(seq_len2 * self.dtw_radius_ratio))
        
        best_dtw_norm = np.inf
        best_idx = -1

        # 2. Sliding Window 搜尋歷史中是否有一段長得很像的子序列
        for i in range(seq_len1 - seq_len2 + 1):
            sub_pattern = p_smooth[i : i + seq_len2]
            
            # 子序列的 Z-Norm (局部形狀對比)
            sub_std = np.std(sub_pattern)
            if sub_std == 0:
                continue
            sub_norm = ((sub_pattern - np.mean(sub_pattern)) / sub_std).astype(np.float64)
            
            # 呼叫 C 語言底層加速的 DTW，加入 window 限制以鎖死頻率作弊空間
            dtw_dist = dtw.distance_fast(q_norm, sub_norm, window=radius)
            
            # 3. 距離標準化 (除以 sqrt(N) 使得閾值與長度脫鉤，意義轉為平均單點誤差)
            norm_dist = dtw_dist / np.sqrt(seq_len2)
            
            # 更新最佳匹配紀錄
            if norm_dist < best_dtw_norm:
                best_dtw_norm = norm_dist
                best_idx = i

        # 4. 異常判定邏輯：如果在歷史紀錄中找不到任何一段低於閾值的子序列，就是異常
        is_anomaly = best_dtw_norm > dtw_threshold
        
        result = {
            "is_anomaly": is_anomaly,
            "best_match_start_idx": best_idx,
            "best_dtw_distance_normalized": best_dtw_norm,
            "reason": f"未知的形狀或頻率過快 (距離 {best_dtw_norm:.3f} > 閾值 {dtw_threshold})" if is_anomaly else "正常，於歷史數據中尋得相似樣態"
        }
                
        return result


# ==========================================
# 測試與驗證邏輯
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)
    
    # 建立長度 1000 的歷史 pattern
    # 使用 sawtooth 建立直角三角波 (width=0.9 代表長緩爬升後垂直掉落)
    t_pattern = np.linspace(0, 20 * np.pi, 1000)
    pattern = sawtooth(t_pattern, width=0.9) + np.random.normal(0, 0.1, 1000)
    
    # 狀況 1：正常 query (從歷史片段第 300 點切取片段並加上一點新雜訊)
    query_normal = pattern[300:400] + np.random.normal(0, 0.05, 100)
    
    # 狀況 2：頻率變快 (Anomaly)
    # 頻率提升 1.5 倍的三角波
    t_fast = np.linspace(0, 20 * np.pi, 1000) * 2  
    pattern_fast = sawtooth(t_fast, width=0.9)
    query_fast = pattern_fast[100:200] + np.random.normal(0, 0.05, 100)
    
    # 狀況 3：形狀完全不同 (Anomaly) - 例如出現反向三角波或完全沒看過的特徵
    query_diff = sawtooth(np.linspace(0, 4 * np.pi, 100), width=0.1) + np.random.normal(0, 0.05, 100) 

    # 初始化模組 (視窗限制 dtw_radius_ratio=0.05，對頻率變化敏感)
    matcher = TimeSeriesPatternMatcher(noise_filter_window=11, dtw_radius_ratio=0.05)
    
    # 圖形化展示 Helper
    def plot_result(title, q_data, res_dict, query_color):
        plt.figure(figsize=(10, 4))
        plt.plot(pattern, color="lightgray", label="Historical Pattern (Background)")
        
        # 若有找到最佳匹配位置，畫出對齊位置的框框
        best_idx = res_dict['best_match_start_idx']
        q_len = len(q_data)
        
        if best_idx != -1:
            plt.plot(range(best_idx, best_idx + q_len), pattern[best_idx:best_idx + q_len], 
                     color="black", linewidth=2, label="Best Found Match in History")
            
        plt.plot(range(best_idx, best_idx + q_len), q_data, 
                 color=query_color, linestyle="--", label="Current Query")
        
        plt.title(f"{title} | Anomaly: {res_dict['is_anomaly']} | Dist: {res_dict['best_dtw_distance_normalized']:.3f}")
        plt.legend()
        plt.grid(color="gray", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

    # --- 執行測試 ---
    print("--- 觀察: 原始直角三角歷史資料 ---")
    plt.figure(figsize=(10, 3))
    plt.plot(pattern, color="black", label="Historical Pattern (Sawtooth)")
    plt.legend()
    plt.grid(color="gray", linestyle="--", alpha=0.4)
    plt.show()

    print("--- 測試 1: 正常序列 ---")
    res1 = matcher.detect_anomalies(pattern, query_normal, dtw_threshold=0.6)
    print(res1)
    plot_result("Test 1: Normal Query", query_normal, res1, "blue")
    
    print("\n--- 測試 2: 頻率變快 (Anomaly) ---")
    res2 = matcher.detect_anomalies(pattern, query_fast, dtw_threshold=0.6)
    print(res2)
    plot_result("Test 2: Fast Frequency Query", query_fast, res2, "orange")

    print("\n--- 測試 3: 形狀不同 / 反向波 (Anomaly) ---")
    res3 = matcher.detect_anomalies(pattern, query_diff, dtw_threshold=0.6)
    print(res3)
    plot_result("Test 3: Different Shape Query", query_diff, res3, "red")
