def dtw_anomaly(query, pattern, window_length=101, polyorder=3, visualization=True):
    # 套件處理
    from dtaidistance.subsequence.dtw import subsequence_alignment
    from dtaidistance import dtw
    from sklearn.linear_model import LinearRegression
    from scipy.signal import savgol_filter
    import numpy as np
    import matplotlib.pyplot as plt

    # 時間序列整理
    query = np.asarray(query, dtype=np.double)
    pattern = np.asarray(pattern, dtype=np.double)

    # 序列平滑化
    if window_length != 0 and polyorder != 0:
        query = savgol_filter(query, window_length=window_length, polyorder=polyorder)
        pattern = savgol_filter(pattern, window_length=window_length, polyorder=polyorder)

    # DTW 計算
    sa = subsequence_alignment(query, pattern, penalty=1.0, use_c=True)
    match = sa.best_match()  # minlength=len(query)*0.8, maxlength=len(query)*1.2
    st, ed = match.segment

    if visualization:
        plt.plot(pattern[st:ed], color="blue", label="subsequence")
        plt.plot(query, color="red", label="query", linestyle="--")
        plt.grid(color="gray", linestyle="--", alpha=.4)
        plt.legend()
        plt.show()

    # 局部序列標準化
    query_norm = (query - np.mean(query)) / (np.std(query) + 1e-5)
    sub_pattern_norm = (pattern[st:ed] - np.mean(pattern[st:ed])) / (np.std(pattern[st:ed]) + 1e-5)
    dist_shape = dtw.distance(query_norm, sub_pattern_norm, use_c=True)

    # 計算異常值
    dist1 = dist_shape  # match.distance # 不需要做長度正規化！因為路徑的Warping本身就是異常
    dist2 = (
        np.abs(np.max(query) - np.max(pattern[st:ed]))
        + np.abs(np.min(query) - np.min(pattern[st:ed]))
    )  # 極值差異

    # 結果
    return dist1, dist2
