import numpy as np
from scipy.stats import norm

def standard_dtw(X, Y):
    n, m = len(X), len(Y)
    DP = np.full((n + 1, m + 1), np.inf)
    DP[0, 0] = 0
    trace = np.zeros((n + 1, m + 1), dtype=int)
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (X[i-1] - Y[j-1])**2
            choices = [DP[i-1, j-1], DP[i-1, j], DP[i, j-1]]
            best_idx = np.argmin(choices)
            DP[i, j] = cost + choices[best_idx]
            trace[i, j] = best_idx
            
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        if trace[i, j] == 0:
            i -= 1; j -= 1
        elif trace[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return tuple(path[::-1]), DP[n, m]

def construct_eta_and_signs(X_obs, Y_obs, path, n, m):
    M_obs = np.zeros((n, m))
    for (i, j) in path:
        M_obs[i, j] = 1
    M_vec = M_obs.flatten()
    
    Omega = np.zeros((n * m, n + m))
    idx = 0
    for i in range(n):
        for j in range(m):
            Omega[idx, i] = 1
            Omega[idx, n + j] = -1
            idx += 1
            
    Data_obs = np.concatenate([X_obs, Y_obs])
    diff_vec = Omega @ Data_obs
    s_hat = np.sign(M_vec * diff_vec) 
    
    eta = Omega.T @ (M_vec * s_hat)
    return M_vec, s_hat, Omega, eta, Data_obs

def data_projection(Data_obs, eta, Sigma):
    z_obs = eta.T @ Data_obs
    denominator = eta.T @ Sigma @ eta
    b = (Sigma @ eta) / denominator
    q_obs = Data_obs - b * z_obs
    return z_obs, b, q_obs, denominator

def compute_Z2_sign_constraint(s_hat, M_vec, Omega, q_obs, b):
    nu_1 = s_hat * M_vec * (Omega @ q_obs)
    nu_2 = s_hat * M_vec * (Omega @ b)
    
    z_low, z_high = -np.inf, np.inf
    eps = 1e-12
    
    for i in range(len(nu_2)):
        if M_vec[i] == 0: continue
        if nu_2[i] > eps:
            z_low = max(z_low, -nu_1[i] / nu_2[i])
        elif nu_2[i] < -eps:
            z_high = min(z_high, -nu_1[i] / nu_2[i])
        else:
            if nu_1[i] < -eps: return None, None 
                
    return z_low, z_high

def get_roots(A, B, C):
    eps = 1e-11
    if abs(A) < eps:
        if abs(B) < eps: return []
        return [-C / B]
    delta = B**2 - 4*A*C
    if delta < -eps: return []
    delta = max(0, delta)
    return [(-B - np.sqrt(delta))/(2*A), (-B + np.sqrt(delta))/(2*A)]

def paraOA_intervals(quadratics):
    if not quadratics: return []
    
    def sort_key_neg_inf(q):
        return (q['A'], -q['B'], q['C'])
    
    quadratics.sort(key=sort_key_neg_inf)
    active_q = quadratics[0]
    z_curr = -np.inf
    intervals = []
    
    eps = 1e-9
    
    while True:
        min_z_next = np.inf
        next_q = None
        
        for q in quadratics:
            if q['path'] == active_q['path']: continue
            
            dA = active_q['A'] - q['A']
            dB = active_q['B'] - q['B']
            dC = active_q['C'] - q['C']
            
            roots = get_roots(dA, dB, dC)
            for r in roots:
                if r > z_curr + eps:
                    test_z = r + 1e-6
                    val_active = active_q['A']*test_z**2 + active_q['B']*test_z + active_q['C']
                    val_q = q['A']*test_z**2 + q['B']*test_z + q['C']
                    
                    if val_q < val_active and r < min_z_next:
                        min_z_next = r
                        next_q = q
                        
        if next_q is None:
            intervals.append((z_curr, np.inf, active_q))
            break
            
        intervals.append((z_curr, min_z_next, active_q))
        active_q = next_q
        z_curr = min_z_next
        
    return intervals

def paraDTW(q_obs, b, n, m):
    q_X, q_Y = q_obs[:n], q_obs[n:]
    b_X, b_Y = b[:n], b[n:]
    
    dp = [[[] for _ in range(m)] for _ in range(n)]
    
    for i in range(n):
        for j in range(m):
            q_diff = q_X[i] - q_Y[j]
            b_diff = b_X[i] - b_Y[j]
            
            local_A = b_diff**2
            local_B = 2 * q_diff * b_diff
            local_C = q_diff**2
            
            candidates = []
            if i == 0 and j == 0:
                candidates.append({
                    'path': ((0, 0),), 
                    'A': local_A, 'B': local_B, 'C': local_C
                })
            else:
                sources = []
                if i > 0 and j > 0: sources.extend(dp[i-1][j-1])
                if i > 0:           sources.extend(dp[i-1][j])
                if j > 0:           sources.extend(dp[i][j-1])
                
                for src in sources:
                    candidates.append({
                        'path': src['path'] + ((i, j),),
                        'A': src['A'] + local_A,
                        'B': src['B'] + local_B,
                        'C': src['C'] + local_C
                    })
            
            intervals = paraOA_intervals(candidates)
            unique_paths = set()
            valid_quadratics = []
            for start, end, q in intervals:
                if q['path'] not in unique_paths:
                    unique_paths.add(q['path'])
                    valid_quadratics.append(q)
            dp[i][j] = valid_quadratics
            
    final_intervals = paraOA_intervals(dp[n-1][m-1])
    return final_intervals

def calculate_selective_p_value(z_obs, z_lower, z_upper, sigma_z):
    if z_lower >= z_upper: return 1.0 
    
    std_z_obs = z_obs / sigma_z
    std_lower = z_lower / sigma_z
    std_upper = z_upper / sigma_z
    
    num = norm.cdf(std_upper) - norm.cdf(std_z_obs)
    den = norm.cdf(std_upper) - norm.cdf(std_lower)
    
    if den <= 1e-12: return 1.0
    return max(0.0, min(1.0, num / den))

if __name__ == "__main__":
    n, m = 5, 5
    X_obs = np.array([0.2, 2.5, 0.1, 0.3, 0.2]) 
    Y_obs = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) 
    
    Sigma_X = np.eye(n) * 1.0
    Sigma_Y = np.eye(m) * 1.0
    Sigma = np.block([
        [Sigma_X, np.zeros((n, m))],
        [np.zeros((m, n)), Sigma_Y]
    ])
    
    print("--- 啟動 Conditional Selective Inference for DTW ---\n")
    
    path_obs, dtw_dist = standard_dtw(X_obs, Y_obs)
    print(f"[1] 觀測到的最優路徑: {path_obs}")
    print(f"    DTW 距離 (Squared L2): {dtw_dist:.4f}")
    
    M_vec, s_hat, Omega, eta, Data_obs = construct_eta_and_signs(X_obs, Y_obs, path_obs, n, m)
    z_obs, b, q_obs, var_z = data_projection(Data_obs, eta, Sigma)
    sigma_z = np.sqrt(var_z)
    
    z2_low, z2_high = compute_Z2_sign_constraint(s_hat, M_vec, Omega, q_obs, b)
    
    final_intervals = paraDTW(q_obs, b, n, m)
    z1_low, z1_high = -np.inf, np.inf
    
    for start, end, q in final_intervals:
        if q['path'] == path_obs:
            z1_low = start
            z1_high = end
            break
            
    if z1_low == -np.inf and z1_high == np.inf:
        print(">> [警告] 數值誤差導致無法在包絡線上精確匹配 M_obs。")
    
    z_lower = max(z1_low, z2_low)
    z_upper = min(z1_high, z2_high)
    
    print(f"[2] 高維投影統計量 z_obs: {z_obs:.4f}")
    print(f"[3] Z1 截斷區間: [{z1_low:.4f}, {z1_high:.4f}]")
    print(f"[4] Z2 截斷區間: [{z2_low:.4f}, {z2_high:.4f}]")
    print(f"[5] 聯合截斷區間 Z: [{z_lower:.4f}, {z_upper:.4f}]")
    
    p_sel = calculate_selective_p_value(z_obs, z_lower, z_upper, sigma_z)
    p_naive = 1.0 - norm.cdf(z_obs / sigma_z)
    
    print("\n--- 統計檢定結果 ---")
    print(f"Naive p-value:     {p_naive:.4f} (未考量 DTW 偏誤，容易產生偽陽性)")
    print(f"Selective p-value: {p_sel:.4f} (考量路徑條件約束後，精確控制 FPR)")
    
    if p_sel < 0.05:
        print(">> 結論：拒絕虛無假設，訊號具有統計顯著異常！")
    else:
        print(">> 結論：未達顯著水準，此距離波動可能僅為演算法選擇偏誤下的雜訊。")
