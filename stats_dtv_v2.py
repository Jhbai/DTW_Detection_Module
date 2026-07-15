import numpy as np
from scipy.stats import norm
from dtaidistance import dtw
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar



def generate_noisy_sequence(mu, sigma):
    epsilon = np.random.normal(0, sigma, size=len(mu))
    return mu + epsilon

def get_optimal_alignment_matrix(X, Y):
    n, m = len(X), len(Y)
    path = dtw.warping_path(X, Y)
    M_hat = np.zeros((n, m))
    for i, j in path:
        M_hat[i, j] = 1
    return M_hat

def construct_omega(n, m):
    I_n = np.eye(n)
    one_m = np.ones((m, 1))
    term1 = np.kron(I_n, one_m)
    
    I_m = np.eye(m)
    one_n = np.ones((n, 1))
    term2 = np.kron(-one_n, I_m)
    
    Omega = np.hstack((term1, term2))
    return Omega

def compute_closed_form_dtw(X, Y, M_hat, Omega):
    M_vec = M_hat.flatten().reshape(-1, 1)
    XY_stacked = np.vstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
    L_prime = M_vec.T @ np.abs(Omega @ XY_stacked)
    return L_prime[0, 0]

def dtw_pipeline(X, Y):
    n, m = len(X), len(Y)
    M_hat = get_optimal_alignment_matrix(X, Y)
    Omega = construct_omega(n, m)
    L_prime = compute_closed_form_dtw(X, Y, M_hat, Omega)
    return L_prime

def estimate_noise_variance(sequence):
    seq = sequence.flatten()
    if len(seq) < 3:
        return 1e-6
    
    diff2 = np.diff(seq, n=2)
    var_est = np.mean(diff2**2) / 6.0
    
    return max(var_est, 1e-6)

def construct_joint_covariance(X, Y):
    n = len(X)
    m = len(Y)
    
    fixed_var = 0.01
    Sigma_X = fixed_var * np.eye(n)
    Sigma_Y = fixed_var * np.eye(m)
    
    Sigma_top = np.hstack((Sigma_X, np.zeros((n, m))))
    Sigma_bottom = np.hstack((np.zeros((m, n)), Sigma_Y))
    Sigma = np.vstack((Sigma_top, Sigma_bottom))
    
    return Sigma

def compute_s_hat(X, Y, M_hat, Omega):
    M_vec = M_hat.flatten().reshape(-1, 1)
    XY_stacked = np.vstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
    s_hat = np.sign(M_vec * (Omega @ XY_stacked))
    return s_hat

def compute_eta(M_hat, s_hat, Omega):
    M_vec = M_hat.flatten().reshape(-1, 1)
    diag_s = np.diag(s_hat.flatten())
    eta = (M_vec.T @ diag_s @ Omega).T
    return eta

def compute_test_statistic(X, Y, eta):
    XY_stacked = np.vstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
    T = eta.T @ XY_stacked
    return T[0, 0]

def compute_b(Sigma, eta):
    numerator = Sigma @ eta
    denominator = eta.T @ Sigma @ eta
    return numerator / denominator

def compute_a(X_obs, Y_obs, b, eta):
    XY_stacked = np.vstack((X_obs.reshape(-1, 1), Y_obs.reshape(-1, 1)))
    T_obs = eta.T @ XY_stacked
    a = XY_stacked - b * T_obs
    return a

def get_quadratic_coefficients(a, b, n):
    a_X, b_X = a[:n], b[:n]
    a_Y, b_Y = a[n:], b[n:]
    m = len(a_Y)
    
    W0 = np.zeros((n, m))
    W1 = np.zeros((n, m))
    W2 = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            p, q = a_X[i, 0], b_X[i, 0]
            r, s = a_Y[j, 0], b_Y[j, 0]
            
            W0[i, j] = (p - r)**2
            W1[i, j] = 2 * (p - r) * (q - s)
            W2[i, j] = (q - s)**2
            
    return W0, W1, W2

def solve_intersection(wA, wB):
    d0 = wA[0] - wB[0]
    d1 = wA[1] - wB[1]
    d2 = wA[2] - wB[2]
    
    if abs(d2) < 1e-9:
        if abs(d1) < 1e-9:
            return []
        return [-d0 / d1]
    
    discriminant = d1**2 - 4*d2*d0
    if discriminant < 0:
        return []
    
    if discriminant == 0:
        return [-d1 / (2*d2)]
        
    r1 = (-d1 + np.sqrt(discriminant)) / (2*d2)
    r2 = (-d1 - np.sqrt(discriminant)) / (2*d2)
    return [r1, r2]

def derivative_at(w, z):
    return w[1] + 2*w[2]*z

def get_min_at_minus_inf(candidates):
    best_idx = 0
    for i in range(1, len(candidates)):
        c_best = candidates[best_idx]['w']
        c_i = candidates[i]['w']
        if c_i[2] < c_best[2] - 1e-9:
            best_idx = i
        elif abs(c_i[2] - c_best[2]) < 1e-9:
            if c_i[1] > c_best[1] + 1e-9:
                best_idx = i
            elif abs(c_i[1] - c_best[1]) < 1e-9:
                if c_i[0] < c_best[0] - 1e-9:
                    best_idx = i
    return best_idx

def paraOA(candidates):
    if not candidates:
        return []
    
    active_idx = get_min_at_minus_inf(candidates)
    z_curr = -float('inf')
    
    envelope = [{'z_start': z_curr, 'cand': candidates[active_idx]}]
    
    while True:
        next_z = float('inf')
        next_idx = -1
        
        w_active = candidates[active_idx]['w']
        
        for i, cand in enumerate(candidates):
            if i == active_idx:
                continue
                
            roots = solve_intersection(w_active, cand['w'])
            valid_roots = [r for r in roots if r > z_curr + 1e-9]
            
            if valid_roots:
                min_valid_root = min(valid_roots)
                if min_valid_root < next_z - 1e-9:
                    next_z = min_valid_root
                    next_idx = i
                elif abs(min_valid_root - next_z) < 1e-9:
                    d_current = derivative_at(candidates[next_idx]['w'], next_z)
                    d_new = derivative_at(cand['w'], next_z)
                    if d_new < d_current:
                        next_idx = i
        
        if next_idx == -1 or next_z == float('inf'):
            break
            
        active_idx = next_idx
        z_curr = next_z
        envelope.append({'z_start': z_curr, 'cand': candidates[active_idx]})
        
    return envelope

def paraDTW(a, b, n, m):
    W0, W1, W2 = get_quadratic_coefficients(a, b, n)
    
    dp = [[[] for _ in range(m)] for _ in range(n)]
    
    for i in range(n):
        for j in range(m):
            w_ij = (W0[i,j], W1[i,j], W2[i,j])
            path_node = [(i,j)]
            
            if i == 0 and j == 0:
                candidates = [{'w': w_ij, 'path': path_node}]
            else:
                candidates = []
                predecessors = []
                if i > 0:
                    predecessors.append(dp[i-1][j])
                if j > 0:
                    predecessors.append(dp[i][j-1])
                if i > 0 and j > 0:
                    predecessors.append(dp[i-1][j-1])
                    
                for pred_envelope in predecessors:
                    for item in pred_envelope:
                        w_pred = item['cand']['w']
                        path_pred = item['cand']['path']
                        new_w = (w_pred[0]+w_ij[0], w_pred[1]+w_ij[1], w_pred[2]+w_ij[2])
                        new_path = path_pred + path_node
                        candidates.append({'w': new_w, 'path': new_path})
            
            dp[i][j] = paraOA(candidates)
            
    return dp[n-1][m-1]

def compute_Z1(a, b, n, m, M_hat_obs):
    final_envelope = paraDTW(a, b, n, m)
    Z1_intervals = []
    
    for idx in range(len(final_envelope)):
        z_start = final_envelope[idx]['z_start']
        z_end = float('inf') if idx == len(final_envelope) - 1 else final_envelope[idx+1]['z_start']
        
        path = final_envelope[idx]['cand']['path']
        M_hat_z = np.zeros((n, m))
        for (i, j) in path:
            M_hat_z[i, j] = 1
            
        if np.array_equal(M_hat_z, M_hat_obs):
            Z1_intervals.append((z_start, z_end))
            
    return Z1_intervals

def compute_Z2(s_hat_obs, M_hat_obs, Omega, a, b):
    M_vec = M_hat_obs.flatten().reshape(-1, 1)
    
    nu1 = s_hat_obs * M_vec * (Omega @ a)
    nu2 = s_hat_obs * M_vec * (Omega @ b)
    
    pos_idx = np.where(nu2 > 1e-9)[0]
    neg_idx = np.where(nu2 < -1e-9)[0]
    
    lower_bound = -np.inf
    if len(pos_idx) > 0:
        lower_bound = np.max(-nu1[pos_idx] / nu2[pos_idx])
        
    upper_bound = np.inf
    if len(neg_idx) > 0:
        upper_bound = np.min(-nu1[neg_idx] / nu2[neg_idx])
        
    return lower_bound, upper_bound

def intersect_intervals(Z1, Z2):
    Z2_lower, Z2_upper = Z2
    Z_intersected = []
    
    for z1_lower, z1_upper in Z1:
        lower = max(z1_lower, Z2_lower)
        upper = min(z1_upper, Z2_upper)
        if lower <= upper:
            Z_intersected.append((lower, upper))
            
    return Z_intersected

def compute_truncated_normal_pvalue(Z_obs, Z, sigma):
    numerator = 0.0
    denominator = 0.0
    
    for lower, upper in Z:
        l_val = norm.cdf(lower / sigma) if lower != -np.inf else 0.0
        u_val = norm.cdf(upper / sigma) if upper != np.inf else 1.0
        
        denominator += (u_val - l_val)
        
        if Z_obs <= upper:
            effective_lower = max(Z_obs, lower)
            el_val = norm.cdf(effective_lower / sigma) if effective_lower != -np.inf else 0.0
            numerator += (u_val - el_val)
            
    if denominator <= 0:
        return 1.0
        
    return numerator / denominator

def si_dtw(X_obs, Y_obs):
    n, m = len(X_obs), len(Y_obs)
    
    M_hat_obs = get_optimal_alignment_matrix(X_obs, Y_obs)
    Omega = construct_omega(n, m)
    s_hat_obs = compute_s_hat(X_obs, Y_obs, M_hat_obs, Omega)
    eta = compute_eta(M_hat_obs, s_hat_obs, Omega)
    
    Sigma = construct_joint_covariance(X_obs, Y_obs)
    
    b = compute_b(Sigma, eta)
    a = compute_a(X_obs, Y_obs, b, eta)
    
    Z_obs = compute_test_statistic(X_obs, Y_obs, eta)
    
    Z1 = compute_Z1(a, b, n, m, M_hat_obs)
    Z2 = compute_Z2(s_hat_obs, M_hat_obs, Omega, a, b)
    
    Z = intersect_intervals(Z1, Z2)
    
    variance = (eta.T @ Sigma @ eta)[0, 0]
    sigma = np.sqrt(variance) if variance > 0 else 1e-9
    
    p_selective = compute_truncated_normal_pvalue(Z_obs, Z, sigma)
    
    return p_selective

def compute_robust_pvalue(z_obs, w, sigma, Z):
    numerator = 0.0
    denominator = 0.0
    
    for lower, upper in Z:
        l_std = (lower - w) / sigma if lower != -np.inf else -np.inf
        u_std = (upper - w) / sigma if upper != np.inf else np.inf
        
        if l_std > 0:
            prob_int = norm.sf(l_std) - norm.sf(u_std)
        else:
            prob_int = norm.cdf(u_std) - norm.cdf(l_std)
            
        denominator += prob_int
        
        if z_obs < upper:
            effective_lower = max(z_obs, lower)
            el_std = (effective_lower - w) / sigma if effective_lower != -np.inf else -np.inf
            
            if el_std > 0:
                prob_num = norm.sf(el_std) - norm.sf(u_std)
            else:
                prob_num = norm.cdf(u_std) - norm.cdf(el_std)
                
            numerator += prob_num
            
    if denominator <= 0 or np.isnan(denominator):
        return 0.0
        
    return numerator / denominator

def compute_selective_ci(z_obs, Z, sigma, alpha):
    def obj_func(w, target):
        return truncated_normal_cdf(z_obs, w, sigma, Z) - target
        
    try:
        res_lower = root_scalar(obj_func, args=(1 - alpha / 2,), bracket=[z_obs - 20 * sigma, z_obs + 20 * sigma])
        w_lower = res_lower.root
    except:
        w_lower = -np.inf
        
    try:
        res_upper = root_scalar(obj_func, args=(alpha / 2,), bracket=[z_obs - 20 * sigma, z_obs + 20 * sigma])
        w_upper = res_upper.root
    except:
        w_upper = np.inf
        
    return w_lower, w_upper

def selective_ci_dtw(X_obs, Y_obs, alpha=0.05):
    n, m = len(X_obs), len(Y_obs)
    
    M_hat_obs = get_optimal_alignment_matrix(X_obs, Y_obs)
    Omega = construct_omega(n, m)
    s_hat_obs = compute_s_hat(X_obs, Y_obs, M_hat_obs, Omega)
    eta = compute_eta(M_hat_obs, s_hat_obs, Omega)
    
    Sigma = construct_joint_covariance(X_obs, Y_obs)
    
    b = compute_b(Sigma, eta)
    a = compute_a(X_obs, Y_obs, b, eta)
    
    z_obs = compute_test_statistic(X_obs, Y_obs, eta)
    
    Z1 = compute_Z1(a, b, n, m, M_hat_obs)
    Z2 = compute_Z2(s_hat_obs, M_hat_obs, Omega, a, b)
    
    Z = intersect_intervals(Z1, Z2)
    
    variance = (eta.T @ Sigma @ eta)[0, 0]
    sigma = np.sqrt(variance) if variance > 0 else 1e-9
    
    ci_lower, ci_upper = compute_selective_ci(z_obs, Z, sigma, alpha)
    
    return ci_lower, ci_upper

def dtw_hypothesis_test(X, Y, tau=0.0, alpha=0.05):
    n, m = len(X), len(Y)
    M_hat_obs = get_optimal_alignment_matrix(X, Y)
    Omega = construct_omega(n, m)
    s_hat_obs = compute_s_hat(X, Y, M_hat_obs, Omega)
    eta = compute_eta(M_hat_obs, s_hat_obs, Omega)
    Sigma = construct_joint_covariance(X, Y)
    b = compute_b(Sigma, eta)
    a = compute_a(X, Y, b, eta)
    z_obs = compute_test_statistic(X, Y, eta)
    Z1 = compute_Z1(a, b, n, m, M_hat_obs)
    Z2 = compute_Z2(s_hat_obs, M_hat_obs, Omega, a, b)
    Z = intersect_intervals(Z1, Z2)
    variance = (eta.T @ Sigma @ eta)[0, 0]
    sigma = np.sqrt(variance) if variance > 0 else 1e-9
    p_value = compute_robust_pvalue(z_obs, tau, sigma, Z)
    return bool(p_value < alpha), p_value
