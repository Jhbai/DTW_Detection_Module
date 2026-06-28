# ----- 提取資料做滾動異常偵測 ----- #
data = np.loadtxt("./EEGRat_10_1000.txt")

len_pattern = 250
len_query = 50
dists1 = list()
dists2 = list()
pattern = data[:len_pattern]

for i in range(len_pattern + len_query, data.shape[0], len_query):
    pattern = data[i - len_pattern - len_query:i - len_query]
    query = data[i - len_query:i]
    dist1, dist2 = dtw_anomaly(query, pattern, window_length=0, polyorder=0, visualization=False)
    dists1 += [[i, dist1]]
    dists2 += [[i, dist2]]

dists1 = np.array(dists1)
dists2 = np.array(dists2)

# ----- 計算閥值 ----- #
theta1 = np.mean(dists1[:, 1]) + 2 * np.std(dists1[:, 1])
idxs1 = np.where(dists1[:, 1] > theta1)[0]

theta2 = np.mean(dists2[:, 1]) + 2 * np.std(dists2[:, 1])
idxs2 = np.where(dists2[:, 1] > theta2)[0]

# ----- 視覺化 ----- #
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(24, 6), sharex=True)
ax0.plot(data, color="blue", label="original array")
ax0.grid(color="gray", linestyle="--", alpha=.4)
ax0.legend()


def anomaly_plot(ax, dists, theta, idxs):
    ax.plot(dists[:, 0], dists[:, 1], color="red", marker="o", label="distances")
    ax.axhline(theta, color="black", label="theta", linestyle="--")
    for i, idx in enumerate(idxs):
        if i == 0:
            ax.axvline(dists[idx, 0], color="orange", linestyle="--", label="Anomaly Point")
        else:
            ax.axvline(dists[idx, 0], color="orange", linestyle="--")
    ax.grid(color="gray", linestyle="--", alpha=.4)
    ax.legend()


anomaly_plot(ax1, dists1, theta1, idxs1)
anomaly_plot(ax2, dists2, theta2, idxs2)

plt.show()
