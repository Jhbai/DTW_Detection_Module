import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_data(num_samples, seq_len):
    x = np.linspace(0, 50 * np.pi, seq_len)
    data = []
    for _ in range(num_samples):
        trend = np.linspace(0, np.random.uniform(10, 50), seq_len)
        wave1 = np.sin(x * np.random.uniform(0.8, 1.2)) * np.random.uniform(3, 8)
        wave2 = np.cos(x * np.random.uniform(2.0, 3.0)) * np.random.uniform(1, 3)
        noise = np.random.normal(0, 0.5, seq_len)
        series = trend + wave1 + wave2 + noise
        data.append(series)
    return torch.tensor(np.array(data), dtype=torch.float32)

def rolling_z_score(x, window_size):
    x_unfold = x.unfold(dimension=-1, size=window_size, step=1)
    means = x_unfold.mean(dim=-1)
    stds = x_unfold.std(dim=-1, unbiased=False) + 1e-8
    x_cropped = x[:, window_size-1:]
    x_norm = (x_cropped - means) / stds
    return x_norm

def find_peaks_valleys(x):
    diff = x[:, 1:] - x[:, :-1]
    sign_change = diff[:, 1:] * diff[:, :-1]
    is_extrema = (sign_change < 0).float()
    padded_extrema = F.pad(is_extrema, (1, 1), mode='constant', value=0)
    return padded_extrema

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ShapeAutoEncoder(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.enc1 = ConvBlock1D(1, 16, dilation=1)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = ConvBlock1D(16, 32, dilation=2)
        self.pool2 = nn.MaxPool1d(2)
        
        self.bottleneck = ConvBlock1D(32, 64, dilation=4)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec1 = ConvBlock1D(64 + 32, 32, dilation=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec2 = ConvBlock1D(32 + 16, 16, dilation=1)
        
        self.value_head = nn.Conv1d(16, 1, kernel_size=1)
        self.peak_head = nn.Conv1d(16, 1, kernel_size=1)
        
        self.target_len = seq_len

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        b = self.bottleneck(p2)
        
        u1 = self.up1(b)
        if u1.size(2) != e2.size(2):
            u1 = F.pad(u1, (0, e2.size(2) - u1.size(2)))
        c1 = torch.cat([u1, e2], dim=1)
        d1 = self.dec1(c1)
        
        u2 = self.up2(d1)
        if u2.size(2) != e1.size(2):
            u2 = F.pad(u2, (0, e1.size(2) - u2.size(2)))
        c2 = torch.cat([u2, e1], dim=1)
        d2 = self.dec2(c2)
        
        val_out = self.value_head(d2)
        peak_out = torch.sigmoid(self.peak_head(d2))
        
        if val_out.size(2) != self.target_len:
            val_out = F.interpolate(val_out, size=self.target_len, mode='linear', align_corners=False)
            peak_out = F.interpolate(peak_out, size=self.target_len, mode='linear', align_corners=False)
            
        return val_out, peak_out

class ShapeAwareLoss(nn.Module):
    def __init__(self, w_val=1.0, w_vel=0.5, w_acc=0.2, w_peak=1.0):
        super().__init__()
        self.w_val = w_val
        self.w_vel = w_vel
        self.w_acc = w_acc
        self.w_peak = w_peak

    def forward(self, pred_val, pred_peak, true_val):
        true_vel = true_val[:, :, 1:] - true_val[:, :, :-1]
        pred_vel = pred_val[:, :, 1:] - pred_val[:, :, :-1]
        
        true_acc = true_vel[:, :, 1:] - true_vel[:, :, :-1]
        pred_acc = pred_vel[:, :, 1:] - pred_vel[:, :, :-1]
        
        true_peaks = find_peaks_valleys(true_val.squeeze(1)).unsqueeze(1)
        
        weight_map = 1.0 + 5.0 * true_peaks
        
        loss_val = (F.mse_loss(pred_val, true_val, reduction='none') * weight_map).mean()
        loss_vel = F.mse_loss(pred_vel, true_vel)
        loss_acc = F.mse_loss(pred_acc, true_acc)
        loss_peak_bce = F.binary_cross_entropy(pred_peak, true_peaks)
        
        total_loss = (self.w_val * loss_val) + (self.w_vel * loss_vel) + (self.w_acc * loss_acc) + (self.w_peak * loss_peak_bce)
        
        return total_loss

seq_length = 200
window_size = 20
norm_seq_length = seq_length - window_size + 1
num_train_samples = 500
num_epochs = 150
batch_size = 32

raw_data = generate_synthetic_data(num_train_samples, seq_length)
norm_data = rolling_z_score(raw_data, window_size).unsqueeze(1)

model = ShapeAutoEncoder(seq_len=norm_seq_length)
criterion = ShapeAwareLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
dataset = torch.utils.data.TensorDataset(norm_data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_x, in dataloader:
        optimizer.zero_grad()
        pred_val, pred_peak = model(batch_x)
        loss = criterion(pred_val, pred_peak, batch_x)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

model.eval()
test_raw = generate_synthetic_data(1, seq_length)
test_norm = rolling_z_score(test_raw, window_size).unsqueeze(1)

with torch.no_grad():
    recon_val, recon_peak = model(test_norm)

test_norm_np = test_norm.squeeze().numpy()
recon_val_np = recon_val.squeeze().numpy()
recon_peak_np = recon_peak.squeeze().numpy()
true_peaks_np = find_peaks_valleys(test_norm.squeeze(1)).squeeze().numpy()

plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(test_raw.squeeze().numpy(), label="Raw Data (With Trend)", color='gray')
plt.title("Original Time Series (Input Representation Block)")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(test_norm_np, label="True Rolling Z-Score", color='blue', alpha=0.6)
plt.plot(recon_val_np, label="Reconstructed Shape", color='red', linestyle='--')
plt.title("Shape Reconstruction (Core Architecture & Loss Design Blocks)")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(test_norm_np, color='blue', alpha=0.3, label="Base Shape")
plt.scatter(np.where(true_peaks_np == 1)[0], test_norm_np[true_peaks_np == 1], color='black', label="True Peaks/Valleys", zorder=5)
plt.plot(recon_peak_np, label="Predicted Peak Probability", color='purple')
plt.axhline(y=0.5, color='gray', linestyle=':')
plt.title("Keypoint Detector / Peak Head (Advanced Strategy Block)")
plt.legend()

plt.tight_layout()
plt.show()
