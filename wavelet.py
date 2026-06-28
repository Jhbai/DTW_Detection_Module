# ----- 小波轉換 ----- #
def wavelet_denoising(x, wavelet='db4', level=2):
    import pywt
    coeff = pywt.wavedec(x, wavelet, mode="per")

    # 基於 Sigma 的動態閾值 (Donoho 萬能閾值公式)
    sigma = np.median(np.abs(coeff[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))

    # 對高頻係數進行軟閾值過濾
    coeff[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:]]
    return pywt.waverec(coeff, wavelet, mode="per")
