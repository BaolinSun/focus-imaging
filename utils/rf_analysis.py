import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

def fft_fun(signal, fs):
    # 进行傅里叶变换
    fft_result = np.fft.fft(signal)
    n = len(signal)
    freq = np.fft.fftfreq(n, d=1/fs)

    # 只取正频率部分
    indices = np.where(freq >= 0)
    freq = freq[indices]
    fft_magnitude = np.abs(fft_result[indices])

    fft_magnitude = fft_magnitude / np.max(fft_magnitude)

    return freq, fft_magnitude

def bandpass_filter_rf_data(x, length, sampling_frequency, low_cutoff, high_cutoff):
    size = int(length)

    fft_result = fft(x)

    df = sampling_frequency / size  # 计算频率分辨率

    # 创建频率数组
    frequencies = np.arange(size) * df

    # 应用带通滤波器
    mask = (frequencies >= low_cutoff) & (frequencies <= high_cutoff)
    fft_result[~mask] = 0.0  # 将不在范围内的频率分量置零

    res = ifft(fft_result)

    return np.real(res)



df = pd.read_csv(r'rfdata\rfdata_1_32.csv', sep=',', header=None)
data = df.values
data = (data - 512) / 512
data = data.T

# for i in range(data.shape[0]):
#     data[i, :] = bandpass_filter_rf_data(data[i, :], data.shape[1], 25e6, 1.0e6, 5.0e6)

fs = 25e6
# signal1 = data[32, 100:228]
# signal2 = data[32, 1000:1128]
# signal3 = data[32, 2000:2128]

# freq1, fft_magnitude1 = fft_fun(signal1, fs)
# freq2, fft_magnitude2 = fft_fun(signal2, fs)
# freq3, fft_magnitude3 = fft_fun(signal3, fs)

# for i in range(5):
#     di = i * 790
#     signal = data[32, di:di+128]
#     signal, fft_magnitude = fft_fun(signal, fs)
#     plt.plot(signal, fft_magnitude)

freq, fft_magnitude = fft_fun(data[33, :], fs)


# 绘制频谱图
# plt.figure(figsize=(10, 4))
plt.plot(freq, fft_magnitude)
# plt.plot(freq2, fft_magnitude2)
# plt.plot(freq3, fft_magnitude3)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 8e6)
plt.grid(True)
plt.show()
