% 1. 读取 CSV 文件数据（推荐使用 readmatrix，适用于纯数值文件）
data = readmatrix('rfdata\rfdata_1_32.csv');
data = (data - 512) / 512;

signal = data(:,32);

fs = 25e6;

N = length(signal);

Y = fft(signal);

f = (0:N-1)*(fs/N);

half_N = floor(N/2);

figure;
plot(f(1:half_N), abs(Y(1:half_N)));
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Frequency Spectrum');
grid on;
