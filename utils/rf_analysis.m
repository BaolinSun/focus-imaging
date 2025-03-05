% 1. 读取 CSV 文件数据（推荐使用 readmatrix，适用于纯数值文件）
data = readmatrix('rfdata\rfdata_1_32.csv');

data = (data - 512) / 512;

% 2. 选择需要进行频谱分析的信号（这里取第一列）
signal = data(:,16);

% 3. 定义采样频率（单位：Hz），请根据实际数据修改
fs = 25e6;  % 例如：1000 Hz

% 4. 获取信号长度
N = length(signal);

% 5. 进行 FFT 变换
Y = fft(signal);

% 6. 生成频率向量（单位：Hz）
f = (0:N-1)*(fs/N);

% 7. 绘制频谱图（只显示正频率部分，一般取前半部分）
half_N = floor(N/2);
figure;
plot(f(1:half_N), abs(Y(1:half_N)));
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Frequency Spectrum');
grid on;
