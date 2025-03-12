% 1. 读取 CSV 文件数据（推荐使用 readmatrix，适用于纯数值文件）
data = readmatrix('rfdata\rfdata_1_32.csv');
data = (data - 512) / 512;
data = bandpass_filter(data);

x = data(3000:3500, 32);
fs = 20e6;
L = length(x);
T = 1/ fs;
t = (0 : L-1)*T;

Y = fft(x);

Y = abs(Y/L);
P1 = Y(1 : L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = fs * (0 : (L/2)) / L;

% 绘制频谱图
figure;
plot(f/1e6, P1);    % 将频率转换为MHz单位
title('单边幅值谱');
xlabel('频率 (MHz)');
ylabel('|幅值|');
grid on;

% 显示主要频率峰值
[~, idx1] = max(P1);
f_peak1 = f(idx1)/1e6;
disp(['检测到峰值频率：', num2str(f_peak1), ' MHz']);


% =================================================================
function filtered_signal = bandpass_filter(signal)
    % 带通滤波器函数
    % 输入参数：
    %   signal - 待滤波的信号
    % 输出参数：
    %   filtered_signal - 滤波后的信号

    % 设置采样频率
    fs = 20e6;  % 20 MHz

    % 设置带通滤波器的截止频率（1MHz 到 5MHz）
    low_cutoff = 0.5e6;
    high_cutoff = 5e6;

    % 归一化截止频率（归一化至 Nyquist 频率 fs/2）
    Wn = [low_cutoff high_cutoff] / (fs/2);

    % 选择滤波器阶数（可根据具体需求调整）
    order = 4;

    % 使用 Butterworth 方法设计带通滤波器
    [b, a] = butter(order, Wn, 'bandpass');

    % 使用 filtfilt 进行零相位滤波，避免相位延迟
    filtered_signal = filtfilt(b, a, signal);
end
