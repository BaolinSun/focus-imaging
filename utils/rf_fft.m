fs = 20e6;
N_total = 4096;
N_seg = 128;
n_segments = N_total / N_seg;
scan_depth = 158;

depth = scan_depth / N_total * N_seg * (1 : n_segments);

rfdata = zeros(64, 4096, 64);
for i = 1 : 64
    rfdata(i, :, :) = readmatrix(['rawdata\heart1\rfdata_1_', num2str(i), '.csv']);
end

% data = readmatrix('rfdata\rfdata_1_1.csv');

% data = data(:, 32);
% data = bandpass_filter(data);s



f = fs * (0:(N_seg/2)) / N_seg;
rf_peak = zeros(64, 64, n_segments);

% cal_index = [1, 16, 32, 40, 48, 64];
cal_index = 1:2:64;
for j = cal_index
    disp(j)
    data = rfdata(j, :, :);
    data = squeeze(data);
    data = (data - 512) / 512;

    for i = 1 : 64
        for k = 1:n_segments
            signal = data(:, i);
            signal = bandpass_filter(signal);
        
            signal_matrix = reshape(signal, N_seg, []);
            segment = signal_matrix(:, k);
        
            Y = fft(segment);
            P2 = abs(Y / N_seg);
        
            P1 = P2(1:N_seg/2+1);
            P1(2:end-1) = 2 * P1(2:end-1);
        
            [~, idx1] = max(P1);
            f_peak1 = f(idx1) / 1e6;
        
            rf_peak(j, i, k) = f_peak1;
        end
    
    end
end

figure;
plot_index = 1;
% angles = [-45, -22.5, 0, 11.25, 22.5, 45];
angles = linspace(-45, 45, 32);
for j = cal_index
    subplot(4, 8, plot_index)

    rf_peak_mean = rf_peak(j, :, :);
    rf_peak_mean = squeeze(rf_peak_mean);
    rf_peak_mean = mean(rf_peak_mean, 1);

    plot(depth, rf_peak_mean);
    title("Scan Angle: " + num2str(angles(plot_index)) + "°");
    xlabel('Depth (mm)');
    ylabel('Freq (MHz)');
    axis([0, 160, 0.5, 3])
    grid on;

    plot_index = plot_index + 1;
end


% rf_peak_mean = mean(rf_peak, 1);

% figure;
% plot(depth, rf_peak_mean);
% xlabel('Depth (mm)')
% ylabel('Freq (MHz)')

% figure;
% hold on;
% plot(rf_peak(1, :));
% plot(rf_peak(32, :));
% plot(rf_peak(45, :));
% plot(rf_peak(63, :));




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
