format long g
format compact
clc
close all
clearvars

% get the location of the script file to save figures
full_path = mfilename('fullpath');
[scriptpath,  filename, ext] = fileparts(full_path);

plot_num = 1;
line_width = 1.0;

commandwindow;

%% load in the data
byte_order = 'ieee-le';
data_type = 'int16';
filename = strcat(scriptpath, '/../data/sdr_test_10M_100m_0001.bin');
% filename = strcat(scriptpath, '/../data/rand_test_10M_100m_0000.bin');
% filename = strcat(scriptpath, '/../data/lfm_test_10M_100m_0002.bin');

[iq, iqc, i_data, q_data] = read_binary_iq_data(filename, data_type, byte_order);

% interleave the data
iq = iq';
iq_int = iq(:);

%% try curve fitting

iq_start = 50001;
io_size = 256;
sine_size = 3;

iq_slice = iq_int(iq_start:io_size+iq_start-1);
y_real = iq_slice(1:2:end);
y_imag = iq_slice(2:2:end);
% cx = (0:1:io_size-1)';
cx = (0:1:length(y_real)-1)';

% try to get a guess on the starting values
fy_real = fft(y_real);
fy_imag = fft(y_imag);

start_r = ones(3*sine_size, 1);
start_i = ones(3*sine_size, 1);

for idx=1:sine_size
    
    start_r(3*idx-2, 1) = mean(abs(y_real))/idx;
    start_i(3*idx-2, 1) = mean(abs(y_imag))/idx;
    
    [mv, ml] = max(fy_real(1:floor(io_size/2)));
    start_r(3*idx-1, 1) = 2*pi*(max(0.5, ml-1))/(cx(end)-cx(1));
    fy_real(ml) = 0;
    
    [mv, ml] = max(fy_imag(1:floor(io_size/2)));
    start_r(3*idx-1, 1) = 2*pi*(max(0.5, ml-1))/(cx(end)-cx(1));
    fy_imag(ml) = 0;
    
    %start(3*idx, 1) = 1.0;
end

% start(2) = io_size;
% start(5) = io_size/2;
% start(8) = io_size/3;

%% run the fit
[cf_r, cf_r_metrics] = fit(cx, y_real, strcat('sin', num2str(sine_size)), 'StartPoint', start_r);
[cf_i, cf_i_metrics] = fit(cx, y_imag, strcat('sin', num2str(sine_size)), 'StartPoint', start_r);

% re = floor(cf(cx)+0.5);
yr_hat = cf_r(cx);
yi_hat = cf_i(cx);

fprintf('Curve Fit:\n');
disp(cf_r);
disp(cf_i);

fprintf('\nFit Metrics:\n');
disp(cf_r_metrics);
disp(cf_i_metrics);

% calculate the ratio (num coeff / iosize) * (coeff bits / data bits)
ratio = 2*(numel(coeffvalues(cf_r))/io_size)*(32/16);
fprintf('\nRatio: %10.6f\n', ratio);

y_hat = cat(1, yr_hat.', yi_hat.');
y_hat = y_hat(:);

[dist_mean, dist_std, phase_mean, phase_std] = zsl_error_metric(iq_slice, y_hat);
fprintf('dist_mean = %10.6f, dist_std = %10.6f, phase_mean = %10.6f, phase_std = %10.6f\n', dist_mean, dist_std, phase_mean, phase_std);

%% plot
figure(plot_num)
set(gcf,'position',([50,50,1400,500]),'color','w')
grid on
box on
hold on

% plot the intial smooth curve of the equation
plot((0:0.01:io_size-1)', cf_r((0:0.01:io_size-1)'), 'color', [.6, .6, .6], 'LineWidth', line_width)

%plot the data
scatter(cx, y_real, 20, 'o', 'b', 'filled')

% plot the reconstructed points
scatter(cx, yr_hat, 20, '*', 'g')

set(gca,'fontweight','bold','FontSize',11);

xlim([0, numel(cx)]);
xlabel('Index', 'fontweight','bold','FontSize',12);

ylabel('Signal', 'fontweight','bold','FontSize',12);

ax = gca;
ax.Position = [0.05 ax.Position(2) 0.91 ax.Position(4)];

plot_num = plot_num + 1;

%% plot
figure(plot_num)
set(gcf,'position',([50,50,1400,500]),'color','w')
grid on
box on
hold on

% plot the intial smooth curve of the equation
plot((0:0.01:io_size-1)', cf_i((0:0.01:io_size-1)'), 'color', [.6, .6, .6], 'LineWidth', line_width)

%plot the data
scatter(cx, y_imag, 20, 'o', 'b', 'filled')

% plot the reconstructed points
scatter(cx, yi_hat, 20, '*', 'g')

set(gca,'fontweight','bold','FontSize',11);

xlim([0, numel(cx)]);
xlabel('Index', 'fontweight','bold','FontSize',12);

ylabel('Signal', 'fontweight','bold','FontSize',12);

ax = gca;
ax.Position = [0.05 ax.Position(2) 0.91 ax.Position(4)];

plot_num = plot_num + 1;

