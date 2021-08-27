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
% filename = strcat(scriptpath, '/../data/sdr_test_10M_100m_0001.bin');
% filename = strcat(scriptpath, '/../data/rand_test_10M_100m_0000.bin');
filename = strcat(scriptpath, '/../data/lfm_test_10M_100m_0002.bin');

[iq, iqc, i_data, q_data] = read_binary_iq_data(filename, data_type, byte_order);

% interleave the data
iq = iq';
iq_int = iq(:);

%% try curve fitting

iq_start = 38941;
io_size = 512;
sine_size = 2;

iq_slice = iq_int(iq_start:io_size+iq_start-1);
cx = (0:1:io_size-1)';

% try to get a guess on the starting values
fy = fft(iq_slice);
start = ones(3*sine_size, 1);

for idx=1:sine_size
    
    start(3*idx-2, 1) = mean(abs(iq_slice))/idx;
    
    [mv, ml] = max(fy(1:floor(io_size/2)));
    start(3*idx-1, 1) = 2*pi*(max(0.5, ml-1))/(cx(end)-cx(1));
    fy(ml) = 0;
    
    %start(3*idx, 1) = 1.0;
end

% start(2) = io_size;
% start(5) = io_size/2;
% start(8) = io_size/3;


[cf, cf_metrics] = fit(cx, iq_slice, strcat('sin', num2str(sine_size)), 'StartPoint', start);
% re = floor(cf(cx)+0.5);
re = cf(cx);

fprintf('Curve Fit:\n');
disp(cf);

fprintf('\nFit Metrics:\n');
disp(cf_metrics);

% calculate the ratio (num coeff / iosize) * (coeff bits / data bits)
ratio = (numel(coeffvalues(cf))/io_size)*(32/16);

fprintf('\nRatio: %10.7f\n', ratio);
%% plot
figure(plot_num)
set(gcf,'position',([50,50,1400,500]),'color','w')
grid on
box on
hold on

% plot the intial smooth curve of the equation
plot((0:0.01:io_size-1)', cf((0:0.01:io_size-1)'), 'color', [.6, .6, .6], 'LineWidth', line_width)

%plot the data
scatter(cx, iq_slice, 20, 'o', 'b', 'filled')

% plot the reconstructed points
scatter(cx, re, 20, '*', 'g')

set(gca,'fontweight','bold','FontSize',11);

xlim([0, numel(cx)]);
xlabel('Index', 'fontweight','bold','FontSize',12);

ylabel('Signal', 'fontweight','bold','FontSize',12);

ax = gca;
ax.Position = [0.05 ax.Position(2) 0.91 ax.Position(4)];

plot_num = plot_num + 1;

