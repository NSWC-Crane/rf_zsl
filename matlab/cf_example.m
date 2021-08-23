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

[iq, iqc, i_data, q_data] = read_binary_iq_data(filename, data_type, byte_order);

% interleave the data
iq = iq';
iq_int = iq(:);

%% try curve fitting

iq_start = 50000;
io_size = 36;

iq_slice = iq_int(iq_start:io_size+iq_start-1)/2048;
cx = (0:1:io_size-1)';

[cf, cf_metrics] = fit(cx, iq_slice, 'sin3');
% re = floor(cf(cx)+0.5);
re = cf(cx);


fprintf('Curve Fit:\n');
disp(cf);

fprintf('\nFit Metrics:\n');
disp(cf_metrics);


plot(cf, cx, iq_slice)

hold on
plot(cx, re, 'g')

% calculate the ratio (num coeff / iosize) * (coeff bits / data bits)
ratio = (numel(coeffvalues(cf))/io_size)*(32/16);

fprintf('\nRatio: %10.7f\n', ratio);
