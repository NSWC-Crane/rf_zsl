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
filename = 'E:/Projects/rf_zsl/data/sdr_test_10M_100m_0001.bin';

[iq, iqc, i_data, q_data] = read_binary_iq_data(filename, data_type, byte_order);

% interleave the data
iq = iq';
iq_int = iq(:);

%% try curve fitting

iq_start = 50000;
io_size = 128;

iq_slice = iq_int(iq_start:io_size+iq_start-1);
cx = (0:1:io_size-1)';

cf = fit(cx, iq_slice, 'sin3');



plot(cf, cx, iq_slice)

cf




