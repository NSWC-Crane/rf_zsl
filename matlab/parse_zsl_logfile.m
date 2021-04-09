format long g
format compact
clc
close all
clearvars

full_path = mfilename('fullpath');
[startpath,  filename, ext] = fileparts(full_path);
plot_num = 1;

%% Select the text files that contain the data

file_filter = {'*.txt','Text Files';'*.*','All Files' };
startpath = 'D:\Projects\rf_zsl\results';

[log_file, results_file_path] = uigetfile(file_filter, 'Select the Main Logfile', startpath, 'multiselect','off');
if(results_file_path == 0)
    return;
end


%% Select the text files that contain the data

file_filter = {'*.txt','Text Files';'*.*','All Files' };
startpath = 'D:\Projects\rf_zsl\results';

[results_file, results_file_path] = uigetfile(file_filter, 'Select the Results File(s)', results_file_path, 'multiselect','on');
if(results_file_path == 0)
    return;
end

commandwindow;

%% Process through the main log file to get the X value

params = parse_input_parameters(fullfile(results_file_path, log_file));

data_stats = str2double(params{1});
X = str2double(params{end});

%% Process through the files and read in the data

if(iscell(results_file))
    
    num_tests = numel(results_file);

    % arrange the data in column form where the PSO member is the column
    results = cell(num_tests,1);
    scenario_name = cell(num_tests,1);
    
    for idx=1:num_tests
        
        fprintf('%s\n', results_file{idx});
        
        % parse the file name to get the plot name.  The first 2 '_' encase the name
        k = strfind( results_file{idx}, '_');       
        scenario_name{idx} = results_file{idx}(k(1)+1:k(2)-1);
        
        %results{idx, :} = parse_csv_file(fullfile(results_file_path, results_file{idx}));
        results{idx, :} = csvread(fullfile(results_file_path, results_file{idx}));        
        
    end
    
else
    num_tests = 1;
    results = cell(num_tests, 1);
    scenario_name = cell(num_tests, 1);
    
    % parse the file name to get the plot name.  The first 2 '_' encase the name
    k = strfind(results_file, '_');
    scenario_name{1} = results_file(k(1)+1:k(2)-1);

    results{1} = csvread(fullfile(results_file_path, results_file));

end

%% plot the data
min_data = zeros(num_tests, 1);
min_idx = zeros(num_tests, 1);

min_x = 1e12;
max_x = 0;

legend_str = cell(num_tests, 1);

figure(plot_num)
set(gcf,'position',([50,100,1400,500]),'color','w')

hold on
box on
grid on

for idx=1:num_tests
    data = results{idx};
    
    [min_data(idx), min_idx(idx)] = min(data(:,2));
    
    X_hat = data(min_idx(idx), 3:end);
    
    x_diff = X - X_hat;
    
    x_error = sum(abs(x_diff(:)))/((data_stats(3) - data_stats(2))*numel(x_diff));
    
%     x_snr = 10*log10(sum((X-mean(X)).^2)/sum((X-X_hat).^2));
    
    d_si = log10(X+1) - log10(X_hat+1);  
    x_silog = sum(d_si.^2)/numel(X) - (sum(d_si)/numel(X))^2;
    
    x_nrmse = sqrt(sum(x_diff.^2))/numel(X);
    
    x_nmae = sum(abs(x_diff))/numel(X);
    
    tmp_x = min(data(:,1));
    if(tmp_x < min_x)
        min_x = tmp_x;
    end
        
    tmp_x = max(data(:,1));
    if(tmp_x > max_x)
        max_x = tmp_x;
    end
    
    plot(data(:,1), data(:,2));

    legend_str{idx} = strcat(scenario_name{idx},': NMAE =', 32, num2str(x_nmae, '%2.4f'), '; scale =', 32, num2str(data(min_idx(idx),1), '%3.3f'));
    fprintf('%s loss: %03d\n', scenario_name{idx}, data(min_idx(idx),2));
end

set(gca, 'fontweight', 'bold', 'FontSize', 13);

xlim([min_x, max_x]);
xlabel('Scale', 'fontweight', 'bold', 'FontSize', 13);

ylim([0 200]);
ylabel('Loss', 'fontweight', 'bold', 'FontSize', 13);

legend(legend_str, 'fontweight', 'bold', 'FontSize', 13) 

ax = gca;
ax.Position = [0.05 0.11 0.93 0.84];

print(plot_num, '-dpng', fullfile(results_file_path, strcat('fp_results.png')));

plot_num = plot_num + 1;  

%% deal the the mins

[min_d2, min_idx2] = min(min_data);

X_hat = results{min_idx2}(min_idx(min_idx2), 3:end);

X_hat



