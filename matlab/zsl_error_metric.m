function [dist_mean, dist_std, phase_mean, phase_std] = zsl_error_metric(x, y)
% 
%     function to calculate the magnitude and phase metrices between an
%     original dataset and a reconstructed dataset
% 
%     Parameters
%     ----------
%     x : numpy array
%         array containing the original values packed in IQIQIQ format
%     y : numpy array
%         array containing the reconstructed values packed in IQIQIQ format
% 
%     Returns
%     -------
%     dist_mean : float64
%         mean of the complex distance between points within x and y
%     dist_std : float64
%         standard deviation of the complex distance between points within x and y
%     phase_mean : float64
%         mean of the phase angle between points within x and y
%     phase_std : float64
%         standard deviation of the complex distance between points within x and y
    
    % convert x into a complex numpy array
    x = x(:);

    xc = complex(x(1:2:end), x(2:2:end));

    % convert y into a complex numpy array
    y = y(:);

    yc = complex(y(1:2:end), y(2:2:end));

    % calculate the distance error
    dist = abs(xc-yc);
    dist_mean = mean(dist);
    dist_std = std(dist);

    % compute that phases for each
    ang_xc = 180/pi * angle(xc);
    ang_yc = 180/pi * angle(yc);

    % remove any zero angles in favor of 360 degrees
    % ang_xc[ang_xc == 0] = 360
    % ang_yc[ang_yc == 0] = 360

    % compute the ratio of the phase difference
    phase_diff = (ang_xc - ang_yc);

    % compute the mean and std of the phase error
    phase_mean = mean(phase_diff);
    phase_std = std(phase_diff);

    bp = 1;

    return 