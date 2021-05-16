function d = stoi2(x, y, fs_signal)
%   d = stoi(x, y, fs_signal) returns the output of the short-time
%   objective intelligibility (STOI) measure described in [1, 2], where x 
%   and y denote the clean and processed speech, respectively, with sample
%   rate fs_signal in Hz. The output d is expected to have a monotonic 
%   relation with the subjective speech-intelligibility, where a higher d 
%   denotes better intelligible speech. See [1, 2] for more details.
%
%   References:
%      [1] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'A Short-Time
%      Objective Intelligibility Measure for Time-Frequency Weighted Noisy
%      Speech', ICASSP 2010, Texas, Dallas.
%
%      [2] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'An Algorithm for 
%      Intelligibility Prediction of Time-Frequency Weighted Noisy Speech', 
%      IEEE Transactions on Audio, Speech, and Language Processing, 2011. 
%
%
% Copyright 2009: Delft University of Technology, Signal & Information
% Processing Lab. The software is free for non-commercial use. This program
% comes WITHOUT ANY WARRANTY.
%
%
%
% Updates:
% 2011-04-26 Using the more efficient 'taa_corr' instead of 'corr'

if length(x)~=length(y)
    error('x and y should have the same length');
end

% initialization
x           = x(:);                             % clean speech column vector
y           = y(:);                             % processed speech column vector

N_fft       = 128;                             % FFT size
N_frame    	= N_fft/2;                          % window support
J           = 15;                               % Number of 1/3 octave bands - 15
N           = 4;                               % Number of frames for intermediate intelligibility measure (Length analysis window)


% remove silent frames
%[x y] = removeSilentFrames(x, y, dyn_range, N_frame, N_frame/2);

% apply 1/3 octave band TF-decomposition
frames    = 1:N_frame:(length(x)-N_fft);
x_hat     = zeros(length(frames), N_fft);
y_hat     = zeros(length(frames), N_fft);

w           = hann(N_fft);

for i = 1:length(frames)
    ii              = frames(i):(frames(i)+N_fft-1);
	x_hat(i, :) 	= fft(x(ii), N_fft)/N_fft;
	y_hat(i, :) 	= fft(y(ii), N_fft)/N_fft;
end

%x_hat     	= stdft(x, N_fft, N_frame, K); 	% apply short-time DFT to clean speech
%y_hat     	= stdft(y, N_fft, N_frame, K); 	% apply short-time DFT to processed speech

%x_hat       = x_hat(:, 1:(N_fft/2+1)).';         	% take clean single-sided spectrum
%y_hat       = y_hat(:, 1:(N_fft/2+1)).';        	% take processed single-sided spectrum
%x_hat = x_hat.';
%y_hat = y_hat.';

% X           = zeros(J, size(x_hat, 2));         % init memory for clean speech 1/3 octave band TF-representation 
% Y           = zeros(J, size(y_hat, 2));         % init memory for processed speech 1/3 octave band TF-representation 
%X           = zeros(size(x_hat, 1), size(x_hat, 2));         % init memory for clean speech 1/3 octave band TF-representation 
%Y           = zeros(size(x_hat, 1), size(y_hat, 2));         % init memory for processed speech 1/3 octave band TF-representation 

X = abs(x_hat);
Y = abs(y_hat);

% for i = 1:size(x_hat, 2)
% %     X(:, i)	= sqrt(H*abs(x_hat(:, i)).^2);      % apply 1/3 octave bands as described in Eq.(1) [1]
% %     Y(:, i)	= sqrt(H*abs(y_hat(:, i)).^2);
% %     
%     X(:, i)	= sqrt(abs(x_hat(:, i)).^2); 
%     Y(:, i)	= sqrt(abs(y_hat(:, i)).^2);    
% end

% loop al segments of length N and obtain intermediate intelligibility measure for all TF-regions
d_interm  	= zeros(size(X, 1), length(N:size(X, 2)));                               % init memory for intermediate intelligibility measure


for m=1:size(X, 1)

    d_int(m) = taa_corr(X(m,:), Y(m,:));

end

d = mean(d_int(:)); 

% for m = N:size(X, 2)
%     X_seg  	= X(:, (m-N+1):m);                                              % region with length N of clean TF-units for all j
%     Y_seg  	= Y(:, (m-N+1):m);                                              % region with length N of processed TF-units for all j
%     %alpha   = sqrt(sum(X_seg.^2, 2)./sum(Y_seg.^2, 2));                     % obtain scale factor for normalizing processed TF-region for all j
%     %aY_seg 	= Y_seg.*repmat(alpha, [1 N]);                               	% obtain \alpha*Y_j(n) from Eq.(2) [1]
%     for j = 1:size(X, 1)
%       	%Y_prime             = min(aY_seg(j, :), X_seg(j, :)+X_seg(j, :)*c); % apply clipping from Eq.(3)   	
%         %d_interm(j, m-N+1)  = taa_corr(X_seg(j, :).', Y_prime(:));          % obtain correlation coeffecient from Eq.(4) [1]
%         d_interm(j, m-N+1)  = taa_corr(X_seg(j,:), Y_seg(j,:));          % obtain correlation coeffecient from Eq.(4) [1]
%     end
% end
%         
% d = mean(d_interm(:));                                                      % combine all intermediate intelligibility measures as in Eq.(4) [1]

%%
function x_stdft = stdft(x, N, K, N_fft)
%   X_STDFT = X_STDFT(X, N, K, N_FFT) returns the short-time
%	hanning-windowed dft of X with frame-size N, overlap K and DFT size
%   N_FFT. The columns and rows of X_STDFT denote the frame-index and
%   dft-bin index, respectively.

frames      = 1:K:(length(x)-N);
x_stdft     = zeros(length(frames), N_fft);

w           = hanning(N);
x           = x(:);

for i = 1:length(frames)
    ii              = frames(i):(frames(i)+N-1);
	x_stdft(i, :) 	= fft(x(ii), N_fft)/N_fft;
end


%%
function rho = taa_corr(x, y)
%   RHO = TAA_CORR(X, Y) Returns correlation coeffecient between column
%   vectors x and y. Gives same results as 'corr' from statistics toolbox.
xn    	= x-mean(x);
yn   	= y-mean(y);

xn  	= xn/sqrt(sum(xn.^2));
yn    	= yn/sqrt(sum(yn.^2));

rho   	= sum(xn.*yn);



