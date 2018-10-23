function [sigma_est,sigma_est_noshift,errCov,errCov_noshift] = ...
    estSigma_avg(prs,f_samp,f_exp)
% Estimates the error covariance scale from parameter samples.
% 
% INPUT: 
%   - prs:  an [N K] array of parameter vectors
%           (usually samples a distribution over the parameter space)
%   - f_samp: a [T N] array, where each column is a model response vector 
%           predicted by a parameter sample
%   - f_exp:  a [T 1] vector of observed response
% OUTPUT:
%   - sigma_est: estimated sigma (a single number), with mean-shifting
%   - sigma_est_noshift: estimated sigma, without mean-shifting (USE THIS)
%   - errCov, errCov_noshift: corresponding T*T error covariance matrices
% ------------------------------------------------------------------------

% 2018 Ji Hyun Bak

%% unpack input

T = size(f_exp,1); % response length (number of timepoints)
N = size(prs,1); % number of parameter samples

% check dimensions
if(~isequal(size(f_samp),[T N]))
    error('estSigma_avg: dimension mismatch');
end

%% estimate from variance

% get errors
allerr_raw = bsxfun(@minus,f_exp,f_samp); % [T N]
errMean = mean(allerr_raw,1); % mean is usually non-zero
allerr = bsxfun(@minus,allerr_raw,errMean); % set mean to zero, to get cov

% rough estimate for error covariance matrix
errCov_sum = zeros(T,T); % bsxfun goes out of memory in this high-dim case
errCov_sum_noshift = zeros(T,T); % without subtracting errMean
for np = 1:N
    errCov_sum = errCov_sum + allerr(:,np)*allerr(:,np)';
    errCov_sum_noshift = errCov_sum_noshift + allerr_raw(:,np)*allerr_raw(:,np)';
end
errCov = errCov_sum/N;
errCov_noshift = errCov_sum_noshift/N;

% standard deviation of model-data discrepancy --> estimate of sigma
sigma_est = sqrt(mean(diag(errCov))); 
sigma_est_noshift = sqrt(mean(diag(errCov_noshift)));

end
