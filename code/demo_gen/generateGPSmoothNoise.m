function generr = generateGPSmoothNoise(T,tcorr,sigma_f)
% Generates gaussian-process noise at given correlation scale, 
% using a square exponential kernel.
% 
% INPUT:
%   T: the dimensionality of the response (length of trajectory)
%   tcorr: correlation scale (in the same unit as T)
%   sigma_f: overall noise scale
% OUTPUT: 
%   generr: a [T 1] vector sampled from the GP noise kernel
% ------------------------------------------------------------------------

% 2018 Ji Hyun Bak

%% set a GP kernel

k = @(x,x2) sigma_f^2*exp(-(x-x2).^2/(tcorr^2)); % square exponential
% k = @(x,x2) sigma_f^2*exp(-abs(x-x2)/tcorr); % exponential kernel

%% draw samples

% construct kernel matrix
Kmat = zeros(T,T);
for i=1:T
    for j=i:T 
        Kmat(i,j) = k(i,j); % just the upper half
    end
end
% fill in the lower half by copying the upper half (matrix is symmetric)
Kmat = Kmat + triu(Kmat,1)';

% add small perturbation to the diagonal, to ensure semi-positive definite
Kmat = Kmat + 0.01*eye(size(Kmat));

% sample from kernel
noise_uncorr = randn(T,1); % uncorrelated noise
A = chol(Kmat);
generr = A * noise_uncorr;

end
