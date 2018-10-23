function [coeff_quadfit,powermat] = getQuadFit(auxPrmList,f_grid)
% Obtain a quadratic approximation of the response surface 
% based on sampled parameter points.
% 
% INPUT:
%   auxPrmList: [N K] array,
%   f_grid: [T N] array,
%   - N: number of parameters sampled
%   - K: number of parameter types (parameter dimension)
%   - T: number of timepoints in response
% OUTPUT:
%   coeff_quadfit: [T P] array, P is number of coefficients being fitted
%   powermat: [P K T] array
% -----------------------------------------------------------------------

% 2018 Ji Hyun Bak

%% unpack input

% response dimensionality
T = size(f_grid,1); % number of timepoints

% number of coefficients for quadratic fit
K = size(auxPrmList,2); % parameter dimension
P = 1 + K + K*(K+1)/2; % number of coefficients to fit

%% quadratic fit

% quadratic fit of the simulated response surface

coeff_quadfit = NaN(T,P); % [T P] array, the columns [p1 p2 ...]
powermat_all = NaN(P,K,T); % [P K T]
for ncrv = 1:T
    % -- use MultiPolyRegress
    myfit = MultiPolyRegress(auxPrmList,f_grid(ncrv,:)',2);
    mycoeffs = myfit.Coefficients; % P-vector
    mypowermat = myfit.PowerMatrix; % [P K] matrix
    % -- put together, while making sure the coeffs are ordered identically
    [~,isrt] = sortrows(mypowermat);
    coeff_quadfit(ncrv,:) = mycoeffs(isrt);
    powermat_all(:,:,ncrv) = mypowermat(isrt,:);
end

powermat = powermat_all(:,:,1); % already sorted
pmcheck = any(bsxfun(@eq,powermat_all,powermat),3); % [P K] matrix
if(any(~pmcheck(:)))
    error('power matrix sorting failed');
end

end
