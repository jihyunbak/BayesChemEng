function fapprox = f_quad_approx(param,coeff,powermat,varargin)
% Compute model responses at a set of parameters values,
% using a quadratic model (possibly obtained from quadratic approximation)
%
% INPUTS
% ----------
%   param: [N * K] array, N: #points, K: dimension of parameter space
%       - each row is a parameter vector (a point in the parameter space)
%       - each column is different parameter type
%   coeff: [T * P] array, 
%                   T : number of timepoints in each response time series
%                   P = 1 + K + K*(K+1)/2 : the # coefficients
%       - each row is [p1 p2 ...] for quadratic approximation, such that
%         f(u) = p1*u^2 + p2*u + p3. (for now for 1D parameter space)
%       - number of rows: dimensionality of the response f.
%         (for now fitted/approximated separately)
%   powermat: [P * K] array, each element is a non-negative integer
%   bound: a pair of numbers [lb ub] (optinally passed as first varargin)
%
% OUTPUT
% ----------
%   fapprox: [T * N] array
%
% =======================================================================

% 2018 Ji Hyun Bak

%% unpack input

lb = -Inf;
ub = Inf;
if(nargin>3)
    bound = varargin{1};
    lb = min(bound);
    ub = max(bound);
end

%% construct quadratic surface

coeff_perm = permute(coeff,[1 3 2 4]); % [T 1 P] array (4th dimension is 1)
param_perm = permute(param,[3 1 4 2]); % [1 N 1 K] array 
powermat_perm = permute(powermat,[3 4 1 2]); % [1 1 P K] array

allbases = prod(bsxfun(@power,param_perm,powermat_perm),4); % [1 N P] array
allterms = bsxfun(@times,coeff_perm,allbases); % [T N P] array
fapprox = sum(allterms,3); % [T N] array

fapprox = max(lb,min(ub,fapprox)); % limit within the bound


end
