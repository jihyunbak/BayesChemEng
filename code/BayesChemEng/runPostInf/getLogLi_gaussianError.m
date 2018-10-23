function logli = getLogLi_gaussianError(prs,myrespfun,f_exp,invSigma)
% Calculates log likelihood given actual and modeled responses, 
% assuming gaussian errors given a inverse covariance matrix.
% 
% INPUT:
%   - prs: [N K] array of parameters, where each row is a param vector
%   - myrespfun: a function handle that takes prs as the only input, 
%                and returns a [T N] array as the output
%   - f_exp: [T 1] vector of observed responses
%   - invSigma: [T T] inverse covariance matrix 
% OUTPUT:
%   - logli: [N 1] vector with the log likelihood at each param vector
% ------------------------------------------------------------------------

% 2018 Ji Hyun Bak

%% set up log likelihood

% myrespfun was fixed outside
allerr = bsxfun(@minus,f_exp,myrespfun(prs)); % [T N] array

allerr_perm2 = permute(allerr,[2 1 3]); % [N T 1]
allerr_perm3 = permute(allerr,[2 3 1]); % [N 1 T]

% now supporting multidimensional cases
mydot = @(A,B,dim) sum(bsxfun(@times,A,B),dim); % extended dot product
logli = -(1/2)*mydot(mydot(permute(invSigma,[3 1 2]),...
    allerr_perm2,2),...
    allerr_perm3,3); ... % N-vector
    
% --> multidimensional version of {-(1/2)*err(:)'*invSigma*err(:)}

% at fixed sigma, this is a constant (but needed for marginal likelihood)
%+ (1/2)*logdet(invSigma); % plus sign for inv Sigma (single number)


end
