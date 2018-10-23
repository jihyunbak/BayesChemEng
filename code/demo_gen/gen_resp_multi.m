function fsol = gen_resp_multi(params,designVars,tt)
% Generates simulated response for a sum of logistic curves
% given multi-dimensional parameter and design variable.
% Function used: 
%   sum of logistic curves f = 1./(1+exp(-a*(b-t))) with multiple {a,b},
%   with f' = -a*f*(1-f), f(0) = 1/(1+exp(-a*b)).
% 
% INPUT:    params: parameters (slope $a$ of logistic curve)
%                   [N K] array, where each row is a K-dim paramater vector
%           designVars: design variables (translate to the "bias" $b$);
%                   should be a [1 K] vector
%           tt: timepoints at which observations are made; a [T 1] vector
% OUTPUT:   fsol: a [T 1] response vector
% ------------------------------------------------------------------------

% 2018 Ji Hyun Bak

%% unpack input

% parameter
if(or(size(params,1)==1,size(params,2)==1))
    % if a single vector, force a row vector
    params = params(:)';
end

% get dimensions
T = numel(tt); % number of timepoints
N = size(params,1); % number of parameter vectors
K = size(params,2); % dimension of parameter space

% design variables
if(~isequal(size(designVars(:)'),[1 K])) % dimension check
    error('gen_resp_multi: params and designVars dimension mismatch');
end
bTrue = log((1-designVars)./designVars); % translate to bias


%% logistic curve model, without noise

fsol = NaN(T,N);
for ia = 1:N
    myprs = params(ia,:);
    myexponent = bsxfun(@plus,-bTrue(:)',bsxfun(@times,myprs,tt(:)));
    ftrue_clean = sum(1./(1+exp(myexponent)),2)/numel(designVars);
    fsol(:,ia) = ftrue_clean;
end

end
