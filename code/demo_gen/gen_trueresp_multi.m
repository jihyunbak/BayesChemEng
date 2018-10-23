function ftrue = gen_trueresp_multi(params,designVars,tt,noiseOpts)
% Generates "true" example response as a sum of logistic curves
% given multi-dimensional parameter and design variable.
% Function used: 
%   sum of logistic curves f = 1./(1+exp(-a*(b-t))) with multiple {a,b},
%   with f' = -a*f*(1-f), f(0) = 1/(1+exp(-a*b)).
% 
% INPUT:    params: parameters (slope $a$ of logistic curve);
%                   this version only supports a single [1 K] vector
%           designVars: design variables (translate to the "bias" $b$)
%                   also a [1 K] vector
%           tt: timepoints at which observations are made; a [T 1] vector
%           noiseOpts: settings for noise (error) scales
% OUTPUT:   ftrue: a [T 1] response vector
% ------------------------------------------------------------------------

% 2018 Ji Hyun Bak

%% unpack input

% parameters
aTrue = params(:)'; % the "slope" parameters

% design variables
offset0 = designVars(:)'; % design variables give the offsets (initial f)
bTrue = log((1-offset0)./offset0); % translate to the "bias" parameters

%% construct response

% ===== set up logistic curve =====

myexponent = bsxfun(@plus,-bTrue(:)',bsxfun(@times,aTrue(:)',tt(:)));
ftrue_clean = sum(1./(1+exp(myexponent)),2)/numel(designVars);

% ===== add noise =====

% unpack error-scale options
eta = noiseOpts.scale; % error size
corrl = noiseOpts.corrl; % noise correlation length (for now in # timepoints)
sigma_f = noiseOpts.sigmaf;

generr = generateGPSmoothNoise(numel(tt),corrl,sigma_f); % generate noise
ftrue_raw = ftrue_clean + eta*generr; % add "smooth" noise

ftrue = max(0,min(1,ftrue_raw)); % constrain within [0 1]

end
