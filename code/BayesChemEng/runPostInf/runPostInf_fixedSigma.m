function [psamps,chainacc,flogpdf,flogli_list] = ...
    runPostInf_fixedSigma(respfun_list,fexp_list,sigma_list,tau_list,myPrior,sampOpt,tmp)
% Run posterior inference with all ingredients fixed:
% construct log likelihoods, then construct and sample posterior.
% 
% INPUT:
%       - respfun_list: {M D} cell array of function handles;
%                  each cell is a different dataset
%       - fexp_list: {M D} cell array of T-vectors (observed response);
%                  T is the response length and may vary across cells
%       - sigma_list: [M D] numeric array of fluctuation scales
%       - tau_list: {M D} cell array of decorrelation timescales;
%                  each cell is either a single number of a T-vector
%       - myPrior: a struct with prior bound and error-cov hyperparameters
%       - sampOpt: a struct with MCMC sampler settings
%       - tmp: tempering factor for the posterior (single number)
% 
% OUTPUT:
%       - psamps: [N K] numeric array for sampled MCMC output,
%                 where each row is a parameter vector
%       - chainacc: acceptance rate for the MCMC chain (single number)
%       - flogpdf: function handle for the target (log) posterior
%       - flogli_list: {M D} cell array of function handles;
%                   each cell provides the log likelihood for each dataset
% ------------------------------------------------------------------------

% 2018 Ji Hyun Bak

%% construct dataset-specific likelihood functions

disp('Constructing likelihoods...');

flogli_list = cell(size(respfun_list));

for idxm = 1:size(flogli_list,1)
    for idxd = 1:size(flogli_list,2)
                
        % get model response
        myrespfun = respfun_list{idxm,idxd};
        if(isempty(myrespfun))
            fnull = @(x) 0;
            flogli_list{idxm,idxd} = fnull; % pass null function
            continue;
        end
        
        % get experimental data
        f_exp = fexp_list{idxm,idxd};
        T = size(f_exp,1);
        
        % build invSigma
        myPrior.sigma = sigma_list(idxm,idxd); % single number
        myPrior.tau = tau_list{idxm,idxd}; % either single number of a T-vector
        invSigma = makeInvErrCov(T,myPrior); % fixed-form error covariance matrix
        
        % compute log likelihood for single response type (function handle)
        flogli = @(prs) getLogLi_gaussianError(prs,myrespfun,f_exp,invSigma); % log likelihood
        
        % pass function handle in cell array
        flogli_list{idxm,idxd} = flogli;
        
    end
end


%% construct and sample posterior

disp('Sampling posterior...');

% set up (log) posterior distribution
flogpdf = @(prs) getLogPost_withBounds(flogli_list(:),prs,myPrior,tmp); % apply range prior (multiple flogli)

% MCMC sampling
[psamps,chainacc] = runMCMC_mh(flogpdf,sampOpt); % Metropolis-Hastings


end
