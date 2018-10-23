function [psamps,accept] = runMCMC_mh(flogpdf,sampOpt)
% Run MCMC sampling with the Metropolis-Hastings algorithm, 
% given neglogpost function & range-constraint prior
%
% INPUTS
%         flogpdf  - function handle for the target function (in log)
%         sampOpt [struct] - options for the MCMC sampler,
%                    with fields {prs0,nsamples,nburn,steps}
%
% OUTPUTS
%           psamps - samples from the MCMC chain
%           accept - acceptance rate
% -----------------------------------------------------------------------

% Copyright 2018 Ji Hyun Bak

%% unpack input

% unpack sampling options
prs0 = sampOpt.prs0; % initial value for the parameter
nsamples = sampOpt.nsamples; % chain length
nburn = sampOpt.nburn; % burn-in

% step size is dynamically changing at each chain (semi-adaptive MCMC)
steps = sampOpt.steps; % either a std vector or a lower matrix
if(isequal(size(steps),numel(prs0)*[1 1]))
    if(istril(steps))
        Lmat = steps; % this is lower triangular L, such that L*L' = Cov
    else
        error('runMCMC: steps can either be a std vector or a lower-chol matrix of cov.');
    end
elseif(numel(steps)==numel(prs0))
    stepvec = reshape(steps,size(prs0));
    Lmat = diag(stepvec); % sqrt of diagonal of covariance matrix (naive ver)
end 

%% sample posterior by MCMC

% proposal distribution, scaled using our semi-adaptive algorithm
myRdist = @(x,myLmat) x(:) + myLmat*randn(size(x(:))); % multivariate normal
proprnd = @(x) reshape(myRdist(x,Lmat),size(x)); % match to prs shape

% run Metropolis-Hastings sampler
[psamps,accept] = mhsample(prs0,nburn+nsamples,'logpdf',flogpdf,...
    'proprnd',proprnd,'symmetric',1);
psamps = psamps(nburn+1:end,:); % burn in manually


end
