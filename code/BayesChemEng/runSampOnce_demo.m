function [resMCMC,psamps,chainacc,outSampOpt,outPack,goodsamp,flogpdf] = ...
    runSampOnce_demo(...
    respfun_list,fexp_list,sigma_list,tau_list,inSampOpt,inPack,myPrior,iterOpt)
% single sampling of the posterior (simplified for demo script)
% 
% INPUT:
%       - respfun_list: {M D} cell array of function handles;
%                  each cell is a different dataset
%       - fexp_list: {M D} cell array of T-vectors (observed response);
%                  T is the response length and may vary across cells
%       - sigma_list: [M D] numeric array of fluctuation scales
%       - tau_list: {M D} cell array of decorrelation timescales;
%                  each cell is either a single number of a T-vector
%       - inSampOpt: incoming sampler options (a struct)
%       - inPack: incoming iteration package (a struct)
%       - myPrior: a struct with prior bound and error-cov hyperparameters
%       - iterOpt: a struct with iterative sampling algorithm settings
% 
% OUTPUT:
%       - resMCMC: a struct for the MCMC result summary
%       - psamps: [N K] numeric array for sampled MCMC output,
%                 where each row is a parameter vector
%       - chainacc: acceptance rate for the MCMC chain (single number)
%       - outSampOpt: updated sampler options for next round (a struct)
%       - outPack: updated iteration package for next round (a struct)
%       - goodsamp: a boolean variable that reports termination condition
%       - flogpdf: function handle for the target (log) posterior
% ------------------------------------------------------------------------

% 2018 Ji Hyun Bak

%% unpack input

% set up sampler options
sampOpt = inSampOpt; % default options
% update with input package for current iteration (inPack)
sampOpt.prs0 = inPack.prs0;
sampOpt.steps = inPack.steps;

tmp = inPack.tmp; % tempering factor

chainCrit = iterOpt.chainCrit; % termination criterion

%% sample the posterior distribution, at fixed sigma

% construct and sample posterior, at fixed err covariance
t0 = tic;
[psamps,chainacc,flogpdf,~] = ...
    runPostInf_fixedSigma(respfun_list,fexp_list,sigma_list,tau_list,myPrior,sampOpt,tmp);
t1 = toc(t0);

disp(['acceptance rate = ',num2str(chainacc)]);
disp(['elapsed time = ',num2str(t1),'s']);

% report sampling result summary
resMCMC = struct('prsInit',sampOpt.prs0,'prsFinal',psamps(end,:),...
    'prsMean',mean(psamps,1),'prsStd',std(psamps,[],1),...
    'accept',chainacc);

% check for chain optimization
goodsamp = checkForChainOptimization_demo(chainacc,iterOpt,chainCrit);


% adjust sampler for next chain
[newprs0,newsteps] = adjustNextSampler(psamps,myPrior,iterOpt);

% adjust tempering factor 
newtmp = tmp; % room for customization (no change in this version)

%% pack output

% pass next sampler settings
outSampOpt = sampOpt;
outPack = struct('prs0',newprs0,'steps',newsteps,'tmp',newtmp);

end

% ------------------------------------------------------------------------

function [newprs0,newsteps] = adjustNextSampler(psamps,myPrior,iterOpt)
% adjust sampler properties (initialization and step sizes) 
% for the next chain, based on the statistics of the current chain.

% unpack info for parameter space
prs0_init = myPrior.prs0;
K = numel(prs0_init); % dimensionality of parameter space

% optional scaling factor
rho = 1; % 1 is the standard choice
if(isfield(iterOpt,'stepScaleFactor'))
    rho = iterOpt.stepScaleFactor; % optionally adjust scaling factor
end

% adjust step size for next chain
sdratio = (2.38^2)/K; % [Gelman1996, via Harrio2001]
myepsilon = 0; %0.01; % we can add small perturbation to the diagonal
newsteps = sqrt(sdratio)*(myepsilon+std(psamps,[],1));
newsteps = newsteps*rho;

% adjust initialization for next chain
newprs0 = mean(psamps,1);
if(any(isnan(newprs0)))
    % in case something goes wrong
    warning('mean sample has NaN, restoring default initialization...');
    newprs0 = prs0_init(:)'; % restore default value
end

end

% ------------------------------------------------------------------------

function goodsamp = checkForChainOptimization_demo(chainacc,iterOpt,chainCrit)
% check if the MCMC chain satisfies the termination conditions 
% according to the termination criterion option (chainCrit)

switch chainCrit
    case 'demo'
        % demo version: just keep iterating, no termination
        goodsamp = false; 
    case 'accept'
        % just an example to show how the acceptance ratio may be 
        % passed through iterOpt and used to evaluate termination condition;
        % in the paper we also used autocorrelation ftn (not shown here)
        acc_range = iterOpt.acc_range; % "optimal" range of acceptance ratio
        crit_acc = and(chainacc>acc_range(1),chainacc<acc_range(2)); % check if within the optimal range
        goodsamp = crit_acc;
    otherwise
        error('other criteria are not supported in this demo.');
end

end
