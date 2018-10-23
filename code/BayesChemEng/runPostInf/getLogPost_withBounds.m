function logP = getLogPost_withBounds(flogli_input,prs,myPrior,varargin)
% Returns the log-posterior given log likelihood as the input,
% with range constraint prior.
% 
% INPUTS:
%   flogli : function handle for log likelihood
%            or a cell array of function handles (which are added up)
%      prs : parameter vector (a single [1 K] vector)
%  myPrior : struct for prior information
% 
% OUTPUT: logP: log posterior (a single number)
% -----------------------------------------------------------------------

% 2018 Ji Hyun Bak

%% apply range prior

% unpack prior information
lb = myPrior.LB;
ub = myPrior.UB;

% apply bounds
if( any(prs>ub) || any(prs<lb) )
    logP = -Inf;
    return;
end

% additional input
if(nargin>3)
    tmp = varargin{1}; % tempering factor
else
    tmp = 1;
end

%% if within range, compute log likelihood

% unpack function handle input
if(iscell(flogli_input))
    % a cell array of function handles
    flogli_list = flogli_input(:); % make a 1D cell array
else
    % a single function handle
    flogli_list = {flogli_input};
end

% sum over multiple flogli functions if applicable
numSets = numel(flogli_list);
logli_list = zeros(numSets,1);
for j = 1:numSets
    flogli = flogli_list{j}; % function handle for log likelihood
    logli_list(j) = flogli(prs);
end
logli = sum(logli_list)/tmp; % total logli (optionally tempered)

% posterior is proportional to likelihood (uniform prior)
logP = logli; % single output argument (no derivatives)

end
