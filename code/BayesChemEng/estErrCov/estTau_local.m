function mytau = estTau_local(myf_raw,varargin)
% Estimates the local decorrelation time tau(t), just from data
% 
% INPUT:
%   myf_raw: [T 1] vector for observed response
%   tnoise (optional): noise timescale
% OUTPUT: 
%   mytau: the local decorrelation timescale
% ------------------------------------------------------------------------

% 2018 Ji Hyun Bak

%% set up timescales

T = numel(myf_raw);

tnoise = 50; % all time unit is # timepoints
if(nargin>1)
    tnoise = varargin{1};
end
tnoise = min(tnoise,T-1); % limited by data length

% moving window
tw = max(tnoise,floor(T/10));


%% pre-processing

% smooth out small-scale noise
myf = smooth(myf_raw,tnoise);


%% estimate the inverse local decorrelation time w(t)

% set up mean difference (reference for "decorrelation")
dfmeanlocal = mean(abs(myf(1:end-tw)-myf(1+tw:end)));

dtlocal = (0:tw)';

% ----- sweep over time series -----

wlocal_list = NaN(T,1);

for i = 1:T
    
    % measure local variation
    cij2 = NaN(1+2*tw,1);
    for dj = -tw:tw
        j = i+dj;
        if(j<1 || j>T)
            continue;
        end
        cij2(1+dj+tw) = abs(myf(i)-myf(j));
    end
    cij1 = mean([cij2(tw+1:end) cij2(tw+1:-1:1)],2,'omitnan'); % fold
    
    % determine timescale tau in terms of # steps to reach mean difference
    slopelocal = mean(cij1(dtlocal>0)./dtlocal(dtlocal>0),'omitnan');
    taulocal = dfmeanlocal/slopelocal;
    wlocal_list(i) = 1/taulocal;
    
end

%% post-processing

% pass after smoothing
myw = smooth(wlocal_list,tw);

% apply lower cutoff (this prevents divergence when signal is flat)
myw = (1/T) + myw; % intepretation: each point carries a minimal importance

% convert back to timescale
mytau = 1./myw;

end
