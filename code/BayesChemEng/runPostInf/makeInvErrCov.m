function invSigma = makeInvErrCov(T,myPrior)
% Constructs the inverse error covariance matrix, from (hyper-)parameters.
% 
% INPUT: T: dimensionality of response
%        myPrior: a struct with error cov parameters (sigma and tau)
% OUTPUT: invSigma: a T*T inverse covariance matrix
% ------------------------------------------------------------------------

% 2018 Ji Hyun Bak

%% unpack input

sigma = myPrior.sigma; % overall scaling factor
tau = myPrior.tau; % correlation timescale, such that C(dt) ~ exp(-dt/tau)

% form of the covariance matrix
covtype = myPrior.covtype;


%% construct the covariance matrix

% choose kernel form
switch covtype
    case 'diag1'
        getkval = @(dtval) kernel_banded_diag1(dtval,tau); % just up to first diagonal (V1)
    case 'exp'
        getkval = @(dtval) kernel_exp_var(dtval,tau,T); % exponential (V2)
    otherwise
        error('unknown covariance form');
end

% build the correlation matrix Rmat
Rmat = eye(T); % zeroth diagonal is filled with 1's
for dt = 1:T
    mykval = getkval(dt);
    Rmat = Rmat + diag(mykval.*ones(T-dt,1),dt) + diag(mykval.*ones(T-dt,1),-dt);
end

% apply the scale sigma
pCov = (sigma^2)*Rmat; % assuming a separated structure


%% get inverse covariance

% use SVD to deal with possible cases where matrix is not full-rank
[U,S,V] = svd(pCov); % errCov = U*S*V'
r = sum(diag(S)>10^-4); % rank

invSigma = U(:,1:r)*diag(1./diag(S(1:r,1:r)))*V(:,1:r)';

%%% also see these for analytical expression for exponential cov:
%%% https://math.stackexchange.com/questions/2005680
%%% https://www.lanl.gov/DLDSTP/fast/OU_process.pdf
%%% https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.74.1060

end

% ------------------------------------------------------------------------

function kval = kernel_banded_diag1(dt,tau)
% simple form, just up to the zeroth and first diagonals (old version)

rho = exp(-1/tau); % first diagonal
kval = 1*double(dt==0) + rho*double(dt==1); % zero if dt>1

end

% ------------------------------------------------------------------------

function kval = kernel_exp_var(dt,tau,T)
% corresponds to the Matern covariance with \nu = 1/2.
% tau is the correlation timescale vector (either scalar or T-vector)

if(numel(tau)==1) % single timescale
    
    kval = exp(-dt/tau); % single exponential
    
elseif(numel(tau)==T) % inhomogeneous timescale
    
    newtau = NaN(T-dt,1);
    for i = 1:(T-dt)
        newtau(i) = 1./mean(1./tau(i:i+dt)); % integrate w(t) from t[i] to t[i]+dt
    end
    kval = exp(-dt./newtau); % (T-dt) vector
    
else
    error('timescale vector size mismatch');
    
end
    
end
