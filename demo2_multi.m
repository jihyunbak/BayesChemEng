% demo2_multi.m 

% Demo script illustrating inference for multidimensional parameters 
% under Gaussian noise assumptions. 
% This version works with multiple datasets (multiple design variables).

% Copyright (c) 2019 Ji Hyun Bak 

%% initialize

clear all;
clc;

setpaths; % include code directory


%% (1) model for observed and simulated responses

% Here we consider TWO datasets,
% from two experiments at different design variables (two x's).
% (In this particular model, x is the initial offset from 1.)
% Note that the design variables are known to the experimenter.

x1 = [4.5*10^-5 1*10^-1];
x2 = [1*10^-2 1*10^-5];
xVars = {x1; x2};



% ===== generated/"observed" response =====

% As in demo1, we generate the response from a hidden model;
% this sub-section can be replaced by your own experimental data.
% Try re-running this block multiple times, to see
% the effect of noise in the observed responses.


% true underlying parameter for the process
% (in this particular model, the slopes)

trueParams = [1 2.5]; 

% in real-world experiments, it is also common to have 
% different noise conditions as well:

noiseOpts1 = struct('scale',0.15,'corrl',2,'sigmaf',1); % noisy!
noiseOpts2 = struct('scale',0.05,'corrl',3,'sigmaf',1); % less noisy

% let's generate the "observed" responses, and assume these are "true" y's

dt = 0.1; % smaller dt improves simulation accuracy
tt = (0:dt:10)'; % for now, experiment duration is fixed

f_exp1 = gen_trueresp_multi(trueParams,x1,tt,noiseOpts1);
f_exp2 = gen_trueresp_multi(trueParams,x2,tt,noiseOpts2);
fexp_all = {f_exp1; f_exp2}; % we will handle multiple datasets as a package



% ===== unpack the "experimental" data =====

% detect dimensions
numExp = size(fexp_all,1); % # experiments (datasets)
numResp = size(fexp_all,2); % # response types from each experiment (here 1)
T = numel(tt); % number of datapoints

% plot true responses
clf;
for idxm = 1:numExp
    subplot(2,2,1+2*(idxm-1))
    plot(tt,fexp_all{idxm},'k-','linewidth',3)
    xlabel('time t')
    ylabel('y')
    title(['observed response at x', num2str(idxm)])
    ylim([0 1])
end
set(findall(gcf,'-property','fontsize'),'fontsize',14)

% adjust figure size
figpos = get(gcf,'position');
set(gcf,'position',[figpos(1) figpos(2) 950 520])



% ===== modeled response (full simulation) =====

% simulate the response trajectory given the design variable
% and a set of free parameters.

% obtain simulated response at varying parameter values
aa_plot = [1 1; 1 2; 2 1; 1 3; 3 2; 3 3];

% function handles for modeled responses, at respective design variables
respfun_full_all = cell(numExp,1);
for idxm = [1 2]
    myrespfun_full = @(prs) gen_resp_multi(prs,xVars{idxm},tt); % full simulation
    respfun_full_all{idxm} = myrespfun_full;
end

for idxm = 1:numExp
    subplot(2,2,2*idxm)
    myrespfun_full = respfun_full_all{idxm};
    myfsim_test = myrespfun_full(aa_plot); % handle multiple parameter values at once
    plot(tt,myfsim_test,'linewidth',2)
    xlabel('time t')
    ylabel('f')
    title(['model responses at x', num2str(idxm)])
    ylim([0 1])
    mylegs = cellstr(strcat('f(t) at \theta=(',strjust(num2str(aa_plot(:,1)),'left'),...
        ',',strjust(num2str(aa_plot(:,2)),'left'),')'));
    legend(mylegs,'location','northeast')
    legend boxoff
end
set(findall(gcf,'-property','fontsize'),'fontsize',14)


%% (2-a) parameter space setup


% ===== set up workspace =====

% set parameter bounds [min mid max]
prs1List = [0 2.5 5]; % for parameter 1
prs2List = [0 2.5 5]; % for parameter 2

% set up parameter space
prmGridInput = {prs1List(:)',prs2List(:)'};
prmGridArray = vertcat(prmGridInput{:});
prmGridTab = array2table(prmGridArray,...
    'variablenames',{'min','mid','max'},...
    'rownames',{'theta1','theta2'});
disp('Parameter grid:');
disp(prmGridTab);

lb = prmGridArray(:,1);
mid = prmGridArray(:,2);
ub = prmGridArray(:,3);

% parameter space dimension
K = numel(prmGridInput); % number of parameters being varied


% ===== construct CCD (2^K corners + K axis-endpoints + 1 center) =====

% get 2^K corner combinations
getEnds = @(myvec) {[myvec(1) myvec(end)]};
prmGridEnds = cellfun(getEnds,prmGridInput);
prmCorners = combvec(prmGridEnds{:})'; 

% get 2K axis-endpoints
getMid = @(myvec) {myvec(2)};
prmGridMids = cellfun(getMid,prmGridInput);
prmAxisEnds = NaN(2*K,K); 
for nk = 1:K
    prmAxis = prmGridMids; % copy all midpoints
    prmAxis(nk) = prmGridEnds(nk); % replace with endpoints for current axis
    prmAxisEnds(2*nk+[-1 0],:) = combvec(prmAxis{:})'; % two axis points for this axis
end

% finally, get center point and concatenate
prmCenter = combvec(prmGridMids{:})';
prmList = [prmCenter; prmAxisEnds; prmCorners]; % default: full CCD

% N = size(prmList,1); % # parameters in the grid


% ===== also prepare for test plots below =====

% finer mesh for plotting
aa1_fine = lb(1)+(ub(1)-lb(1))*(0:(1/50):1);
aa2_fine = lb(2)+(ub(2)-lb(2))*(0:(1/50):1);
aa_fine = combvec(aa1_fine(:)',aa2_fine(:)')'; % finer mesh: [N K] array

N = size(aa_fine,1); % number of test parameters (just for plotting)


%% (2-b) quadratic approximation of the response surface

% In many real-world problems, a full simulation is not feasible;
% working with a reasonable surrogate model can speed up the inference.
% Here we use a simple quadratic approximation, as in demo1.


% ===== quadratic approximation =====

respfun_approx_all = cell(numExp,numResp); % collect function handles

for idxm = 1:numExp
    for idxd = 1:numResp
    
        % sample (fully simulated) responses at the CCD parameters
        myrespfun_full = respfun_full_all{idxm,idxd};
        f_sampled_prior = myrespfun_full(prmList);
        
        % quadratic fit
        [coeff_quadfit,powermat] = getQuadFit(prmList,f_sampled_prior);
        
        % set up approx response function
        myrespfun_approx = @(prs) f_quad_approx(prs,coeff_quadfit,powermat); % quadratic approx.
        respfun_approx_all{idxm,idxd} = myrespfun_approx;
        
    end
end



% ===== visually validate quadratic approximation =====

% Let's just do this for one dataset...
% you can try other experiments/responses by simply changing the indices.

idxm = 2; % index for experiment #
idxd = 1; % index for response # (only 1 in this demo)
myrespfun_approx = respfun_approx_all{idxm,idxd};
myrespfun_full = respfun_full_all{idxm,idxd};


% construct quadratic approximation of response surface
f_approx = myrespfun_approx(aa_fine); % returns a [T N] array

% also get fully simulated responses
f_sim = myrespfun_full(aa_fine); % also a [T N] array

% reshape into 3D grid
f_approx_grid = reshape(f_approx,[T numel(aa1_fine) numel(aa2_fine)]);
f_sim_grid = reshape(f_sim,[T numel(aa1_fine) numel(aa2_fine)]);


% fix either parameter to the true value
[~,na1] = min((aa1_fine-trueParams(1)).^2);
[~,na2] = min((aa2_fine-trueParams(2)).^2);
fixtag1= ['(\theta_1 fixed at ',num2str(aa2_fine(na1)),')'];
fixtag2 = ['(\theta_2 fixed at ',num2str(aa2_fine(na2)),')'];


% ----- plot -----

clf;
colormap parula

subplot(2,3,1)
contourf(aa1_fine,tt,f_sim_grid(:,:,na2))
axis square
ylabel('time t')
xlabel('param \theta_1')
xlm1 = xlim; % axis flipped
title(['[f-full] ',fixtag2])

subplot(2,3,4)
contourf(aa2_fine,tt,squeeze(f_sim_grid(:,na1,:)))
axis square
ylabel('time t')
xlabel('param \theta_2')
xlm2 = xlim; % axis flipped
title(['[f-full] ',fixtag1])

subplot(2,3,2)
contourf(aa1_fine,tt,f_approx_grid(:,:,na2))
axis square
ylabel('time t')
xlabel('param \theta_1')
% xlm1 = xlim; % axis flipped
title(['[f-approx] ',fixtag2])

subplot(2,3,5)
contourf(aa2_fine,tt,squeeze(f_approx_grid(:,na1,:)))
axis square
ylabel('time t')
xlabel('param \theta_2')
% xlm2 = xlim; % axis flipped
title(['[f-approx] ',fixtag1])

subplot(2,3,3)
surf(aa2_fine,tt,f_sim_grid(:,:,na2),'facecolor',0.5*[1 1 1],'edgecolor','none','facealpha',0.8)
hold on
surf(aa2_fine,tt,f_approx_grid(:,:,na2),'facecolor',[0.8 0.3 0.3],'edgecolor','none','facealpha',0.8)
hold off
axis square
view(gca,[75 35]) % set orientation [azimuth,elevation]
xlabel('t')
ylabel('\theta_1')
zlabel('y')
legend('full model','quadratic approx.','location','southoutside')
legend boxoff
title('approx. response (fixed \theta_2)')

subplot(2,3,6)
surf(aa2_fine,tt,squeeze(f_sim_grid(:,na1,:)),'facecolor',0.5*[1 1 1],'edgecolor','none','facealpha',0.8)
hold on
surf(aa2_fine,tt,squeeze(f_approx_grid(:,na1,:)),'facecolor',[0.8 0.3 0.3],'edgecolor','none','facealpha',0.8)
hold off
axis square
view(gca,[75 35]) % set orientation [azimuth,elevation]
xlabel('t')
ylabel('\theta_2')
zlabel('y')
legend('full model','quadratic approx.','location','southoutside')
legend boxoff
title('approx. response (fixed \theta_1)')

set(findall(gcf,'-property','fontsize'),'fontsize',14)


%% (3-a) posterior inference - preparation


% ===== estimate error covariance based on data =====

% estimate fluctuation scale sigma (initial estimate; will be updated)
sigma_list_init = NaN(numExp,1); % numeric array
for idxm = 1:numExp
    for idxd = 1:numResp
        myrespfun = respfun_approx_all{idxm,idxd}; % use approx model
        f_prior = myrespfun(prmList); % model response: T*N double
        f_exp = fexp_all{idxm,idxd}; % observed response: T*1 double
        [~,sigma_est] = estSigma_avg(prmList,f_prior,f_exp);
        sigma_list_init(idxm,idxd) = sigma_est;
    end
end

% estimate local timescale tau
tau_list = cell(numExp,numResp); % cell array
for idxm = 1:numExp
    for idxd = 1:numResp
        f_exp = fexp_all{idxm,idxd}; % directly from observed responses
        tau_est = estTau_local(f_exp); % local timescale
        tau_list{idxm,idxd} = tau_est;
    end
end

% function to get a representative value of sigma, in multi-dataset cases
get_prod_sigma = @(sigarray) sqrt(1/mean(1./(sigarray(:).^2))); % product of gaussians



% ===== set options for iterative sampling ===== 

opts = [];

% basic info
opts.respNames = {'resp'}; % response names
opts.paramNames = {'theta1','theta2'}; % parameter names

% prior
opts.prior = struct('LB',lb(:)','UB',ub(:)','prs0',mid(:)'); % range constraints for parameters
opts.prior.covtype = 'exp'; % sigma and tau will be specified during inference

% sampler
opts.samp.prs0 = mid(:)'; % initial value for the parameter
ds = (ub-lb)/100;
opts.samp.steps = ds(:)'; % initial step size; will be adjusted during iteration
opts.samp.nsamples = 5000; % chain length (for iterative sampling)
opts.samp.nburn = 500; % burn-in

% iteration options
opts.iter.maxIter = 5; %20; % max # to iterate posterior inference
opts.iter.chainCrit = 'demo'; % dummy criterion (not used in this demo)
% opts.iter.acc_range = [0.1 0.5]; %[0.1 0.5]; % "optimal" range of acceptance ratio
% opts.iter.maxlag = 5000; % for autocorrelation function
% opts.iter.tau_cutoff = 500; %(sampOpt0.nsamples)/10; % 10*tau
opts.iter.stepScaleFactor = 0.5; % 1 means no extra scaling
opts.iter.tmpInit = 1; % no annealing


%% (3-b) posterior inference - iterative sampling


% ===== iterative sampling =====

% tempering factor is fixed to 1 (original distribution) in this demo
tmpFinal = 1;

% pass observed responses
fexp_list = fexp_all;

% ^ NOTE: If you want to consider selected experiments only,
% you can pass the corresponding subset of fexp_all here.

% for plotting
clf;
taglist = {'[f-full]','[f-approx]'};
myd0 = opts.samp.steps; % averging length scale for pdf estimate

for npan = 1:2
    
    % pack function handles for model responses.
    % (if you passed a subset for fexp_list, pass matching subsets here)
    if(npan==1)
        respfun_list = respfun_full_all; % full simulation
    elseif(npan==2)
        respfun_list = respfun_approx_all; % quadratic approx.
    end
    mytag = taglist{npan}; % for plotting

    
    % ~~~~~~~~ runIterSamp ~~~~~~~
    
    %%% ===== iterative sampling =====
    
    % reset options
    sampOpt = opts.samp;
    iterOpt = opts.iter;
    myPrior = opts.prior;
    sigma_list = sigma_list_init;
      
    % initialize
    mysigma = get_prod_sigma(sigma_list); % initial sigma estimate
    iterPack = struct('prs0',sampOpt.prs0,'steps',sampOpt.steps,'tmp',iterOpt.tmpInit);
    
    for iter = 1:(iterOpt.maxIter)
        
        iterstr = ['Iter #',num2str(iter)];
        
        disp(' ');
        disp(iterstr);
        
        % ===== sample posterior =====
        
        [resMCMC,psamps,chainacc,sampOpt,iterPack,goodsamp,flogpdf] = ...
            runSampOnce_demo(respfun_list,fexp_list,sigma_list,tau_list,sampOpt,iterPack,myPrior,iterOpt);
        
        
        % ===== plot intermediate result =====
        
        % ----- full target posterior (with estimated Sigma) -----
        
        pdf_targ = NaN(N,1);
        for na = 1:N
            pdf_targ(na) = exp(flogpdf(aa_fine(na,:)));
        end
        
        subplot(2,4,1+4*(npan-1))
        % -- color plot
        Zmat = reshape(pdf_targ,[numel(aa1_fine) numel(aa2_fine)]);
        contour(aa1_fine,aa2_fine,Zmat','linewidth',1); % transpose needed
        set(gca,'Ydir','normal')
        hold on
        plot(trueParams(1),trueParams(2),'kx','linewidth',2,'markersize',10)
        hold off
        xlim([lb(1) ub(1)])
        ylim([lb(2) ub(2)])
        axis square
        xlabel('param \theta_1')
        ylabel('param \theta_2')
        title([mytag,' target posterior'])% (up to const factor)')
        
        % ----- final sample of posterior (with estimated Sigma) -----
        
        probSamp = zeros(numel(aa1_fine),numel(aa2_fine));
        mykernel = @(dsq,lsq) exp(-dsq./(2*lsq));
        for ia1 = 1:numel(aa1_fine)
            for ia2 = 1:numel(aa2_fine)
                mya0 = [aa1_fine(ia1) aa2_fine(ia2)];
                dsq = sum((bsxfun(@minus,psamps,mya0)./myd0).^2,2);
                mycnt = sum(mykernel(dsq,1.^2)); % already scaled above
                probSamp(ia1,ia2) = mycnt;
            end
        end
        Zsamp = reshape(probSamp,[numel(aa1_fine) numel(aa2_fine)]);
        % [i1,i2]=find(Zsamp==max(Zsamp(:))); % find mode
        % prsMode = [aa1_fine(i1) aa2_fine(i2)];
        
        subplot(2,4,2+4*(npan-1))
        plot(psamps(:,1),psamps(:,2),'.','color',0.7*[1 1 1])
        hold on
        %     plot(mean(psamps(:,1)),mean(psamps(:,2)),'r+','linewidth',2,'markersize',10)
        contour(aa1_fine,aa2_fine,Zsamp','linewidth',1); % transpose needed
        plot(trueParams(1),trueParams(2),'kx','linewidth',2,'markersize',10)
        hold off
        xlim([lb(1) ub(1)])
        ylim([lb(2) ub(2)])
        axis square
        xlabel('param \theta_1')
        ylabel('param \theta_2')
        title([mytag,' sampled posterior'])% (# samples)')
        legend({'samples','density','true \theta'},'location','northeast')
        
        prsMean = mean(psamps,1);
        
        
        
        % --- response fit ---
        
        for idxm = 1:numExp
            
            idxd = 1; % only 1 response type in this demo experiment
            myrespfun_full = respfun_full_all{idxm,idxd};
            f_exp = fexp_all{idxm,idxd};
            
            subplot(2,4,2+4*(npan-1)+idxm)
            
            myftrue = myrespfun_full(trueParams);
            myfsamp = myrespfun_full(psamps);
            myfsmean = mean(myfsamp,2);
            myf_atmean = myrespfun_full(prsMean);
            
            plot(tt,myftrue,'-','color',0.7*[1 1 1],'linewidth',2)
            hold on
            plot(tt,f_exp,'k-','linewidth',2)
            plot(tt,myfsmean,'b:','linewidth',2)
            plot(tt,myf_atmean,'r--','linewidth',2)
            hold off
            legend({'true f','data y','mean(f)','f(mean)'},'location','northeast')
            xlabel('time t')
            ylabel('response y')
            title([mytag,' response fit: Exp #',num2str(idxm)])
            
        end
        
        set(findall(gcf,'-property','fontsize'),'fontsize',14)
        drawnow;
        
        
        % ===== prepare for next iteration =====
        
        % update sigma estimate 
        sigma_list_new = NaN(size(respfun_list));
        for idxm = 1:size(respfun_list,1)
            for idxd = 1:size(respfun_list,2)
                myfexp = fexp_list{idxm,idxd};
                myrespfun = respfun_list{idxm,idxd};
                myfmodel = myrespfun(psamps);
                [~,sigma_est_post] = estSigma_avg(psamps,myfmodel,myfexp);
                sigma_list_new(idxm,idxd) = sigma_est_post;
            end
        end
        sigma_list = sigma_list_new;
        
        oldsigma = mysigma;
        mysigma = get_prod_sigma(sigma_list);
        
        % update tempering factor (constant in this demo)
        oldtmp = iterPack.tmp;
        newtmp = oldtmp; % this can be replaced with a tempering factor update
        
        % adjust newsteps once more
        newsteps = iterPack.steps;
        newsteps = newsteps*((mysigma/oldsigma)^2)*(newtmp/oldtmp); % scale with sigma and tmp
        
        % update next sampler setting
        iterPack.steps = newsteps;
        iterPack.tmp = newtmp;
        
    end
    
    
    % ~~~~~~~~
    % NOTE: In practice, it is useful to wrap this block enclosed by "~~~" 
    % into a function `runIterSamp.m`. The options are intentionally 
    % packaged to support modular coding.
    
end

% Some remarks on the demo result:
% 
% * Termination condition for the iterative sampling is not implemented
%   in this demo. If you want to set you own termination criterion, you can
%   customize runSampOnce_demo.m > checkForChainOptimization_demo() in this
%   package.
% 
% * If you followed this demo as-is, you may probably see deviations in the 
%   inferred response for Exp #2, Resp #1 with the surrogate model. In this 
%   particular case, the mismatch is primarily due to the inaccuracy of the 
%   quadratic approximation (also see the example plot we made from (2-b) 
%   above) but not due to the inference algorithm. Indeed, when the model 
%   matches the actual phenomena (as in the full simulation), the response 
%   is predicted very accurately.
% 
