% demo1_single.m 

% Demo script illustrating inference for multidimensional parameters 
% under Gaussian noise assumptions. 
% This version works with a single dataset (single design variable).

% Copyright (c) 2018 Ji Hyun Bak 

%% initialize

clear all;
clc;

setpaths; % include code directory


%% (1) model for observed and simulated responses

% ===== generated/"observed" response =====

% in this demo, generate the response from a hidden model;
% to be replaced by experimental data.

% true parameter and design variable (xVar)
trueParams = [1 2.5]; % two slopes
xVars = [4.5*10^-5 1*10^-2]; % initial offset from 1

% noise setting
noiseOpts = struct('scale',0.05,'corrl',3,'sigmaf',1);

% generate a set of "observed" response
dt = 0.1; % smaller dt improves simulation accuracy
tt = (0:dt:10)'; % for now, experiment duration is fixed
f_exp = gen_trueresp_multi(trueParams,xVars,tt,noiseOpts);

% plot true response
clf;
subplot(2,2,1)
plot(tt,f_exp,'k-','linewidth',3)
xlabel('time t')
ylabel('y')
title('observed response')
ylim([0 1])
set(findall(gcf,'-property','fontsize'),'fontsize',14)

% adjust figure size
figpos = get(gcf,'position');
set(gcf,'position',[figpos(1) figpos(2) 950 520])


% ===== modeled response (full simulation) =====

% simulate the response trajectory given the design variable
% and a set of free parameters (in this example, only one parameter a).

myrespfun_full = @(prs) gen_resp_multi(prs,xVars,tt); % full simulation

% obtain simulated response at varying parameter values
aa_plot = [1 1; 1 2; 2 1; 1 3; 3 2; 3 3];

subplot(2,2,2)
myfsim_test = myrespfun_full(aa_plot);
plot(tt,myfsim_test,'linewidth',2)
xlabel('time t')
ylabel('f')
title('model response')
ylim([0 1])
mylegs = cellstr(strcat('f(t) at \theta=(',strjust(num2str(aa_plot(:,1)),'left'),...
    ',',strjust(num2str(aa_plot(:,2)),'left'),')'));
legend(mylegs,'location','northeast')
legend boxoff
set(findall(gcf,'-property','fontsize'),'fontsize',14)


%% (2) quadratic approximation of the response surface


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

% detect dimensions
T = numel(tt); % number of datapoints
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


% ===== quadratic approximation =====

% sample (fully simulated) responses at the CCD parameters
f_sampled_prior = myrespfun_full(prmList);

% quadratic fit
[coeff_quadfit,powermat] = getQuadFit(prmList,f_sampled_prior);

% set up approx response function
myrespfun_approx = @(prs) f_quad_approx(prs,coeff_quadfit,powermat); % quadratic approx.


% ===== visually validate quadratic approximation =====

% finer mesh for plotting
aa1_fine = lb(1)+(ub(1)-lb(1))*(0:(1/50):1);
aa2_fine = lb(2)+(ub(2)-lb(2))*(0:(1/50):1);
aa_fine = combvec(aa1_fine(:)',aa2_fine(:)')'; % finer mesh: [N K] array
N = size(aa_fine,1);

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


%% (3) posterior inference

% ===== estimate error covariance based on data =====

% estimate fluctuation scale sigma
f_prior = myrespfun_approx(prmList); % initialize using prior
[~,sigma_est] = estSigma_avg(prmList,f_prior,f_exp);

% estimate local timescale tau
tau_est = estTau_local(f_exp); 

% pack
sigma_list_init = sigma_est; % numeric array (initial estimate; will be updated)
tau_list = {tau_est}; % cell array

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

% ===== iterative sampling =====

% tempering factor is fixed to 1 (original distribution) in this demo
tmpFinal = 1;

% pack observed responses
fexp_list = {f_exp};

% for plotting
clf;
taglist = {'[f-full]','[f-approx]'};
myd0 = opts.samp.steps; % averging length scale for pdf estimate

for npan = 1:2
    
    % pack function handles for model responses
    if(npan==1)
        respfun_list = {myrespfun_full}; % full simulation
    elseif(npan==2)
        respfun_list = {myrespfun_approx}; % quadratic approx.
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
        
        subplot(2,3,1+3*(npan-1))
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
        
        subplot(2,3,2+3*(npan-1))
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
        
        myftrue = myrespfun_full(trueParams);
        
        myfsamp = myrespfun_full(psamps);
        myfsmean = mean(myfsamp,2);
        myf_atmean = myrespfun_full(prsMean);
        
        subplot(2,3,3+3*(npan-1))
        plot(tt,myftrue,'-','color',0.7*[1 1 1],'linewidth',2)
        hold on
        plot(tt,f_exp,'k-','linewidth',2)
        plot(tt,myfsmean,'b:','linewidth',2)
        plot(tt,myf_atmean,'r--','linewidth',2)
        hold off
        legend({'true f','data y','mean(f)','f(mean)'},'location','northeast')
        xlabel('time t')
        ylabel('response y')
        title([mytag,' response fit'])
        
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
                [~,sigma_est_post] = estSigma_avg(psamps,myfmodel,myfexp); % use "noshift" version
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
    
end

