clear
clc
addpath(genpath(pwd))
%% 
load yaleB02.mat
% load yaleB04.mat
% load yaleB06.mat
raw = double(img)./255;
Isize = size(raw);
imSize = [192 168];
r = 9;
num = 15;
clear img
%%
% load yaleB04_selected16.mat
% load yaleB02_selected16.mat
load yaleB06_selected16.mat
W=repmat(img,[1,4])./255;
clear img
c = 40; % num of atoms
t = 30;
params.data=W;
params.Tdata=t;
params.dictsize=c;
params.memusage='high';
[X,~,~]=ksvd(params,'');
F1=orth(X);

params.data=W';
params.Tdata=40;
params.dictsize=30;
params.memusage='high';
[Y,~,~]=ksvd(params,'');
F2=orth(Y);

clear X Y params c t W
%% NRFG
param_graph.use_flann = 0; 
param_graph.k = 1;
G1 = gsp_nn_graph(raw',param_graph);
L_init = repmat(mean(raw,2),[1,size(raw,2)]);
lambda.lambda1 = 0.5/sqrt(max(size(raw)));
lambda.lambda2 = 0.3/sqrt(max(size(raw)));
lambda.lambda3 = 2; 
[L_nrf, L_in, N_nrf, E_nrf] = NRFNG(raw,F1,F2,G1.L,L_init,lambda);
figure,imshow(reshape(raw(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,0,168,192]);
axis normal
figure,imshow(reshape(L_nrf(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,0,168,192]);
axis normal
%% PCP
L_pcp=pcp(raw);
figure,imshow(reshape(L_pcp(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,0,168,192]);
axis normal
%% PCPF
L_pcpf=pcpf(raw,zeros(size(raw)),F1,F2);
figure,imshow(reshape(L_pcpf(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,0,168,192]);
axis normal
%% RPCAG
param_graph.use_flann = 0; % to use the fast C++ library for construction of huge graphs.
param_graph.k = 7;
param_data=raw';
G1 = gsp_nn_graph(param_data,param_graph);
lambda = sqrt(max(size(param_data)));  % nuclear norm 
gamma = 1;  % graph regularization
[L_rpcag, ~] = gsp_rpcag(param_data, lambda, gamma, G1,[]);
L_rpcag = L_rpcag';
clear lambda gamma info_rpcag
figure,imshow(reshape(L_rpcag(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,0,168,192]);
axis normal
%% FRPCAG
param.maxit = 500;
param.tol = 1e-4;
param.verbose = 2;
gamma1 = 1; %graph regularization along rows
gamma2 = 1; %graph regularization along columns
G2 = gsp_nn_graph(param_data',param_graph);
[L_frpcag, ~] = gsp_frpcaog_2g(param_data, gamma1, gamma2, G1, G2, param);
clear gamma1 gamma2 param G2 param_graph
L_frpcag = L_frpcag';
figure,imshow(reshape(L_frpcag(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,0,168,192]);
axis normal
%% IRPCA-IHT
[L_irpca,~,~,~,~] = irpca_real(raw,r,F1',F2');
clear W_t_IN S_t_IN
figure,imshow(reshape(L_irpca(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,0,168,192]);
axis normal
%% IRCUR-R
para.beta_init = 2*max(max(abs(raw)));%2
para.beta      = para.beta_init;
para.tol       = 1e-5;
para.con       = 3;
para.resample  = true;
[C1, pinv_U1, R1] = IRCUR(raw, r, para);
L_ircur = C1 * pinv_U1 * R1;
clear para ircur_r_timer ircur_r_err C1 pinv_U1 R1
figure,imshow(reshape(L_ircur(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,0,168,192]);
axis normal
%% AltProj
L_altproj = ncrpca(raw,r);
figure,imshow(reshape(L_altproj(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,0,168,192]);
axis normal
%% RPCA-GD
params.step_const = 0.5; % step size parameter for gradient descent
params.max_iter   = 30;  % max number of iterations
params.tol        = 2e-4;% stop when ||Y-UV'-S||_F/||Y||_F < tol 2e-4
alpha = 0.1;
gamma = 0.5;
alpha_bnd = gamma*alpha; 
[U_fast, V_fast] = rpca_gd(raw, r, alpha_bnd, params);
L_fastrpca=U_fast*V_fast';
clear params U_fast V_fast  alpha_bnd alpha gamma alpha_bnd
figure,imshow(reshape(L_fastrpca(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,0,168,192]);
axis normal