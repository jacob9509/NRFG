clear
clc
addpath(genpath(pwd))
%% load data 
load CAVIAR1.mat
r = 1;
Isize = size(image);
imSize = [Isize(1),Isize(2),Isize(3)];
raw = reshape(image,[Isize(1)*Isize(2)*Isize(3),Isize(4)]);
clear image
grandtruth = imresize(double(imread('GT_CAVIAR1.png'))/255,[Isize(1),Isize(2)]);
grandtruthm = repmat(reshape(grandtruth,[Isize(1)*Isize(2)*Isize(3),1]),[1,Isize(4)]);

%% calculate side information 
W = repmat(raw(:,[1:5,256:260]),[1,61]);
c = 60; 
t = 40;
params.data = W; 
params.Tdata = t;
params.dictsize = c;
params.memusage = 'high';
[X,~,~] = ksvd(params,'');
F1 = orth(X);
F2 = eye(size(raw,2));
clear X params c t W

%% NRFG
param_graph.use_flann = 0; 
param_graph.k = 16;
G1 = gsp_nn_graph(raw',param_graph);
lambda.lambda1 = 0.85/sqrt(max(size(raw)));
lambda.lambda2 = 0.2/sqrt(max(size(raw))); 
lambda.lambda3 = 5;
L_init = repmat(mean(raw,2),[1,Isize(4)]);
[L_nrf,L_in, N_nrf, E_nrf] = NRFNG(raw,F1,F2,G1.L,L_init,lambda);
MPSNR = PSNRframewise(grandtruthm, L_nrf)
num = 16;
figure,imshow(reshape(raw(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
figure,imshow(grandtruth,'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
figure,imshow(reshape(abs(N_nrf(:,num)),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
figure,imshow(reshape(abs(L_nrf(:,num)),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
figure,imshow(reshape(abs(E_nrf(:,num)),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
clear lambda param_graph G1

%% PCP
[L_pcp,S_pcp]=pcp(raw);
MPSNR = PSNRframewise(grandtruthm, L_pcp)
figure,imshow(reshape(L_pcp(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
figure,imshow(reshape(abs(S_pcp(:,num)),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal

%% PCPF
[L_pcpf,S_pcpf]=pcpf(raw,zeros(size(raw)),F1,F2);
MPSNR = PSNRframewise(grandtruthm, L_pcpf)
figure,imshow(reshape(L_pcpf(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
figure,imshow(reshape(abs(S_pcpf(:,num)),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal

%% RPCAG
param_graph.use_flann = 0; % to use the fast C++ library for construction of huge graphs.
param_graph.k = 16;
param_data=raw';
G1 = gsp_nn_graph(param_data,param_graph);
lambda = sqrt(max(size(param_data)));  % nuclear norm 
gamma = 1;  % graph regularization
[L_rpcag, S_rpcag] = gsp_rpcag(param_data, lambda, gamma, G1,[]);
MPSNR = PSNRframewise(grandtruthm, L_rpcag')
L_rpcag = L_rpcag';
S_rpcag = S_rpcag';
clear lambda gamma info_rpcag
figure,imshow(reshape(L_rpcag(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
figure,imshow(reshape(abs(S_rpcag(:,num)),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal

%% FPRCAG
param.maxit = 500;
param.tol = 1e-4;
param.verbose = 2;
gamma1 = 1; %graph regularization along rows
gamma2 = 1; %graph regularization along columns
G2 = gsp_nn_graph(param_data',param_graph);
[L_frpcag,S_frpcag] = gsp_frpcaog_2g(param_data, gamma1, gamma2, G1, G2, param);
MPSNR = PSNRframewise(grandtruthm, L_frpcag')
clear gamma1 gamma2 param G2 param_graph
L_frpcag = L_frpcag';
S_frpcag = S_frpcag';
figure,imshow(reshape(L_frpcag(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
figure,imshow(reshape(abs(S_frpcag(:,num)),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal

%% IRCUR-R
para.beta_init = 2*max(max(abs(raw)));%2
para.beta      = para.beta_init;
para.tol       = 1e-5;
para.con       = 3;
para.resample  = true;
[C1, pinv_U1, R1] = IRCUR(raw, r, para);
L_ircur = C1 * pinv_U1 * R1;
S_ircur = raw - L_ircur;
MPSNR = PSNRframewise(grandtruthm, L_ircur)
clear para ircur_r_timer ircur_r_err C1 pinv_U1 R1
figure,imshow(reshape(L_ircur(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
figure,imshow(reshape(abs(S_ircur(:,num)),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal

%% IRPCA-IHT
[L_irpca,~,S_irpca,~,~] = irpca_real(raw,r,F1',F2');
MPSNR = PSNRframewise(grandtruthm, L_irpca)
figure,imshow(reshape(L_irpca(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
figure,imshow(reshape(abs(S_irpca(:,num)),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal

%% AltProj
[L_altproj,~,~,~] = ncrpca(raw,r);
S_altproj = raw - L_altproj;
MPSNR = PSNRframewise(grandtruthm, L_altproj)
figure,imshow(reshape(L_altproj(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
figure,imshow(reshape(abs(S_altproj(:,num)),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal

%% RPCA-GD
params.step_const = 0.5; % step size parameter for gradient descent
params.max_iter   = 30;  % max number of iterations
params.tol        = 2e-4;% stop when ||Y-UV'-S||_F/||Y||_F < tol 2e-4
alpha = 0.1;
gamma = 2;%0.9-1.4-1.7-...-3.1-4;  2,1.5,2,1
alpha_bnd = gamma*alpha; 
% alpha_bnd = 0.1;
[U_fast, V_fast] = rpca_gd(raw, r, alpha_bnd, params);
L_gd=U_fast*V_fast';
S_gd = raw -L_gd;
MPSNR = PSNRframewise(grandtruthm, L_gd)
clear params U_fast V_fast  alpha_bnd alpha gamma alpha_bnd
figure,imshow(reshape(L_gd(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
figure,imshow(reshape(abs(S_gd(:,num)),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal