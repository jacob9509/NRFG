%function demo_PCAG
%This code belongs to 2 models:
%1) "Robust PCA on graphs" (RPCAG) published in IEEE ICCV 2015. 
% 2) "Fast Robust PCA on graphs" (FRPCAG) published in IEEE Journal of
% Selected Topics in Signal Processing in 2016.
% The code provides a clustering demo for 30 subjects of the ORL
%dataset which is available at
%"http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html". 
% This dataset is also provided with this folder. We
%corrupt each of the 300 images belonging to 30 subjects (10 images each)
%with 10% random missing pixels and then apply the above two models to perform clustering:
% 1. robust PCA on graphs (RPCAG):
% min_{Lr} |X-Lr|_1 + lambda*|L|_* + gamma*tr(Lr L_1 Lr^T) 
% 2. fast robust PCA on graphs (FRPCAG): 
% min_{Lr} |X-L|_1 + gamma_1 tr(Lr^T L_1 Lr) + gamma_2 tr(Lr L_2 Lr^T)
% 3. PCA using graph total variation
% min_{Lr} |X-Lr|_1 + gamma_1 |Lr|_GTV + gamma_2 tr(Lr L_2 Lr^T)
% where X is the dataset with dimension N times Nx*Ny, and Lr is the
% low-rank, ||_GTV is the graph total variation
% N is the number of samples, Nx is the x dimension of the images and Ny is the y dimension of
% the images. For the non-image datasets, Nx should be set to 1 and Ny should be set to the number of features. 
%
% H_1 is the normalized graph Laplacian between the rows and 
% H_2 is the normalized graph Laplacian between the columns of X
%
% This code uses GSPBOX for graph construction and UNLOCBOX for the
% optimization part! DO NOT FORGET TO DOWNLOAD AND INSTALL THEM!

%%
clear
addpath('./utils/');
addpath('./gspbox/');
addpath('./utils/fast_kmeans/');
addpath('./orl_faces/');
addpath('./algorithms/');
gsp_start;
init_unlocbox;
%% get the orl dataset
% dbstop if error;
% param_data = get_orl_data_2();%一行为一个样本，有10%的随机像素缺失 0-255
param_data = get_orl_data();
param_data.C = param_data.C/255;
param_data.X = param_data.X/255;
select = round(rand(1,30)*300);
W = repmat(param_data.C(select,:),[10,1]);
W = W';
c = 40; % num of atoms
t = 40;
params.data=W;
params.Tdata=t;
params.dictsize=c;
params.memusage='high';
[X,~,~]=ksvd(params,'');
F1=orth(X);

params.data=W';
params.Tdata=40;
params.dictsize=40;
params.memusage='high';
[Y,~,~]=ksvd(params,'');
F2=orth(Y);

clear X Y params c t W select
save F1.mat F1
save F2.mat F2
%% normalize
% normalize the dataset to zero mean and unit standard deviation along the
% features. this transformation should be applied after corrupting the
% images.
param_data_g = zero_means(param_data);%沿特征将数据集规格化为零平均值和单位标准差
%每行是一个样本，样本数为N
%每列是一个特征，特征数为Nx*Ny
%% parameters for graph construction. (see GSPBOX for details)
param_graph.use_flann = 0; % to use the fast C++ library for construction of huge graphs.
param_graph.k = 7;

%% create the graphs
% G1 = gsp_nn_graph(param_data.X,param_graph);
% G2 = gsp_nn_graph(param_data.X',param_graph);
G1_g = gsp_nn_graph(param_data_g.X,param_graph);
G2_g = gsp_nn_graph(param_data_g.X',param_graph);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Robust PCA on Graphs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lambda = sqrt(max([param_data.N param_data.Nx*param_data.Ny]));  % nuclear norm 
gamma = 1;  % graph regularization
[L_rpcag, info_rpcag] = gsp_rpcag(param_data_g.X, lambda, gamma, G1_g,[]);
clear lambda gamma info_rpcag
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Fast Robust PCA on Graphs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.maxit = 500;
param.tol = 1e-4;
param.verbose = 2;

gamma1 = 1; %graph regularization along rows
gamma2 = 1; %graph regularization along columns
[L_frpcag, info_frpcag] = gsp_frpcaog_2g(param_data_g.X, gamma1, gamma2, G1_g, G2_g,param);
clear param gamma1 gamma2 info_frpcag
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PCA using graph total variation
gamma1 = 3; %graph regularization along rows
gamma2 = 1; %graph regularization along columns
L_pcagtv = gsp_gpcatv_2g(param_data.X, gamma1, gamma2, G1, G2);

%% PCP
raw = param_data.X;
raw = raw';
L_pcp=pcp(raw);
L_pcp = L_pcp';
%% PCPF
L_pcpf=pcpf(raw,zeros(size(raw)),F1,F2);
L_pcpf = L_pcpf';
%% IRPCA
r = 30;%10
extra = zeros(size(F1,1),r-size(F1,2));
F1_r = [F1,extra];
extra = zeros(size(F2,1),r-size(F1,2));
F2_r = [F2,extra];
[L_irpca,~,~,~,~] = irpca_real(raw,r,F1_r',F2_r');
L_irpca = L_irpca';
%% altproj
[L_altproj,~,~,~] = ncrpca(raw,r);
L_altproj =L_altproj';
%% IRCUR
para.beta_init = 2*max(max(abs(raw)));%2
para.beta      = para.beta_init;
para.tol       = 1e-5;
para.con       = 3;
para.resample  = true;
[C1, pinv_U1, R1, ircur_r_timer, ircur_r_err] = IRCUR(raw, r, para);
L_ircur = C1 * pinv_U1 * R1;
% figure,imshow(reshape(L_ircur(:,10),[56,46]))%[48,42]
L_ircur = L_ircur';
clear para ircur_r_timer ircur_r_err C1 pinv_U1 R1
%% rpca-gd
params.step_const = 0.5; % step size parameter for gradient descent
params.max_iter   = 30;  % max number of iterations
params.tol        = 2e-4;% stop when ||Y-UV'-S||_F/||Y||_F < tol 2e-4
alpha = 0.1;
gamma = 0.3;%0.9-1.4-1.7-...-3.1-4;  %2,1.5,2,1
alpha_bnd = gamma*alpha; 
% alpha_bnd = 0.1;
[U_fast, V_fast] = rpca_gd(raw, r, alpha_bnd, params);
L_fastrpca=U_fast*V_fast';
clear params U_fast V_fast  alpha_bnd alpha gamma alpha_bnd
L_fastrpca = L_fastrpca'; 
%% NRF
param_graph.use_flann = 0; 
param_graph.k = 4;
G1 = gsp_nn_graph(raw',param_graph);
L_init = rand(size(raw));
lambda.lambda1 = 0.3/sqrt(max(size(raw)));%
lambda.lambda2 = 0.5/sqrt(max(size(raw)));%
lambda.lambda3 = 1; % 0.14
[L_nrf, L_in, N_nrf, E_nrf] = NRFNG_1(raw,F1,F2,G1.L,L_init,lambda);
% figure,imshow(reshape(L_nrf(:,10),[56,46]))
clear L_in N_nrf E_nrf
L_nrf = L_nrf';
%% clustering comparison

% [err_rpcag, err_frpcag, err_pcagtv] = clustering_quality(L_rpcag, L_frpcag, L_pcagtv , param_data)
[err_rpcag, err_frpcag] = clustering_quality_g(L_rpcag, L_frpcag, param_data_g)
[err_altproj, err_rpcagd, err_pcp] = clustering_quality(L_altproj, L_fastrpca, L_pcp , param_data)
[err_nrf, err_ircur, err_irpca] = clustering_quality(L_nrf, L_ircur, L_irpca, param_data)
[err_pcpf, err_pcpf, err_irpca] = clustering_quality(L_pcpf, L_pcpf, L_irpca, param_data)

