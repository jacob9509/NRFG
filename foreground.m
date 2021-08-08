clear
clc
addpath(genpath(pwd))
%%
r = 1;

load Curtain_1180.mat;
begin = 22751;
file_path = strcat(pwd,'\data\groundtruthcurtain\');

namelist = dir(strcat(file_path,'*.bmp'));
len = length(namelist);
sp = zeros(1,len);
Isize = size(image);
imSize = [Isize(1),Isize(2),Isize(3)];
for i=1:len
    temp = namelist(i).name;
    sp(i)=str2double(temp(isstrprop(namelist(i).name,'digit')))-begin+1;
end
s=cell(len,1);
for i=1:len
    s{i}=imbinarize(rgb2gray(imresize(double(imread(strcat(file_path,namelist(i).name)))/255,[Isize(1),Isize(2)])));
end
raw = reshape(image,[Isize(1)*Isize(2)*Isize(3),Isize(4)]);
clear image srcDir namelist i file_path namelist temp

%% side information
W = repmat(raw(:,[254:258,556:560]),[1,floor(Isize(4)/10)]);
c = 60; 
t = 40;
params.data = W; 
params.Tdata = t;
params.dictsize = c;
params.memusage = 'high';
[X,~,~] = ksvd(params,'');
F1 = orth(X);
F2 = eye(size(raw,2));
clear X params c t W side_matrix

%% 
index = 20;
num = sp(index);
%% NRFG
param_graph.use_flann = 0; 
param_graph.k = 1;
G1 = gsp_nn_graph(raw',param_graph);
L_init = repmat(mean(raw,2),[1,Isize(4)]);
lambda.lambda1 = 0.4/sqrt(max(size(raw)));% 1.5  0.05
lambda.lambda2 = 0.35/sqrt(max(size(raw))); %0.17 1 0.18
lambda.lambda3 = 2; % 0.14
[L_nrf,L_in, N_nrf, E_nrf] = NRFNG_1(raw,F1,F2,G1.L,L_init,lambda);
F_nrf_list=zeros(len,1);
for i=1:len
    J=imbinarize(rgb2gray(abs(reshape(E_nrf(:,sp(i)),imSize))));
    [val,~] = findFMeasure(J, s{i});
    F_nrf_list(i,1)=val;
end
mean(F_nrf_list)
clear J val i
figure,imshow(reshape(raw(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
figure,imshow(s{index},'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
figure,imshow(reshape(L_nrf(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
figure,imshow(reshape(abs(N_nrf(:,num)),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
figure,imshow(imbinarize(rgb2gray(abs(reshape(E_nrf(:,num),imSize)))),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
%% PCP
[L_pcp,S_pcp]=pcp(raw);
F_pcp_list=zeros(len,1);
for i=1:len
    J=imbinarize(rgb2gray(abs(reshape(S_pcp(:,sp(i)),imSize))));
    [val,~] = findFMeasure(J, s{i});
    F_pcp_list(i,1)=val;
end
mean(F_pcp_list)
clear J val i
figure,imshow(reshape(L_pcp(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
figure,imshow(imbinarize(rgb2gray(abs(reshape(S_pcp(:,num),imSize)))),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
%% PCPF
[L_pcpf,S_pcpf]=pcpf(raw,zeros(size(raw)),F1,F2);
F_pcpf_list=zeros(len,1);
for i=1:len
    J=imbinarize(rgb2gray(abs(reshape(S_pcpf(:,sp(i)),imSize))));
    [val,~] = findFMeasure(J, s{i});
    F_pcpf_list(i,1)=val;
end
mean(F_pcpf_list)
clear J val i
figure,imshow(imbinarize(rgb2gray(abs(reshape(S_pcpf(:,num),imSize)))),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
figure,imshow(reshape(L_pcpf(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
%% RPCAG
param_graph.use_flann = 0; % to use the fast C++ library for construction of huge graphs.
param_graph.k = 9;
param_data=raw';
G1 = gsp_nn_graph(param_data,param_graph);
lambda = sqrt(max(size(param_data)));  % nuclear norm 
gamma = 1;  % graph regularization
[L_rpcag, S_rpcag] = gsp_rpcag(param_data, lambda, gamma, G1,[]);
L_rpcag = L_rpcag';
S_rpcag = S_rpcag';
clear lambda gamma 
F_rpcag_list=zeros(len,1);
for i=1:len
    J=imbinarize(rgb2gray(abs(reshape(S_rpcag(:,sp(i)),imSize))));
    [val,~] = findFMeasure(J, s{i});
    F_rpcag_list(i,1)=val;
end
mean(F_rpcag_list)
clear J val i
figure,imshow(imbinarize(rgb2gray(abs(reshape(S_rpcag(:,num),imSize)))),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
figure,imshow(reshape(L_rpcag(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
%% FRPCAG
param.maxit = 500;
param.tol = 1e-4;
param.verbose = 2;
gamma1 = 1; %graph regularization along rows
gamma2 = 1; %graph regularization along columns
G2 = gsp_nn_graph(param_data',param_graph);
[L_frpcag, S_frpcag] = gsp_frpcaog_2g(param_data, gamma1, gamma2, G1, G2, param);
clear gamma1 gamma2 param G2 param_graph
L_frpcag = L_frpcag';
% S_frpcag = raw-L_frpcag;
S_frpcag = S_frpcag';
F_frpcag_list=zeros(len,1);
for i=1:len
    J=imbinarize(rgb2gray(abs(reshape(S_frpcag(:,sp(i)),imSize))));
    [val,~] = findFMeasure(J, s{i});
    F_frpcag_list(i,1)=val;
end
mean(F_frpcag_list)
clear J val i
figure,imshow(imbinarize(rgb2gray(abs(reshape(S_frpcag(:,num),imSize)))),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
figure,imshow(reshape(L_frpcag(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
%% AltProj
[L_altproj,~,~,~] = ncrpca(raw,r);
S_altproj = raw - L_altproj;
F_altproj_list=zeros(len,1);
for i=1:len
    J=imbinarize(rgb2gray(abs(reshape(S_altproj(:,sp(i)),imSize))));
    [val,~] = findFMeasure(J, s{i});
    F_altproj_list(i,1)=val;
end
mean(F_altproj_list)
clear J val i
figure,imshow(imbinarize(rgb2gray(abs(reshape(S_altproj(:,num),imSize)))),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
figure,imshow(reshape(L_altproj(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
%% IRCUR-R
para.beta_init = 2*max(max(abs(raw)));%2
para.beta      = para.beta_init;
para.tol       = 1e-5;
para.con       = 3;
para.resample  = true;
[C1, pinv_U1, R1, ~, ~] = IRCUR(raw, r, para);
L_ircur = C1 * pinv_U1 * R1;
S_ircur = raw - L_ircur;
clear para C1 pinv_U1 R1
F_ircur_list=zeros(len,1);
for i=1:len
    J=imbinarize(rgb2gray(abs(reshape(S_ircur(:,sp(i)),imSize))));
    [val,~] = findFMeasure(J, s{i});
    F_ircur_list(i,1)=val;
end
mean(F_ircur_list)
clear J val i
figure,imshow(reshape(L_ircur(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
figure,imshow(imbinarize(rgb2gray(abs(reshape(S_ircur(:,num),imSize)))),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
%% IRPCA
[L_irpca,~,S_irpca,~,~] = irpca_real(raw,r,F1',F2');
F_irpca_list=zeros(len,1);
for i=1:len
    J=imbinarize(rgb2gray(abs(reshape(S_irpca(:,sp(i)),imSize))));
    [val,~] = findFMeasure(J, s{i});
    F_irpca_list(i,1)=val;
end
mean(F_irpca_list)
clear J val i
figure,imshow(imbinarize(rgb2gray(abs(reshape(S_irpca(:,num),imSize)))),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
figure,imshow(reshape(L_irpca(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
%% RPCA-GD
params.step_const = 0.5; % step size parameter for gradient descent
params.max_iter   = 30;  % max number of iterations
params.tol        = 2e-4;% stop when ||Y-UV'-S||_F/||Y||_F < tol 2e-4
alpha = 0.1;
gamma = 2;%0.9-1.4-1.7-...-3.1-4;  %2,1.5,2,1
alpha_bnd = gamma*alpha; % alpha_bnd = 0.1;
[U_fast, V_fast] = rpca_gd(raw, r, alpha_bnd, params);
L_gd = U_fast*V_fast';
S_gd = raw -L_gd;
clear params U_fast V_fast  alpha_bnd alpha gamma alpha_bnd
F_gd_list=zeros(len,1);
for i=1:len
    J=imbinarize(rgb2gray(abs(reshape(S_gd(:,sp(i)),imSize))));
    [val,~] = findFMeasure(J, s{i});
    F_gd_list(i,1)=val;
end
mean(F_gd_list)
clear J val i
figure,imshow(imbinarize(rgb2gray(abs(reshape(S_gd(:,num),imSize)))),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal
figure,imshow(reshape(L_gd(:,num),imSize),'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,imSize(1),0,imSize(2)]);
axis normal