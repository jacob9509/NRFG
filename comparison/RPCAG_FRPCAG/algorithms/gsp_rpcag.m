function [Lr, Sp, U, S, V] = gsp_rpcag(X, lambda, gamma, G1, param)
% Robust PCA on graphs 
% 
%   Usage: [Lr, Sp] = gsp_rpcag(X, lambda, gamma);
%          [Lr, Sp] = gsp_rpcag(X, lambda, gamma, param);
%          [Lr, Sp] = gsp_rpcag( ... );
%          [Lr, Sp] = gsp_rpcag( ... );
%          [Lr, Sp,   U, S, V] = gsp_rpcag( ... );
%
%   Input parameters: 
%       X       : Input data (matrix of double)
%       lambda  : Nuclear norm regularization paramter
%       gamma   : graph regularization
%       param   : Optional optimization parameters
%
%   Output Parameters:
%       Lr      : Low-rank part of the data
%       Sp      : Sparse part of the data
%       G       : Graph (between the line of the data )
%       U       : Part of the SVD of Lr
%       S       : Part of the SVD of Lr
%       V       : Part of the SVD of Lr
%
%   This function computes a low rank approximation of the data stored in
%   *Lr* by solving an optimization problem:
%
%   .. argmin_Lr ||Lr||* + lambda ||X - Lr||^1 + gamma tr(Lr^T L Lr)  
%
%   .. math:: argmin_Lr ||Lr||* + \lambda ||X - Lr||^1 + gamma tr(Lr^T L Lr)  
%
%   
%   If $0$ is given for *G*, the corresponding graph will
%   be computed internally. The graph construction can be tuned using the
%   optional parameter: *param.paramnn*.
%
%
%   If the number of output argument is greater than 2. The function, will
%   additionally compute a very economical SVD such that $ Lr = U S V^T$.
%
%   This function uses the UNLocBoX to be working.
%
% Author: Nauman Shahid
% Date  : 20th October 2015

%% Optional parameters

if nargin<4
    param = struct;
end

if ~isfield(param, 'verbose'), param.verbose = 1; end

if ~isfield(G1,'lmax')
    G1 = gsp_estimate_lmax(G1);
end

paraml1.verbose = param.verbose;
paraml1.y = X;
f1.prox = @(x,T) prox_l1(x,T,paraml1);
f1.eval = @(x) norm(x(:),1);

f2.grad = @(x) gamma*2*G1.L*x;
f2.eval = @(x) gamma*sum(gsp_norm_tik(G1,x));
f2.beta = 2*gamma*G1.lmax;

f3.prox = @(x,T) prox_nuclearnorm(x,lambda*T);
f3.eval = @(x) lambda*norm_nuclear(x);

param_solver.verbose = param.verbose;
param_solver.maxit = 400;%1000
param_solver.tol = 1e-5;

Lr = solvep(X,{f1,f2,f3},param_solver);
Sp = X - Lr;

%% Optional output parameters
if nargout>2
    [U, S , V] = svdecon(Lr);
end

end