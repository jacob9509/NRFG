function [L_t,W_t,S_t,iters,frob_err] = irpca_real(M,r,F1,F2)

frob_err(1) = inf; t = 1; idx = []; MAX_ITER = 31; EPS = 1e-3;
thresh_const = 0;
thresh = 5*abs(max(M(:)));
thresh_red = 0.6;% seg 0.8 real 0.6 raw 0.9// 0.4-0.1-0.5
D_t = M;
idx = unique([find(abs(D_t) > thresh); idx]);
S_t = zeros(size(M));
S_t(idx) = D_t(idx);
if max(idx(:))==0
    idx = [];
end
while frob_err(t)/norm(M, 'fro')>=EPS && t<MAX_ITER % convergence check
    %if ~mod(t, 10)
        %fprintf('Iter no. %d, nnz(S_t) = %d, nnz(S) = %d \n',t,nnz(S_t),nnz(S));
    %    fprintf('Iter no. %d, nnz(S_t) = %d\n',t,nnz(S_t));
    %end
    t = t+1;
    [U_t,Sig_t,V_t] = svds(pinv(F1')*(M-S_t)*pinv(F2),r);
    W_t = U_t(:,1:r)*Sig_t(1:r,1:r)*V_t(:,1:r)';
    L_t = F1'*W_t*F2;
    D_t = M-L_t;
    thresh = thresh_red*thresh+thresh_const;
    idx = unique([find(abs(D_t) > thresh); idx]);
    S_t(idx) = D_t(idx);
    frob_err(t) = norm(M-(L_t+S_t), 'fro');
end
S_t(abs(S_t)<EPS) = 0; % threshold to remove misc noise
iters = length(frob_err)-1;
