function [L, L_in, N, E] = NRFNG(R, X, Y, G, lambda)

lambda1 = lambda.lambda1;
lambda2 = lambda.lambda2;
lambda3 = lambda.lambda3;

[m,n]=size(R);
M_FNorm = norm(R,'fro');

gamma1 = 4;
gamma2 = 4;
gamma3 = 4;

innerLoopDisplay = true;
innertol = 1e-7;
outertol = 1e-5;
innerMaxIter = 200;
outerMaxIter = 3;
rho = 1.3;
mu = 1e-3;
mu_bar = mu * 1e7;

L = repmat(mean(R,2),[1,n]);%zeros(m,n)
E = sparse(m,n);
N = zeros(m,n);
Ge = L;

Y1_0 = zeros(m,n);
Y2_0 = zeros(m,n);
Y3_0 = zeros(m,n);

ObjFunValues = zeros(outerMaxIter + 1, 1);
% -------------------------- Outer Loop ------------------------ %
for outiter = 1:outerMaxIter
    sigma_L = svd(L, 'econ');
    varLambda_L = max(1 - sigma_L / gamma1, 0);
    W_E = max(1 - abs(E) / gamma3, 0);
    W_N = max(1 - abs(N) / gamma2, 0);
    V_1 = Y1_0;
    V_2 = Y2_0;
    V_3 = Y3_0;
    % -------------------- Inner Loop --------------------- %
    for i = 1: innerMaxIter
        H = X'*(L-N+V_2/mu)*Y;
        L = GeneralizedSVT((X*H*Y'+N+Ge+R-E-(V_2+V_3-V_1)/mu)/3, varLambda_L, 1 / (3*mu));
        E = GeneralizedThresholding(R - L + V_1 / mu, W_E, lambda2 / mu);
        Ge = (mu*(L+V_3/mu))/(mu*eye(n)+2*lambda3*G);
        N = GeneralizedThresholding(L- X*H*Y' + V_2 / mu, W_N, lambda1 / mu);

        leq1 = R - L - E;
        leq2 = L - X*H*Y' - N;
        leq3 = L - Ge;
        V_1 = V_1 + mu * leq1;
        V_2 = V_2 + mu * leq2;
        V_3 = V_3 + mu * leq3;
        mu = min(mu*rho, mu_bar);
        stopCriterion=max([norm(leq1, 'fro')/M_FNorm,norm(leq2, 'fro')/M_FNorm,norm(leq3, 'fro')/M_FNorm]);
        if stopCriterion < innertol
            break;
        end
        if innerLoopDisplay && mod(i, 10) == 0
            disp(['\t#svd ' num2str(2*i) ' r(L) ' num2str(rank(L))...
                ' r(N) ' num2str(rank(N))...
                ' r(L_in) ' num2str(rank(H))...
                ' stopCriterion ' num2str(stopCriterion)]);
        end 
    end
    objective = EvaluateMCP(sigma_L, 1, gamma1) + lambda1*EvaluateMCP(N, 1, gamma2) + lambda2 * EvaluateMCP(E, 1, gamma3);
    ObjFunValues(outiter+1) = objective;
    if abs(ObjFunValues(outiter) - objective) < outertol
       break;
    end
end
L_in = X*H*(Y');
end


function [A_new] = GeneralizedThresholding(A_old, W_input, tau)
A_sign = sign(A_old);
A_new = abs(A_old) - tau * abs(W_input);
A_new(A_new < 0) = 0;
A_new = A_new .* A_sign;
end

function [X_new] = GeneralizedSVT(X_input, Lambda, tau)
[UX, SigmaX, VX] = svd(X_input, 'econ');
SigmaX = diag(SigmaX);
svp = length(find(SigmaX > tau));
sigma_new = GeneralizedThresholding(SigmaX(1:svp), Lambda(1:svp), tau);
X_new = UX(:, 1:svp) * diag(sigma_new) * VX(:, 1:svp)';
end

function [mcp] = EvaluateMCP(x, lambda, gamma)
x = abs(x);
phi = x;
idx = (x > lambda * gamma);
phi(idx) = lambda^2 * gamma / 2;
phi(~idx) = lambda * x(~idx) - x(~idx).^2 / 2 / gamma;
mcp = sum(sum(phi));
end