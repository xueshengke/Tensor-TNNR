function [X_opt, iter] = tnn_admm(A, B, X, M, omega, opts)

DISPLAY_EVERY = 10;
rho = 1.05; 
mu = 1e-3; 
max_mu = 1e10; 
admm_tol = 1e-4;
admm_iter = 500;

if ~exist('opts', 'var'),   opts = [];  end

if isfield(opts, 'rho');        rho = opts.rho;              end
if isfield(opts, 'mu');         mu = opts.mu;                end
if isfield(opts, 'max_mu');     max_mu = opts.max_mu;        end
if isfield(opts, 'admm_tol');   admm_tol = opts.admm_tol;    end
if isfield(opts, 'admm_iter');	admm_iter = opts.admm_iter;  end

W = X;
Y = X;
AtB = tprod(tran(A), B);
norm_M = norm(M(:));

for k = 1 : admm_iter
    last_X = X;
    last_W = W;
    % update X
    [X, tnn, trank] = prox_tnn(W - Y/mu, 1/mu); 
    % update W
    W = X + (Y + AtB) / mu;
    W(omega) = M(omega);

    dY = X - W;    
    dX = norm(vec(last_X - X)) / norm_M;
    dW = norm(vec(last_W - W)) / norm_M;
    delta = max(dX, dW);
    if mod(k, DISPLAY_EVERY) == 0
        obj = tnn;
        err = norm(dY(:));
        fprintf(['iter %d,\t delta=%.4f,\t mu=%.4f,\t t-rank=%d,' ...
            '\t obj=%.4f,\t err=%.4f\n'], k, delta, mu, trank, obj, err);
    end
    if delta < admm_tol
        break;
    end 
    % update Y
    Y = Y + mu * dY;
    % update mu
    mu = min(rho*mu, max_mu);
end

X_opt = X;
iter = k;

end