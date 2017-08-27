function [X_opt, iter] = t_admmAXB(A, B, X, M, omega, opts)

DISPLAY_EVERY = 10;
DEBUG = 0;

rho = 1.1; 
mu = 1e-3; 
max_mu = 1e10; 
tol = 1e-4;
max_iter = 500;

if ~exist('opts', 'var')
    opts = [];
end
if isfield(opts, 'rho');        rho = opts.rho;              end
if isfield(opts, 'mu');         mu = opts.mu;                end
if isfield(opts, 'max_mu');     max_mu = opts.max_mu;        end
if isfield(opts, 'tol');        tol = opts.tol;              end
if isfield(opts, 'max_iter');   max_iter = opts.max_iter;    end
if isfield(opts, 'DEBUG');      DEBUG = opts.DEBUG;          end

W = X;
Y = X;
AtB = tprod(tran(A), B);
norm_M = norm(M(:));
% dim = size(M);
% omegac = setdiff(1:prod(dim), omega);

for k = 1 : max_iter
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
    if DEBUG && mod(k, DISPLAY_EVERY) == 0
        obj = tnn;
        err = norm(dY(:));
        fprintf(['iter %d,\t delta=%.4f,\t mu=%f,\t rank=%d,' ...
            '\t obj=%.4f,\t err=%.4f\n'], ...
            k, delta, mu, trank, obj, err);
%             disp(['iter ' num2str(k) ', delta= ' num2str(delta) ...
%                   ', mu=' num2str(mu) ', rank=' num2str(trankX) ...
%                   ', obj=' num2str(obj) ', err=' num2str(err)]); 
    end
    if delta < tol
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