function [X_opt, iter] = t_apglAXB(A, B, X, M, omega, opts)

DISPLAY_EVERY = 10;
DEBUG = 0;

lambda = 1e-2;
opts.apgl_tol = 1e-4;
opts.apgl_iter = 200;
% rho = 1.1; 
% mu = 1e-3; 
% max_mu = 1e10; 
% tol = 1e-4;
% max_iter = 500;

if ~exist('opts', 'var')
    opts = [];
end
% if isfield(opts, 'rho');        rho = opts.rho;              end
if isfield(opts, 'lambda');     lambda = opts.lambda;        end
% if isfield(opts, 'mu');         mu = opts.mu;                end
% if isfield(opts, 'max_mu');     max_mu = opts.max_mu;        end
if isfield(opts, 'apgl_tol');   apgl_tol = opts.apgl_tol;    end
if isfield(opts, 'apgl_iter');  apgl_iter = opts.apgl_iter;  end
if isfield(opts, 'DEBUG');      DEBUG = opts.DEBUG;          end

% W = X;
Y = X;
t = 1;
AtB = tprod(tran(A), B);
norm_M = norm(M(:));
mask = zeros(size(X));
mask(omega) = 1;
% dim = size(M);
% omegac = setdiff(1:prod(dim), omega);

for k = 1 : apgl_iter
    last_X = X;
    last_Y = Y;
    last_t = t;
    % update X
    X_temp = Y + t * (AtB - lambda * (Y - M) .* mask);
    [X, tnn, trank] = prox_tnn(X_temp, t); 
    % update t
    t = (1 + sqrt(1+4*t*t)) / 2;
    % update Y
    Y = X + (last_t - 1) / t * (X - last_X);
  
    dX = norm(vec(last_X - X)) / norm_M;
    dY = norm(vec(last_Y - Y)) / norm_M;
    delta = max(dX, dY);
    if DEBUG && mod(k, DISPLAY_EVERY) == 0
        obj = tnn;
        fprintf(['iter %d,\t delta=%.4f,\t mu=%f,\t rank=%d,' ...
            '\t obj=%.4f\n'], ...
            k, delta, lambda, trank, obj);
%             disp(['iter ' num2str(k) ', delta= ' num2str(delta) ...
%                   ', mu=' num2str(mu) ', rank=' num2str(trankX) ...
%                   ', obj=' num2str(obj) ', err=' num2str(err)]); 
    end
    if delta < apgl_tol
        break;
    end 
    % update Y
%     Y = Y + mu * dY;
    % update mu
%     mu = min(rho*mu, max_mu);
end

X_opt = X;
iter = k;

end