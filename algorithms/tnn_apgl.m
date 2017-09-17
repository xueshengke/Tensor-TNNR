function [X_opt, iter] = tnn_apgl(A, B, X, M, omega, opts)
%--------------------------------------------------------------------------
% Shengke Xue, Zhejiang University, September 2017. 
% Contact information: see readme.txt.
%
% Xue et al. (2018) Tensor-TNN paper, International Conference on ***.
%--------------------------------------------------------------------------
% 
% Input:
%       A       -    orthogonal tensor by tensor SVD
%       B       -    orthogonal tensor by tensor SVD
%       X       -    incomplete tensor
%       M       -    original tensor
%       omega   -    index of the known elements
%       opts    -    struct contains parameters
%
% Output:
%       X_opt   -    recovered tensor
%       iter    -    numebr of iterations
%--------------------------------------------------------------------------

DISPLAY_EVERY = 10;
lambda = 1e-2;
opts.apgl_tol = 1e-4;
opts.apgl_iter = 200;

if ~exist('opts', 'var'),   opts = [];  end

if isfield(opts, 'lambda');     lambda = opts.lambda;        end
if isfield(opts, 'apgl_tol');   apgl_tol = opts.apgl_tol;    end
if isfield(opts, 'apgl_iter');  apgl_iter = opts.apgl_iter;  end

Y = X;
t = 1;
AtB = tprod(tran(A), B);
norm_M = norm(M(:));
mask = zeros(size(X));
mask(omega) = 1;

for k = 1 : apgl_iter
    last_X = X;
    last_t = t;
    % update X
    X_temp = Y + t * (AtB - lambda * (Y - M) .* mask);
    [X, tnn, trank] = prox_tnn(X_temp, t); 
    % update t
    t = (1 + sqrt(1+4*t*t)) / 2;
    % update Y
    Y = X + (last_t - 1) / t * (X - last_X);
  
    dX = norm(vec(last_X - X)) / norm_M;
    delta = max(dX);
    if mod(k, DISPLAY_EVERY) == 0
        obj = tnn;
        err = norm(dX(:));
        fprintf(['iter %d,\t delta=%.4f,\t mu=%.4f,\t t-rank=%d,' ...
            '\t obj=%.4f,\t err=%.4f\n'], k, delta, lambda, trank, obj, err);
    end
    if delta < apgl_tol
        break;
    end 
end

X_opt = X;
iter = k;

end