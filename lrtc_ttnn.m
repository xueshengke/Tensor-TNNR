function [X_best_rec, result] = lrtc_ttnn(M, omega, opts)

% Solve the Low-Rank Tensor Completion (LRTC) based on Truncated Tensor 
% Nuclear Norm (TTNN) problem by M-ADMM
%
% min_X ||X||_*, s.t. P_Omega(X) = P_Omega(M)
%
% ---------------------------------------------
% Input:
%       M       -    d1*d2*d3 tensor
%       omega   -    index of the observed entries
%       opts    -    Structure value in Matlab. The fields are
%           opts.tol        -   termination tolerance
%           opts.max_iter   -   maximum number of iterations
%           opts.mu         -   stepsize for dual variable updating in ADMM
%           opts.max_mu     -   maximum stepsize
%           opts.rho        -   rho>=1, ratio used to increase mu
%           opts.DEBUG      -   0 or 1
%
% Output:
%       X       -    d1*d2*d3 tensor
%       err     -    residual
%       obj     -    objective function value
%       iter    -    number of iterations

out_tol = 1e-3; 
out_iter = 50;
% rho = 1.1;
% mu = 1e-1;
% max_mu = 1e10;
min_R = 1;
max_R = 20;
max_P = 255;
DEBUG = 0;

if ~exist('opts', 'var')
    opts = [];
end
if isfield(opts, 'tol');         out_tol = opts.out_tol;      end
if isfield(opts, 'out_iter');    out_iter = opts.out_iter;    end
% if isfield(opts, 'rho');         rho = opts.rho;              end
% if isfield(opts, 'mu');          mu = opts.mu;                end
% if isfield(opts, 'max_mu');      max_mu = opts.max_mu;        end
if isfield(opts, 'min_R');       min_R = opts.min_R;          end
if isfield(opts, 'max_R');       max_R = opts.max_R;          end
if isfield(opts, 'max_P');       max_P = opts.max_P;          end
if isfield(opts, 'DEBUG');       DEBUG = opts.DEBUG;          end

[n1, n2, n3] = size(M);
Erec = zeros(max_R, 1);  % reconstruction error, best value in each rank
Psnr = zeros(max_R, 1);  % PSNR, best value in each rank
time_cost = zeros(max_R, 1);      % consuming time, each rank
iter_outer = zeros(max_R, 1);     % number of outer iterations
iter_total = zeros(max_R, 1);     % number of total iterations
X_rec = zeros(n1, n2, n3, out_iter);  % recovered image under the best rank

best_rank = 0;  % record the best value
best_psnr = 0;
best_erec = 0;

dim = size(M);
% k = length(dim);
% omegac = setdiff(1:prod(dim), omega);
% X = zeros(dim);
% X(omega) = M(omega);
% W = X;
% Y = zeros(dim);
norm_M = norm(M(:));
        
for R = min_R : max_R    
    X_iter = zeros(n1, n2, n3, out_iter);
    X = zeros(dim);
    X(omega) = M(omega);
    t_rank = tic;
    for i = 1 : out_iter
        fprintf('rank=%d, outer_iter=%d\n', R, i);
        last_X = X;  
        [U, S, V] = t_SVD(X);
        A = tran(U(:, 1:R, :)); B = tran(V(:, 1:R, :)); 
        [X, iter_inner] = t_admmAXB(A, B, X, M, omega, opts);
        
        X_iter(:, :, :, i) = X;
        iter_outer(R) = iter_outer(R) + 1;
        iter_total(R) = iter_total(R) + iter_inner;
        
        delta = norm(vec(X - last_X)) / norm_M;
        fprintf('||X_k+1-X_k||_F/||M||_F = %.4f\n', delta);
        if delta < out_tol
            fprintf('converged at iter=%d(%d)\n', i, iter_total(R));
            break ;
        end                   
    end
    time_cost(R) = toc(t_rank);
    X = max(X, 0);
    X = min(X, max_P);
    [Erec(R), Psnr(R)] = PSNR(M, X, omega);
    if best_psnr < Psnr(R)
        best_rank = R;
        best_psnr = Psnr(R);
        best_erec = Erec(R);
        X_rec = X_iter;
    end
end

%% compute the reconstruction error and PSNR in each iteration 
%  for the best rank
num_iter = iter_outer(best_rank);
psnr_iter = zeros(num_iter, 1);
erec_iter = zeros(num_iter, 1);
for t = 1 : num_iter
    X_temp = X_rec(:, :, :, t);
    [erec_iter(t), psnr_iter(t)] = PSNR(M, X_temp, omega);
end
X_best_rec = X_rec(:, :, :, num_iter);
X_best_rec = max(X_best_rec, 0);
X_best_rec = min(X_best_rec, max_P);

%% record performances for output
result.time = time_cost;
result.iterations = iter_outer;
result.total_iter = iter_total;
result.best_rank = best_rank;
result.best_psnr = best_psnr;
result.best_erec = best_erec;
result.Rank = (min_R : max_R)';
result.Psnr = Psnr(min_R:max_R);
result.Erec = Erec(min_R:max_R);
result.Psnr_iter = psnr_iter;
result.Erec_iter = erec_iter;

end