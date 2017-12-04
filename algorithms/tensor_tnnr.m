function [X_best_rec, result] = tensor_tnnr(M, omega, opts)
%--------------------------------------------------------------------------
% Shengke Xue, Zhejiang University, September 2017. 
% Contact information: see readme.txt.
%
% Xue et al. (2018) Tensor-TNN paper, ICPR.
%--------------------------------------------------------------------------
%    main loop of tensor completion based on truncated nuclear norm 
%
%    min_X ||X||_* - max tr(A * X * B')  s.t.  (X)_Omega = (M)_Omega
%
% Input:
%       M       -    n1 x n2 x n3 tensor
%       omega   -    index of the known elements
%       opts    -    struct contains parameters
%
% Output:
%       X_best_rec -  recovered tensor at the best rank
%       result     -  result of algorithm
%--------------------------------------------------------------------------

min_R = 1;
max_R = 20;
out_tol = 1e-3; 
out_iter = 50;
max_P = 255;
method = 'ADMM';    % ADMM or APGL

if ~exist('opts', 'var'),   opts = [];  end

if isfield(opts, 'min_R');      min_R = opts.min_R;         end
if isfield(opts, 'max_R');      max_R = opts.max_R;         end
if isfield(opts, 'out_tol');    out_tol = opts.out_tol;     end
if isfield(opts, 'out_iter');   out_iter = opts.out_iter;	end
if isfield(opts, 'max_P');      max_P = opts.max_P;         end
if isfield(opts, 'method');     method = opts.method;       end

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
norm_M = norm(M(:));

for R = min_R : max_R    % search from [min_R, max_R] one by one
    X_iter = zeros(n1, n2, n3, out_iter);
    X = zeros(dim);
    X(omega) = M(omega);
    t_rank = tic;
    for i = 1 : out_iter
        fprintf('rank=%d, outer_iter=%d\n', R, i);
        last_X = X;  
        [U, S, V] = t_SVD(X);           % Step 1: tensor SVD
        A = tran(U(:, 1:R, :)); B = tran(V(:, 1:R, :));
        if strcmp(method, 'ADMM')       % Step 2: use ADMM
            [X, iter_in] = tnn_admm(A, B, X, M, omega, opts);
        elseif strcmp(method, 'APGL')   % Step 2: use APGL
            [X, iter_in] = tnn_apgl(A, B, X, M, omega, opts);
        end  
        X_iter(:, :, :, i) = X;
        iter_outer(R) = iter_outer(R) + 1;
        iter_total(R) = iter_total(R) + iter_in;
        
        delta = norm(vec(X - last_X)) / norm_M;
        fprintf('||X_k+1-X_k||_F/||M||_F = %.4f\n', delta);
        if delta < out_tol
            fprintf('converged at iter=%d(%d)\n\n', i, iter_total(R));
            break ;
        end                   
    end
    time_cost(R) = toc(t_rank);
    X = max(X, 0);
    X = min(X, max_P);
    [Erec(R), Psnr(R)] = psnr(M, X, omega);
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
    [erec_iter(t), psnr_iter(t)] = psnr(M, X_temp, omega);
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