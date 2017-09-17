% Shengke Xue, Zhejiang University, September 2017. 
% Contact information: see readme.txt.
%
% Reference: 
% S. Xue, et~al., Low-rank Tensor Completion by Truncated Nuclear Norm 
% Regularization, International Conference on ***, submitted, 2018.

%% add path
addpath(genpath(cd))
close all
clear
clc

%% read image files directory information
admm_result = './result/admm/synthetic';
apgl_result = './result/apgl/synthetic';
if ~exist(admm_result, 'dir'),	mkdir(admm_result);	end
if ~exist(apgl_result, 'dir'),  mkdir(apgl_result);	end

%% parameter configuration
opts.lost = 0.50;       % percentage of lost elements in matrix

opts.min_R = 1;         % minimum rank of chosen image
opts.max_R = 20;        % maximum rank of chosen image

opts.out_iter = 50;     % maximum number of outer iteration
opts.out_tol = 1e-3;    % tolerance of outer iteration

opts.mu = 5e-4;         % mu of ADMM optimization
opts.rho = 1.15;        % rho of ADMM optimization
opts.max_mu = 1e10;     % max value of mu
opts.admm_iter = 200;   % maximum number of ADMM iteration
opts.admm_tol = 1e-4;   % tolerance of ADMM iteration

opts.lambda = 1e-1;     % lambda of APGL optimization
opts.apgl_iter = 200;   % maximum number of APGL iteration
opts.apgl_tol = 1e-4;   % tolerance of APGL iteration

opts.maxP = 1;          % max pixel value

%% generate synthetic data for experiment
image_name = 'synthetic_data';
n1 = 100;
n2 = 100;
n3 = 50;
r0 = 10;

% random loss different positions along all channels
lost = opts.lost;
fprintf('loss: %d%% elements are randomly missing\n', lost*100);
mask = double(rand(n1,n2,n3) < (1-lost));
omega = find(mask);

max_P = opts.maxP;
L = randn(n1, r0, n3);
R = randn(n2, r0, n3);
M = tprod(L, tran(R));
X_full = max_P * ( M - min(M(:)) ) / ( max(M(:)) - min(M(:)) );
M = zeros(n1, n2, n3);
M(omega) = X_full(omega);

%% tensor truncated tensor nuclear norm, using ADMM
fprintf('ADMM method to recover an image with missing pixels\n');
opts.method = 'ADMM';

t1 = tic;
[X_hat, admm_res] = tensor_tnnr(X_full, omega, opts);
toc(t1)

admm_rank = admm_res.best_rank;
admm_psnr = admm_res.best_psnr;
admm_erec = admm_res.best_erec / max_P;
admm_time_cost = admm_res.time(admm_rank);
admm_iteration = admm_res.iterations(admm_rank);
admm_total_iter = admm_res.total_iter(admm_rank);

fprintf('\nTensor TNNR (ADMM):\n');
fprintf('rank=%d, psnr=%.4f, erec=%.4f, time=%.3f s, iteration=%d(%d)\n', ...
    admm_rank, admm_psnr, admm_erec, admm_time_cost, admm_iteration, ...
    admm_total_iter);
disp(' ');

figure('NumberTitle', 'off', 'Name', 'Tensor TNNR (ADMM) result')
subplot(2, 2, 1)
plot(admm_res.Rank, admm_res.Psnr, 'o-')
xlabel('Rank')
ylabel('PSNR')

subplot(2, 2, 2)
plot(admm_res.Rank, admm_res.Erec / max_P, 'diamond-')
xlabel('Rank')
ylabel('Recovery error')

subplot(2, 2, 3)
plot(admm_res.Psnr_iter, 'square-')
xlabel('Iteration')
ylabel('PSNR')

subplot(2, 2, 4)
plot(admm_res.Erec_iter / max_P, '^-')
xlabel('Iteration')
ylabel('Recovery error')

%% record test results
outputFileName = fullfile(admm_result, 'parameters.txt'); 
fid = fopen(outputFileName, 'a') ;
fprintf(fid, '****** %s ******\n', datestr(now,0));
fprintf(fid, '%s\n', ['image: '           image_name               ]);
fprintf(fid, '%s\n', ['loss ratio: '      num2str(opts.lost)       ]);
fprintf(fid, '%s\n', ['min rank: '        num2str(opts.min_R)      ]);
fprintf(fid, '%s\n', ['max rank: '        num2str(opts.max_R)      ]);
fprintf(fid, '%s\n', ['max iteration: '   num2str(opts.out_iter)   ]);
fprintf(fid, '%s\n', ['tolerance: '       num2str(opts.out_tol)    ]);
fprintf(fid, '%s\n', ['ADMM mu: '         num2str(opts.mu)         ]);
fprintf(fid, '%s\n', ['ADMM rho: '        num2str(opts.rho)        ]);
fprintf(fid, '%s\n', ['ADMM max_mu: '     num2str(opts.max_mu)     ]);
fprintf(fid, '%s\n', ['ADMM iteration: '  num2str(opts.admm_iter)  ]);
fprintf(fid, '%s\n', ['ADMM tolerance: '  num2str(opts.admm_tol)   ]);
fprintf(fid, '%s\n', ['max pixel value: ' num2str(opts.maxP)       ]);

fprintf(fid, '%s\n', ['rank: '            num2str(admm_rank)       ]);
fprintf(fid, '%s\n', ['psnr: '            num2str(admm_psnr)       ]);
fprintf(fid, '%s\n', ['recovery error: '  num2str(admm_erec)       ]);
fprintf(fid, '%s\n', ['time cost: '       num2str(admm_time_cost)  ]);
fprintf(fid, 'iteration: %d(%d)\n',       admm_iteration, admm_total_iter);
fprintf(fid, '--------------------\n');
fclose(fid);

%% tensor truncated tensor nuclear norm, using APGL
fprintf('APGL method to recover an image with missing pixels\n');
opts.method = 'APGL';

t2 = tic;
[X_hat, apgl_res] = tensor_tnnr(X_full, omega, opts);
toc(t2)

apgl_rank = apgl_res.best_rank;
apgl_psnr = apgl_res.best_psnr;
apgl_erec = apgl_res.best_erec / max_P;
apgl_time_cost = apgl_res.time(apgl_rank);
apgl_iteration = apgl_res.iterations(apgl_rank);
apgl_total_iter = apgl_res.total_iter(apgl_rank);

fprintf('\nTensor TNNR (APGL):\n');
fprintf('rank=%d, psnr=%.4f, erec=%.4f, time=%.3f s, iteration=%d(%d)\n', ...
    apgl_rank, apgl_psnr, apgl_erec, apgl_time_cost, apgl_iteration, ...
    apgl_total_iter);
disp(' ');

figure('NumberTitle', 'off', 'Name', 'Tensor TNNR (APGL) result')
subplot(2, 2, 1)
plot(apgl_res.Rank, apgl_res.Psnr, 'o-')
xlabel('Rank')
ylabel('PSNR')

subplot(2, 2, 2)
plot(apgl_res.Rank, apgl_res.Erec / max_P, 'diamond-')
xlabel('Rank')
ylabel('Recovery error')

subplot(2, 2, 3)
plot(apgl_res.Psnr_iter, 'square-')
xlabel('Iteration')
ylabel('PSNR')

subplot(2, 2, 4)
plot(apgl_res.Erec_iter / max_P, '^-')
xlabel('Iteration')
ylabel('Recovery error')

%% record test results
outputFileName = fullfile(apgl_result, 'parameters.txt'); 
fid = fopen(outputFileName, 'a') ;
fprintf(fid, '****** %s ******\n', datestr(now,0));
fprintf(fid, '%s\n', ['image: '           image_name               ]);
fprintf(fid, '%s\n', ['loss ratio: '      num2str(opts.lost)       ]);
fprintf(fid, '%s\n', ['min rank: '        num2str(opts.min_R)      ]);
fprintf(fid, '%s\n', ['max rank: '        num2str(opts.max_R)      ]);
fprintf(fid, '%s\n', ['max iteration: '   num2str(opts.out_iter)   ]);
fprintf(fid, '%s\n', ['tolerance: '       num2str(opts.out_tol)    ]);
fprintf(fid, '%s\n', ['APGL lambda: '     num2str(opts.lambda)     ]);
fprintf(fid, '%s\n', ['APGL iteration: '  num2str(opts.apgl_iter)  ]);
fprintf(fid, '%s\n', ['APGL tolerance: '  num2str(opts.apgl_tol)   ]);
fprintf(fid, '%s\n', ['max pixel value: ' num2str(opts.maxP)       ]);

fprintf(fid, '%s\n', ['rank: '            num2str(apgl_rank)       ]);
fprintf(fid, '%s\n', ['psnr: '            num2str(apgl_psnr)       ]);
fprintf(fid, '%s\n', ['recovery error: '  num2str(apgl_erec)       ]);
fprintf(fid, '%s\n', ['time cost: '       num2str(apgl_time_cost)  ]);
fprintf(fid, 'iteration: %d(%d)\n',       apgl_iteration, apgl_total_iter);
fprintf(fid, '--------------------\n');
fclose(fid);
