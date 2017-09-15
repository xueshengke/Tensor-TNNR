%% add path
addpath(genpath(cd))
close all
clear
clc

%% read image files directory information
admm_result = './result/admm/image';
apgl_result = './result/apgl/image';
if ~exist(admm_result, 'dir'),	mkdir(admm_result);	end
if ~exist(apgl_result, 'dir'),  mkdir(apgl_result);	end
% image_list = {'re1.jpg', 're2.jpg', 're3.jpg', 're4.jpg', 're5.jpg', ...
%               're6.jpg', 're7.jpg', 're8.jpg', 're9.jpg', 're10.jpg', ...
%              };
image_list = {'new1.jpg', 'new2.jpg', 'new3.jpg', 'new4.jpg', 'new5.jpg', ...
              'new6.jpg', 'new7.jpg', 'new8.jpg', 'new9.jpg', 'new10.jpg', ...
             };
         
file_list = dir('mask');
num_mask = length(file_list) - 2;
mask_list = cell(num_mask, 1);
for i = 1 : num_mask
    mask_list{i} = file_list(i+2).name; 
end

%% parameter configuration
image_id = 1;           % select an image for experiment
mask_id  = 4;           % select a mask for experiment

opts.block = 0;         % 1 for block occlusion, 0 for random noise
opts.lost = 0.50;       % percentage of lost elements in matrix
opts.save_eps = 1;      % save eps figure in result directory
% it requires to test all ranks from min_R to max_R, note that different
% images have different ranks, and various masks affect the ranks, too.

opts.min_R = 1;         % minimum rank of chosen image
opts.max_R = 20;        % maximum rank of chosen image

opts.out_iter = 50;     % maximum number of outer iteration
opts.out_tol = 1e-3;    % tolerance of outer iteration

opts.mu = 5e-4;         % mu of ADMM optimization
opts.rho = 1.15;        % rho of ADMM optimization 1.05
opts.max_mu = 1e10;     % max value of mu
opts.admm_iter = 200;   % maximum number of ADMM iteration
opts.admm_tol = 1e-4;   % tolerance of ADMM iteration

opts.lambda = 1e-2;     % lambda of APGL optimization
opts.apgl_iter = 200;   % maximum number of APGL iteration
opts.apgl_tol = 1e-4;   % tolerance of APGL iteration

opts.maxP = 255;

%% select an image and a mask for experiment
image_name = image_list{image_id};
X_full = double(imread(image_name));
[n1, n2, n3] = size(X_full);
fprintf('choose image: %s, ', image_name);

if opts.block  
    % block occlusion
    mask = double(imread(mask_list{mask_id}));
    mask = mask ./ max(mask(:));       % index matrix of the known elements
    fprintf('mask: %s.\n', mask_list{mask_id});
    omega = find(mask);
else
    lost = opts.lost;
    fprintf('loss: %d%% elements are randomly missing\n', lost*100);
%     % random loss 1, same along all channels
%     rnd_idx = double(rand(n1,n2) < (1-para.lost));
%     mask = repmat(rnd_idx, [1 1 n3]); % index matrix of the known elements
    % random loss 2, different positions along all channels
    mask = double(rand(n1,n2,n3) < (1-lost));
    omega = find(mask);
end

M = zeros(n1, n2, n3);
M(omega) = X_full(omega);
max_P = opts.maxP;

%% tensor truncated tensor nuclear norm, using ADMM
fprintf('ADMM method to recover an image with missing pixels\n');
opts.method = 'ADMM';

t1 = tic;
[X_hat, admm_res] = tensor_tnnr(X_full, omega, opts);
toc(t1)

admm_rank = admm_res.best_rank;
admm_psnr = admm_res.best_psnr;
admm_erec = admm_res.best_erec;
admm_time_cost = admm_res.time(admm_rank);
admm_iteration = admm_res.iterations(admm_rank);
admm_total_iter = admm_res.total_iter(admm_rank);

figure
subplot(1,3,1)
imshow(X_full/max_P)
title('original image')
subplot(1,3,2)
imshow(M/max_P)
title('incompelte image')
subplot(1,3,3)
imshow(X_hat/max_P)
title('recovered image')

%% save eps figure in result directory
if opts.save_eps
    fig_eps = figure;
    imshow(X_hat ./ 255, 'border', 'tight');
    split_name = regexp(image_name, '[.]', 'split');
    fig_name = sprintf('%s/%s_rank_%d_PSNR_%.2f', ...
        admm_result, split_name{1}, admm_rank, admm_psnr);
    saveas(gcf, [fig_name '.eps'], 'psc2');
    fprintf('eps figure saved in %s.eps\n', fig_name);
    close(fig_eps);
end

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
plot(admm_res.Rank, admm_res.Erec, 'diamond-')
xlabel('Rank')
ylabel('Recovery error')

subplot(2, 2, 3)
plot(admm_res.Psnr_iter, 'square-')
xlabel('Iteration')
ylabel('PSNR')

subplot(2, 2, 4)
plot(admm_res.Erec_iter, '^-')
xlabel('Iteration')
ylabel('Recovery error')

%% record test results
outputFileName = fullfile(admm_result, 'parameters.txt'); 
fid = fopen(outputFileName, 'a') ;
fprintf(fid, '****** %s ******\n', datestr(now,0));
fprintf(fid, '%s\n', ['image: '           image_name               ]);
fprintf(fid, '%s\n', ['mask: '            mask_list{mask_id}       ]);
fprintf(fid, '%s\n', ['block or noise: '  num2str(opts.block)      ]);
fprintf(fid, '%s\n', ['loss ratio: '      num2str(opts.lost)       ]);
fprintf(fid, '%s\n', ['save eps figure: ' num2str(opts.save_eps)   ]);
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
apgl_erec = apgl_res.best_erec;
apgl_time_cost = apgl_res.time(apgl_rank);
apgl_iteration = apgl_res.iterations(apgl_rank);
apgl_total_iter = apgl_res.total_iter(apgl_rank);

figure
subplot(1,3,1)
imshow(X_full/max_P)
title('original image')
subplot(1,3,2)
imshow(M/max_P)
title('incompelte image')
subplot(1,3,3)
imshow(X_hat/max_P)
title('recovered image')

%% save eps figure in result directory
if opts.save_eps
    fig_eps = figure;
    imshow(X_hat ./ 255, 'border', 'tight');
    split_name = regexp(image_name, '[.]', 'split');
    fig_name = sprintf('%s/%s_rank_%d_PSNR_%.2f', ...
        apgl_result, split_name{1}, apgl_rank, apgl_psnr);
    saveas(gcf, [fig_name '.eps'], 'psc2');
    fprintf('eps figure saved in %s.eps\n', fig_name);
    close(fig_eps);
end

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
plot(apgl_res.Rank, apgl_res.Erec, 'diamond-')
xlabel('Rank')
ylabel('Recovery error')

subplot(2, 2, 3)
plot(apgl_res.Psnr_iter, 'square-')
xlabel('Iteration')
ylabel('PSNR')

subplot(2, 2, 4)
plot(apgl_res.Erec_iter, '^-')
xlabel('Iteration')
ylabel('Recovery error')

%% record test results
outputFileName = fullfile(apgl_result, 'parameters.txt'); 
fid = fopen(outputFileName, 'a') ;
fprintf(fid, '****** %s ******\n', datestr(now,0));
fprintf(fid, '%s\n', ['image: '           image_name               ]);
fprintf(fid, '%s\n', ['mask: '            mask_list{mask_id}       ]);
fprintf(fid, '%s\n', ['block or noise: '  num2str(opts.block)      ]);
fprintf(fid, '%s\n', ['loss ratio: '      num2str(opts.lost)       ]);
fprintf(fid, '%s\n', ['save eps figure: ' num2str(opts.save_eps)   ]);
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
