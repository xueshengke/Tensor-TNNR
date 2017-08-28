%% add path
addpath(genpath(cd))
close all
clear
clc

%% read image files directory information
admm_result = './TNNR-admm/result/image';
apgl_result = './TNNR-apgl/result/image';
if ~exist(admm_result, 'dir'),   mkdir(admm_result); end
if ~exist(apgl_result, 'dir'),   mkdir(apgl_result); end
image_list = {'re1.jpg', 're2.jpg', 're3.jpg', 're4.jpg', 're5.jpg', ...
              're6.jpg', 're7.jpg', 're8.jpg', 're9.jpg', 're10.jpg', ...
              're11.jpg' };

file_list = dir('mask');
num_mask = length(file_list) - 2;
mask_list = cell(num_mask, 1);
for i = 1 : num_mask
    mask_list{i} = file_list(i+2).name;
end

%% parameter configuration
image_id = 9;            % select an image for experiment
mask_id  = 4;            % select a mask for experiment

para.block = 0;          % 1 for block occlusion, 0 for random noise
para.lost = 0.40;        % percentage of lost elements in matrix
para.save_eps = 1;       % save eps figure in result directory
% it requires to test all ranks from min_R to max_R, note that different
% images have different ranks, and various masks affect the ranks, too.

para.outer_iter = 100;     % maximum number of iteration
para.outer_tol = 3e-4;     % tolerance of iteration

para.admm_iter = 200;    % iteration of the ADMM optimization
para.admm_tol = 1e-4;    % epsilon of the ADMM optimization
para.admm_rho = 5e-2;    % rho of the the ADMM optimization

para.apgl_iter = 200;    % iteration of the APGL optimization
para.apgl_tol = 1e-4;    % epsilon of the APGL optimization
para.apgl_lambda = 1e-2; % lambda of the the APGL optimization
para.progress = 0;

%%-------------------------------------------------------------------------
opts.min_R =  1;         % minimum rank of chosen image
opts.max_R = 15;         % maximum rank of chosen image

opts.mu = 1e-3;
opts.rho = 1.05;
opts.max_mu = 1e10;
opts.out_tol = 1e-3;
opts.tol = 1e-4;
opts.out_iter = 50;
opts.max_iter = 200;
opts.maxP = 255;
opts.DEBUG = 1;

%% select an image and a mask for experiment
image_name = image_list{image_id};
X_full = double(imread(image_name));
[n1, n2, n3] = size(X_full);
fprintf('choose image: %s, ', image_name);

if para.block  
    % block occlusion
    mask = double(imread(mask_list{mask_id}));
    mask = mask ./ max(mask(:));       % index matrix of the known elements
    fprintf('mask: %s.\n', mask_list{mask_id});
    omega = find(mask);
else
    lost = para.lost;
    fprintf('loss: %d%% elements are randomly missing\n', lost*100);
%     % random loss 1, same along all channels
%     rnd_idx = double(rand(n1,n2) < (1-para.lost));
%     mask = repmat(rnd_idx, [1 1 n3]); % index matrix of the known elements
    % random loss 2, different along all channels
    mask = double(rand(n1,n2,n3) < (1-lost));
    omega = find(mask);
end

M = zeros(n1, n2, n3);
M(omega) = X_full(omega);
max_P = opts.maxP;
%% test
%% low-rank tensor completion based on truncated tensor nuclear norm
fprintf('ADMM optimization method to recovery an image with missing pixels\n');

t1 = tic;
[Xhat, admm_res] = lrtc_ttnn(X_full, omega, opts);
toc(t1)

% Xhat = max(Xhat, 0);
% Xhat = min(Xhat, max_P);
% RSE = norm(X_full(:)-Xhat(:))/norm(X_full(:))
% [erec, psnr] = PSNR(X_full, Xhat, omega, max_P)

figure(1)
subplot(1,3,1)
imshow(X_full/max_P)
title('original image')
subplot(1,3,2)
imshow(M/max_P)
title('incompelte image')
subplot(1,3,3)
imshow(Xhat/max_P)
title('recovered image')

admm_rank = admm_res.best_rank;
admm_psnr = admm_res.best_psnr;
admm_erec = admm_res.best_erec;
admm_time_cost = admm_res.time(admm_rank);
admm_iteration = admm_res.iterations(admm_rank);
admm_total_iter = admm_res.total_iter(admm_rank);

fprintf('\nTNNR-ADMM: ');
fprintf('rank=%d, psnr=%.4f, erec=%.4f, time=%f s, iteration=%d(%d)\n', ...
    admm_rank, admm_psnr, admm_erec, admm_time_cost, admm_iteration, ...
    admm_total_iter);
disp(' ');

figure('NumberTitle', 'off', 'Name', 'TNNR-ADMM result');
subplot(2, 2, 1);
plot(admm_res.Rank, admm_res.Psnr, 'o-');
xlabel('Rank');
ylabel('PSNR');

subplot(2, 2, 2);
plot(admm_res.Rank, admm_res.Erec, 'diamond-');
xlabel('Rank');
ylabel('Recovery error');

subplot(2, 2, 3);
plot(admm_res.Psnr_iter, 'square-');
xlabel('Iteration');
ylabel('PSNR');

subplot(2, 2, 4);
plot(admm_res.Erec_iter, '^-');
xlabel('Iteration');
ylabel('Recovery error');
