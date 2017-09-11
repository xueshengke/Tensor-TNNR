addpath(genpath(cd))
close all
clear
clc

pic_name = [ './new_image/new9.jpg'];
% pic_name = [ './image/testimg.jpg'];
I = double(imread(pic_name));
X = I;
% X = I/255;
    
[n1,n2,n3] = size(X);

opts.mu = 1e-3;
opts.max_mu = 1e10;
opts.tol = 1e-6;
opts.rho = 1.05;
opts.max_iter = 500;
opts.DEBUG = 1;

p = 0.5;    % observed ratio
% maxP = max(abs(X(:)));
maxP = 255;

omega = find(rand(n1*n2*n3,1)<p);   % random loss over all elements

%% read mask directory information
% file_list = dir('mask');
% num_mask = length(file_list) - 2;
% mask_list = cell(num_mask, 1);
% for i = 1 : num_mask
%     mask_list{i} = file_list(i+2).name;
% end
% cover_id  = 12;           % select a mask for experiment
% cover = double(imread(mask_list{cover_id}));
% cover = cover ./ max(cover(:));       % index matrix of the known elements
% fprintf('mask: %s\n', mask_list{cover_id});
% omega = find(cover);

% observed = rand(n1,n2) < p;
% omega = find(repmat(observed, [1 1 n3]));

M = zeros(n1,n2,n3);
M(omega) = X(omega);


%% test lrtc_tnn
%% low-rank tensor completion based on tensor nuclear norm

% t1 = tic;
% [Xhat,obj,err,iter] = lrtc_tnn(M,omega,opts);
% toc(t1)
% 
% obj
% err
% iter
%  
% Xhat = max(Xhat,0);
% Xhat = min(Xhat,maxP);
% RSE = norm(X(:)-Xhat(:))/norm(X(:))
% psnr = PSNR(X,Xhat,omega,maxP)
% 
% figure(1)
% subplot(1,3,1)
% imshow(X/maxP)
% title('original image')
% subplot(1,3,2)
% imshow(M/maxP)
% title('incompelte image')
% subplot(1,3,3)
% imshow(Xhat/maxP)
% title('recovered image')

%% test LRMC
%% low-rank matrix completion

lambda = 1/sqrt(min(n1,n2));

Xrec = zeros(size(X));
mask = zeros(n1*n2*n3, 1);
mask(omega) = 1;
mask = reshape(mask, [n1 n2 n3]);
iteration = zeros(n3, 1);
t2 = tic;
for i = 1 : n3
    [Xhat,E,obj,err,iter] = lrmcR(M(:,:,i), find(mask(:,:,i)), lambda, opts);
%     [Xhat,obj,err,iter] = lrmc(M(:,:,i), find(mask(:,:,i)), opts);
%     [Xhat,obj,err,iter] = lrmc(M(:,:,i), find(observed), opts);
    obj;
    err;
    iteration(i) = iter;
    Xhat = max(Xhat,0);
    Xhat = min(Xhat,maxP);
    Xrec(:,:,i) = Xhat;
%     RSE = norm(X(:,:,i)-Xhat)/norm(X(:,:,i))
%     psnr = PSNR(X(:,:,i),Xhat,find(mask(:,:,i)),maxP)
%     psnr = PSNR(X(:,:,i),Xhat,find(observed),maxP)
end
toc(t2)

sum(iteration)
RSE = norm(X(:)-Xrec(:))/norm(X(:))
[erec, psnr] = PSNR(X, Xrec, omega, maxP)
rankX = rank(Xrec(:,:,1))

figure(2)
subplot(1,3,1)
imshow(X/maxP)
title('original image')
subplot(1,3,2)
imshow(M/maxP)
title('incompelte image')
subplot(1,3,3)
imshow(Xrec/maxP)
title('recovered image')
