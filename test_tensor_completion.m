addpath(genpath(cd))
close all
clear
clc

pic_name = [ './new_image/new1.jpg'];
% pic_name = [ './image/testimg.jpg'];
I = double(imread(pic_name));
X = I;
% X = I/255;
    
[n1,n2,n3] = size(X);

opts.mu = 1e-3;
opts.tol = 1e-6;
opts.rho = 1.05;    % 1.05
opts.max_iter = 500;
opts.DEBUG = 1;


p = 0.5;
% maxP = max(abs(X(:)));
maxP = 255;
omega = find(rand(n1*n2*n3,1)<p);
M = zeros(n1,n2,n3);
M(omega) = X(omega);


%% %% test lrtc_snn and lrtc_tnn
obj = 0;
alpha = [1, 1, 1e-3];
alpha = alpha / sum(alpha);

tic
[Xhat,err,iter] = lrtc_snn(M,omega,alpha,opts);
% [Xhat,obj,err,iter] = lrtc_tnn(M,omega,opts);
toc

obj
err
iter
 
Xhat = max(Xhat,0);
Xhat = min(Xhat,maxP);
RSE = norm(X(:)-Xhat(:))/norm(X(:))
[erec, psnr] = PSNR(X, Xhat, omega, maxP)
rankX = rank(Xhat(:,:,1))

figure(1)
subplot(1,3,1)
imshow(X/maxP)
title('original image')
subplot(1,3,2)
imshow(M/maxP)
title('incompelte image')
subplot(1,3,3)
imshow(Xhat/maxP)
title('recovered image')

%% test lrtcR_snn
% E = randn(n1,n2,n3)/100;
% M = M+E; 
% 
% alpha = [1, 1, 0.001]*10;
% % alpha = alpha / sum(alpha);
% 
% [Xhat,err,iter] = lrtcR_snn(M,omega,alpha,opts);
% err
% iter
%  
% 
% Xhat = max(Xhat,0);
% Xhat = min(Xhat,maxP);
% RSE = norm(X(:)-Xhat(:))/norm(X(:))
% psnr = PSNR(X,Xhat,maxP)
% 
% figure(2)
% subplot(1,3,1)
% imshow(X/maxP)
% subplot(1,3,2)
% imshow(M/maxP)
% subplot(1,3,3)
% imshow(Xhat/maxP)

% pause
%% test lrtcR_tnn

% E = randn(n1,n2,n3)/100;
% M = M+E; 
% 
% lambda = 0.1;
% [Xhat,E,obj,err,iter] = lrtcR_tnn(M,omega,lambda,opts);
% err
% iter
%  
% 
% Xhat = max(Xhat,0);
% Xhat = min(Xhat,maxP);
% RSE = norm(X(:)-Xhat(:))/norm(X(:))
% psnr = PSNR(X,Xhat,maxP)
% 
% figure(3)
% subplot(1,3,1)
% imshow(X/maxP)
% title('original image')
% subplot(1,3,2)
% imshow(M/maxP)
% title('incomplete image')
% subplot(1,3,3)
% imshow(Xhat/maxP)
% title('recovered image')
