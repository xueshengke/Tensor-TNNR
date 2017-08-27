function [erec, psnr] = PSNR(X_full, X_rec, omega, maxP)

if ~exist('maxP', 'var')
    maxP = 255;
end

X_rec = max(0, X_rec);
X_rec = min(maxP, X_rec);

dim = size(X_rec);
omegac = setdiff(1:prod(dim), omega);
num_missing = length(omegac);

Xtemp = X_full - X_rec;
erec = norm(Xtemp(:))^2;
MSE = norm(Xtemp(omegac))^2 / num_missing;
psnr = 10 * log10(maxP^2 / MSE);

% function psnr = PSNR(Xfull,Xrecover,maxP)
% 
% Xrecover = max(0,Xrecover);
% Xrecover = min(maxP,Xrecover);
% [m,n,dim] = size(Xrecover);
% MSE = norm(Xfull(:)-Xrecover(:))^2/(3*m*n);
% psnr = 10*log10(maxP^2/MSE);