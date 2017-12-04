function [X, tnn, trank] = prox_tnn(Y, rho)

% The proximal operator of the tensor nuclear norm of a 3-order tensor
%
% min_X rho*||X||_*+0.5*||X-Y||_F^2
%
% Y     -    n1*n2*n3 tensor
%
% X     -    n1*n2*n3 tensor
% tnn   -    tensor nuclear norm of X
% trank -    tensor tubal rank of X

dim = ndims(Y);
[n1, n2, n3] = size(Y);
n12 = min(n1, n2);
Yf = fft(Y, [], dim);
Uf = zeros(n1, n12, n3);
Vf = zeros(n2, n12, n3);
Sf = zeros(n12,n12, n3);

Yf(isnan(Yf)) = 0;
Yf(isinf(Yf)) = 0;

trank = 0;
for i = 1 : n3
    [Uf(:,:,i), Sf(:,:,i), Vf(:,:,i)] = svd(Yf(:,:,i), 'econ');
    s = diag(Sf(:, :, i));
    s = max(s - rho, 0);
    Sf(:, :, i) = diag(s);
    temp = length(find(s>0));
    trank = max(temp, trank);
end
Uf = Uf(:, 1:trank, :);
Vf = Vf(:, 1:trank, :);
Sf = Sf(1:trank, 1:trank, :);

U = ifft(Uf, [], dim);
S = ifft(Sf, [], dim);
V = ifft(Vf, [], dim);

X = tprod( tprod(U,S), tran(V) );

% tnn = 0;
% for i = 1 : n3
%     diagS = diag(Sf(:, :, i));
%     tnn = tnn + sum(diagS(:));
% end
% tnn = tnn / n3;   % average sum of frontal slices in Fourier domain
                    % definition in Lu, Canyi et al. (2016) CVPR paper

% our definition: tensor nuclear norm is the sum of singular values of the 
% first frontal slice $\bar{\bm{S}}^{(1)}$ in the Fourier domain
tnn = sum( diag( Sf(:,:,1) ) );

end