function [U, S, V] = t_SVD(X)

% the singular value decomposition of a 3D tensor

dim = ndims(X);
[n1, n2, n3] = size(X);
n12 = min(n1, n2);
Xf = fft(X, [], dim);
Uf = zeros(n1, n12, n3);
Vf = zeros(n2, n12, n3);
Sf = zeros(n12, n12, n3);

Xf(isnan(Xf)) = 0;
Xf(isinf(Xf)) = 0;

t_rank = 0;
for i = 1 : n3
    [Uf(:,:,i), Sf(:,:,i), Vf(:,:,i)] = svd(Xf(:,:,i), 'econ');
    diagS = diag(Sf(:, :, i));
    temp = length(find(diagS > 0));
    t_rank = max(temp, t_rank);
end

Uf = Uf(:, 1:t_rank, :);
Vf = Vf(:, 1:t_rank, :);
Sf = Sf(1:t_rank, 1:t_rank, :);

U = ifft(Uf, [], dim);
S = ifft(Sf, [], dim);
V = ifft(Vf, [], dim);

end