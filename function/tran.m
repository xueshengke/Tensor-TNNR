function Xt = tran(X)

% conjugate transpose of a 3-order tensor 
% X  - n1 x n2 x n3 tensor
% Xt - n2 x n1 x n3 tensor

[n1, n2, n3] = size(X);
Xt = zeros(n2, n1, n3);
Xt(:, :, 1) = X(:, :, 1)';
if n3 > 1
    for i = 2 : n3
        Xt(:, :, i) = X(:, :, n3-i+2)';
    end
end
