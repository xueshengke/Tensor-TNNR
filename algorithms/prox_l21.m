function Y = prox_l21(X, lambda)

% The proximal operator of the l21 norm of a matrix
% l21 norm is the sum of the l2 norm of all columns of a matrix 
%
% min_X lambda*||X||_{2,1}+0.5*||X-B||_2^2

Y = zeros(size(X));
for i = 1 : size(Y, 2)
    nxi = norm(X(:, i));
    if nxi > lambda  
        Y(:, i) = (1 - lambda / nxi) * X(:, i);
    end
end