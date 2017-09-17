function Y = prox_l1(X, lambda)

% The proximal operator of the l1 norm
% 
% min_x lambda*||x||_1+0.5*||x-b||_2^2

Y = max(0, X - lambda) + min(0, X + lambda);