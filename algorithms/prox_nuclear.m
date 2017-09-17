function [Y, nuclear] = prox_nuclear(X, lambda)

% The proximal operator of the nuclear norm of a matrix
% 
% min_X lambda*||X||_*+0.5*||X-B||_F^2

[U, S, V] = svd(X);
S = diag(S);
svp = length(find(S > lambda));
if svp >= 1
    S = S(1 : svp) - lambda;
    Y = U(:, 1 : svp) * diag(S) * V(:, 1 : svp)';
    nuclear = sum(S);
else
    Y = zeros(size(X));
    nuclear = 0;
end