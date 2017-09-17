function [ Y ] = unfold( X )

[n1, n2, n3] = size(X);
Y = zeros(n1*n3, n2);
for i = 1 : n3
    Y((i-1)*n1+1:i*n1, :) = X(:, :, i);
end

end