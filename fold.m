function [ Y ] = fold( X, n1, n2, n3)

Y = zeros(n1, n2, n3);
for i = 1 : n3
    Y(:, :, i) = X((i-1)*n1+1:i*n1, :);
end

end

