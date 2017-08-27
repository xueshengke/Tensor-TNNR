function [ B ] = bcirc(A)

[n1, n2, n3] = size(A);
B = zeros(n1*n3, n2*n3);
C = unfold(A);

for i = 1 : n3
    B(:, (i-1)*n2+1 : i*n2) = circshift(C, (i-1)*n1);
end

end