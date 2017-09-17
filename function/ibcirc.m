function [ A ] = ibcirc(B, n1, n2, n3)

% transform the block circulant matrix back to tensor

C = B(:, 1:n2);
A = fold(C, n1, n2, n3);

end