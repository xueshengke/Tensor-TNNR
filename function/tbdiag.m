function Abdiag = tbdiag(A)
%
% reformulate a 3-order tensor as a block diagonal matrix 
% A      - n1 x n2 x n3 tensor
% Abdiag - n1n3 x n2n3 matrix

[n1, n2, n3] = size(A);
Abdiag = zeros(n1*n3, n2*n3);

for i = 1 : n3
    Abdiag((i-1)*n1+1 : i*n1, (i-1)*n2+1 : i*n2) = A(:, :, i);
end

end