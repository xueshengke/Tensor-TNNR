function A = itbdiag(Abdiag, n1, n2, n3)
%
% reformulate a block diagonal matrix as a 3-order tensor 
% Abdiag - n1n3 x n2n3 matrix
% A      - n1 x n2 x n3 tensor

A = zeros(n1, n2, n3);
for i = 1 : n3
    A(:, :, i) = Abdiag((i-1)*n1+1 : i*n1, (i-1)*n2+1 : i*n2);
end

end