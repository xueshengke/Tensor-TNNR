function C = tprod(A, B)

% tensor-tensor product of two 3-order tensors : C = A * B
% compute in the Fourier domain, efficiently
% A - n1 x n2 x n3 tensor
% B - n2 x l  x n3 tensor
% C - n1 x l  x n3 tensor

[n1, ~, n3] = size(A);
l = size(B, 2);
Af = fft(A, [], 3);
Bf = fft(B, [], 3);
Cf = zeros(n1, l, n3);
for i = 1 : n3
    Cf(:, :, i) = Af(:, :, i) * Bf(:, :, i);
end
C = ifft(Cf, [], 3);

end