function t = trace(A)
%TRACE  Sum of diagonal elements.
%   TRACE(A) is the sum of the diagonal elements of A, which is
%   also the sum of the eigenvalues of A.

if size(A,1) ~= size(A,2)
  error(message('MATLAB:trace:square'));
end

dim = ndims(A);
t = 0;
if dim == 2
    t = full(sum(diag(A)));
elseif dim == 3
    for i = 1 : size(A,3)
        t = t + full(sum(diag(A(:,:,i))));
    end
end
