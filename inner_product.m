function [ C ] = inner_product(A, B)

dimA = ndims(A);
dimB = ndims(B);

if dimA ~= dimB
    error('The dimensions of A and B do not match!');
end

C = 0;
if     dimA == 1 && dimB == 1
    C = A' * B;
elseif dimA == 2 && dimB == 2
    C = trace(A' * B);
elseif dimA == 3 && dimB == 3
    [n1, n2, n3] = size(A);
    for i = 1 : n3
        C = C + trace( A(:,:,i)' * B(:,:,i) );
    end
end

end

