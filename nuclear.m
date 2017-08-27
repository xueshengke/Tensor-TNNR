function [ Y ] = nuclear( X )

Y = 0;
dim = ndims(X);

if dim == 2
    Y = sum( svd(X, 'econ') );
elseif dim == 3
    [n1, n2, n3] = size(X);
%     n12 = min(n1, n2);
%     Sf = zeros(n12, n12, n3);
    Xf = fft(X, [], dim);
    for i = 1 : n3
        Sf = svd(Xf(:, :, i), 'econ');
        Y = Y + sum(Sf);
    end
    Y = Y / n3;
end

end