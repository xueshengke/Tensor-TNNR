function [ Y ] = nuclear( X )

% compute the nuclear norm for matrices or tensors

Y = 0;
dim = ndims(X);

if dim == 2
    Y = sum( svd(X, 'econ') );
elseif dim == 3
    Xf = fft(X, [], dim);
    Sf = svd(Xf(:, :, 1), 'econ');
    Y = sum( Sf(:) );
end

end