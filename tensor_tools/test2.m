
clear
n1 = 20;
n2 = 30;
n3 = 50;
n12 = min(n1,n2);
l = 40;
A = rand(n1,n1,n3);
B = rand(n1,n2,n3);
X = rand(n1,n2,n3);
I = zeros(n12,n12,n3);
I(:,:,1) = diag([ones(1,5) zeros(1,n12-5)]);
t1 = trace( tprod( tprod(A,X), tran(B) ) );
t3 = trace( tprod( tprod(X,tran(B)), A ) );
t1 - t3
If = fft(I, [], 3);

Af = fft(A, [], 3);
Bf = fft(B, [], 3);
Xf = fft(X, [], 3);
for i = 1 : n3
    P(:,:,i) = Af(:,:,i) * Xf(:,:,i) * tran(Bf(:,:,i));
end
t2 = trace(ifft(P,[],3));
t1 - t2
svd(Xf(:,:,1))

[U,S,V] = t_SVD(X);

UtU = tprod(U, tran(U));
UtU(:,:,1) * UtU(:,:,1)'
Uf = fft(U, [], 3)
Uf(:,:,1) * Uf(:,:,1)'

VtV = tprod(tran(V), V);
VtV(:,:,1) * VtV(:,:,1)'
Vf = fft(V, [], 3)
Vf(:,:,1)' * Vf(:,:,1)

r = 5;
sigma = 0;
for i = 1 : n3
    sigma = sigma + sum(diag(S(1:r,1:r,i)));
end
A = tran(U(:,1:r,:));
B = tran(V(:,1:r,:));
tr = trace( tprod( tprod(A,X), tran(B) ) );


% sum(vec( A - fold(unfold(A), n1, n2, n3) ))
% sum(vec( A - itbdiag(tbdiag(A), n1, n2, n3) ))
% sum(vec( A - ibcirc(bcirc(A), n1, n2, n3) ))
% tic
% for i = 1 : 100
%     C = tprod(A, B);
% end
% toc
% tic
% for i = 1 : 100
%     Cp = fold(bcirc(A) * unfold(B), n1, l ,n3);
% end
% toc
% sum(C(:) - Cp(:))
% 
% [U,S,V] = t_SVD(A);
% At = tprod( tprod(U,S), tran(V));
% sum(A(:)-At(:))
tic
nuclear(A)
toc
tic
nuclear(bcirc(A)) / n3
toc

C = tprod(A,B);
C2 = tprod2(A,B);

norm(C(:)-C2(:))

At = tran(A)
At2 = tran2(A)
norm(At(:)-At2(:))

[U,S,V,trank] = tsvd(A);
trank
Ar = tprod(tprod(U,S),tran2(V));

norm(A(:)-Ar(:))

% U

UtU = tprod(tran(U),U)

dif = UtU - tI(size(UtU,1),size(UtU,3));
norm(dif(:))


C = tprod(A,tran(B));
Ab = fft(A,[],3);
Ab = tbdiag(Ab);
Bb = fft(tran(B),[],3);
Bb = tbdiag(Bb);
Cb = Ab*Bb;
C2 = itbdiag(Cb,n1,n2,n3);
C2 = ifft(C2,[],3);
norm(C(:)-C2(:))

Bt = rand(n1,n2,n3)-i*rand(n1,n2,n3);
Bt = ifft(Bt,[],3)



[U,S,V,trank,tnn] = tsvd(A);

s = S(:,:,1);
tnn - sum(s(:))



