% proof of the property of the fft function in the t-SVD

clear; clc;
n1 = 50; n2 = 40; n3 = 20; r = 10;
X = rand(n1,n2,n3);

[U,S,V] = t_SVD(X);
Xf = fft(X,[],3);
Uf = fft(U,[],3);
Sf = fft(S,[],3);
Vf = fft(V,[],3);

Xsqz = sum(X, 3);
Xfsqz= Xf(:,:,1);
sum(vec(Xsqz-Xfsqz))

sumX = sum(X(:));
sumXf = sum(vec(Xf(:,:,1)));
sumUSVf = sum(vec( Uf(:,:,1) * Sf(:,:,1) * Vf(:,:,1)' ));

nuclearS = 0;
for i = 1 : n3
    nuclearS = nuclearS + sum(diag(S(:,:,i)));
end

nuclearSf = sum(diag(Sf(:,:,1)));

Ssqz = sum(S,3);
Sfsqz= Sf(:,:,1);
sum(vec(Ssqz-Sfsqz))

nuclearX1 = nuclear(Xf(:,:,1));
nuclearX2 = nuclear(X);

A = tran(U(:,1:r,:));
B = tran(V(:,1:r,:));
Af = tran(Uf(:,1:r,:));
Bf = tran(Vf(:,1:r,:));

t1 = trace(Af(:,:,1) * Xf(:,:,1) * Bf(:,:,1)');
t2 = trace(tprod(tprod(A,X),tran(B)));

% proof succeeded!