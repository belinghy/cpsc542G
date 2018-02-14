h = 2^-7;
n = 127;
A = 1/h^2 * gallery('poisson', n);
b = zeros(n*n,1)+1e-8;

[u,FLAG,RELRES,ITER,RESVEC] = gmres(@(x)Afun(x,A),b,10,1e-6,2000);

u2d = reshape(u, [n,n]);
pu = padarray(u2d, [1,1], 0);
[X, Y] = meshgrid(0:1/(n+1):1);
figure;
ax1 = subplot(1,2,1);
ax2 = subplot(1,2,2);
surf(ax1, X, Y, pu)
semilogy(ax2, RESVEC)

function Ax=Afun(x,A)
  Ax = A*x - exp(x);
end
