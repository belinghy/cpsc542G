n = 127;
h = 1/(n+1);
funA = @(n,w) (n+1)^2 * gallery('poisson', n) - w^2 * eye(n*n);
g = ones(n*n,1); % Random RHS

% w = 2
A = funA(n, 2);
[L,~] = ilu(sparse(A));
[du1,FLAG1,RELRES1,ITER1,RESVEC1] = gmres(A, g, 20, 1e-6, 300, L);

% w = 100
A2 = funA(n, 10);
[L2,~] = ilu(sparse(A));
[du2,FLAG2,RELRES2,ITER2,RESVEC2] = gmres(A2, g, 20, 1e-6, 300, L2);

% Plotting only
u12d = reshape(du1, [n,n]);
pu1 = padarray(u12d, [1,1], 0);
u22d = reshape(du2, [n,n]);
pu2 = padarray(u22d, [1,1], 0);
[X, Y] = meshgrid(0:1/(n+1):1);

figure;
ax1 = subplot(1,2,1);
ax2 = subplot(1,2,2);
surf(ax1, X, Y, pu1);
surf(ax2, X, Y, pu2);