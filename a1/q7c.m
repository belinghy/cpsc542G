W = [2 10];
for w=W
  % 2D Helmholtz Problem using GMRES
  h = 2^-7;
  n = 127;
  iter_max = 8;
  tol = 1.e-6;
  % initial guess for u, (n*n, 1)
  % u = zeros(n*n, 1) + 0.1;
  [X, Y] = meshgrid(0:1/(n+1):1);
  X = X(2:n+1, 2:n+1);
  Y = Y(2:n+1, 2:n+1);
  u = reshape(1*X.*(1-X).*Y.*(1-Y), [n*n,1]);
  A = 1/h^2 * gallery('poisson', n) - w^2 * eye(n*n);
  [L,U] = ilu(sparse(A));
  
  for i=1:iter_max
    fu = A*u;
    Ju = A;
    [du,FLAG,RELRES,ITER,RESVEC] = gmres(-Ju,fu,10,1e-6,20,L);
    u = u + du;
    norm_p = norm(du);
    norm_u = norm(u);
    fprintf('\t norm(p)=%0.10f; tol*(1+norm(u))=%0.10f\n', norm_p, tol*(1+norm_u))

    if norm_p < tol*(1+norm_u) || i == iter_max
      fprintf('%d: ', i)
      fprintf('norm(exp(u), inf)=%0.10f | norm(u, 2)=%0.10f\n', ...
        norm(exp(u),Inf)/sqrt(n), norm(u,2))
      break
    end
  end
  u2d = reshape(u, [n,n]);
  pu = padarray(u2d, [1,1], 0);
  [X, Y] = meshgrid(0:1/(n+1):1);
  figure;
  surf(X, Y, pu)
end

