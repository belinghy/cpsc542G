e = exp(1);
f = @(x) e.^(3*x).*sin(200*x.^2)./(1+20*x.^2);
f_xi = @(n) (0:n)/n;
x = 0:.001:1;
y_real = f(x);

j_max = 14;
j_min = 4;
errors = zeros(1,j_max-j_min+1);
for j=j_min:j_max
  n = 2^j;
  xi = f_xi(n);
  yi = f(xi);
  pp = spline(xi, yi);
  y_interp = ppval(pp, x);
  errors(j-j_min+1) = max(abs(y_real-y_interp));
end

figure;
loglog(2.^(j_min:j_max),errors);
xlabel('$n = 2^j$','Interpreter','latex')
ylabel('$max.\ error$','Interpreter','latex')