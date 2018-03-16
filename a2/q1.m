% Langrange seems to be best choice
% Monomial is not stable enough
% Don't need adaptive of Newton

f = @(x) cos(30*pi*x);
f_xi = @(i,n) cos((2*i+1)/(2*n+2)*pi);
x = -1:.001:1;
y_real = f(x);

figure;
hold on;
plot(x, y_real);

ns = 10:10:80;
errors = zeros(1,length(ns));
loop_index = 1;
for n = ns
  xi = f_xi(0:n,n);
  A = vander(xi); % Vandermonde, flipped
  yi = f(xi);
  [coeffs,fl] = gmres(A,yi',size(A,1),1e-5); % backslash blows up  
  y_interp = polyval(coeffs,x); % Polyval expect flipped
  errors(loop_index) = max(abs(y_real-y_interp));
  plot(x, y_interp);
  loop_index = loop_index + 1;
end

figure;
semilogy(ns, errors);
xlabel('$n\ (degree)$','Interpreter','latex')
ylabel('$max.\ error$','Interpreter','latex')