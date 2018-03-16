f = @(x) x.^2;
f_xi = @(n) linspace(1,2,n);
x = 1:.001:2;
y_real = f(x);

figure;
hold on;
plot(x, y_real);

for n = 1:4
  xi = f_xi(n);
  A = vander(xi); % Vandermonde, flipped
  y = f(xi);
  coeffs = A\y';
  
  y_interp = polyval(coeffs,x); % Polyval expect flipped
  error = abs(y_real-y_interp);
  plot(x, y_interp);
  pause;
end