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

for n = 10:10:80
  xi = f_xi(0:n,n);
  A = vander(xi); % Vandermonde, flipped
  yi = f(xi);
  coeffs = A\yi';
  
  y_interp = polyval(coeffs,x); % Polyval expect flipped
  error = abs(y_real-y_interp);
  plot(x, y_interp);
  max(error)
end