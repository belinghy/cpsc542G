a = -30;
b = 30;
f = @(x) cos(30*pi*x);
f_down = @(x) cos(pi*x); % downsample
f_xi = @(i,n) cos((2*i+1)/(2*n+2)*pi);
x = -1:.001:1;
y_real = f(x);

figure;
hold on;
plot(x, y_real);

ns = 10:10:170;
errors = zeros(1,length(ns));
loop_index = 1;
for n = ns
  xi = f_xi(0:n,n);
  xi = a+(b-a)/2.*(xi+1); % scale to range
  A = vander(xi); % Vandermonde, flipped
  yi = f_down(xi);
  %[coeffs,fl] = gmres(A,yi',size(A,1),1e-5); % backslash blows up  
  coeffs = A\yi';
  y_interp = polyval(coeffs,x*30); % Polyval expect flipped
  errors(loop_index) = max(abs(y_real-y_interp));
  plot(x, y_interp);
  loop_index = loop_index + 1;
  pause;
end

figure;
semilogy(ns, errors);
xlabel('$n\ (degree)$','Interpreter','latex')
ylabel('$max.\ error$','Interpreter','latex')