f = @(x) cos(30*pi*x);
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
  yi = f(xi);
  p = lagrangePoly(xi, yi);
  y_interp = p(x);
  errors(loop_index) = max(abs(y_real-y_interp));
  plot(x, y_interp);
  loop_index = loop_index + 1;
  pause;
end

figure;
semilogy(ns, errors);
xlabel('$n\ (degree)$','Interpreter','latex')
ylabel('$max.\ error$','Interpreter','latex')

function f = lagrangePoly(xi, yi)
  xlength = length(xi);
  w = zeros(1,xlength);
  for i=1:xlength
    cur_x = xi(i);
    cur_x = ones(1,xlength) .* cur_x;
    w(i) = 1 / prod(cur_x([1:i-1 i+1:end]) - xi([1:i-1 i+1:end]));
  end
  
  function y=evalPoly(x)
    x_new = repmat(x, [xlength,1]); 
    xi_new = repmat(xi', [1,length(x)]);
    invDiff = 1 ./ (x_new - xi_new);
    numerator = (w.*yi)*(invDiff);
    denom = w*invDiff;
    y = numerator ./ denom;
  end

  f = @evalPoly;
end
