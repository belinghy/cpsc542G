
f = @(x) x + log(x);
g = @(x) x - x*f(x);

x0 = 0.5;
x1 = g(x0);
index = 1;

while abs(x1 - x0) > 1E-10
  index = index + 1;
  x0 = x1;
  x1 = g(x0);
  fprintf('%d: x*~ = %0.10f; f = %0.11f\n', index, x1, f(x1))
end