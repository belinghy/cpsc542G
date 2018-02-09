% Bisection Method

f = @(x) x + log(x);

x0 = 0.5;
x1 = 0.6;
f0 = f(x0);
f1 = f(x1);
index = 0;

while abs(x1 - x0) >= 1E-10
  index = index + 1;
  mid = (x0 + x1) / 2;
  fmid = f(mid);
  fprintf('%d: x*~ = %0.10f; f = %0.11f\n', index, mid, fmid)
  if f0 * fmid >= 0
      x0 = mid;
  else
      x1 = mid;
  end
end