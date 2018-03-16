f = @(x) cos(30*pi*x);
f_xi = @(i,n) cos((2*i+1)/(2*n+2)*pi);
x = -1:.001:1;
y_real = f(x);

figure;
hold on;
plot(x, y_real);

for n = 10:10:40
  xi = f_xi(0:n,n);
  yi = f(xi);
  coeffs = lagrangepoly(xi,yi); % not stable enough to use backslash
  
  y_interp = polyval(coeffs,x);
  error = abs(y_real-y_interp);
  plot(x, y_interp);
end

function coeffs=lagrangepoly(x,y)
  coeffs=0;
  for i=1:length(x)
    p=1;
    for j=1:length(x)
      if j~=i
        c = poly(x(j))/(x(i)-x(j));
        p = conv(p,c);
      end
    end
    term = p*y(i);
    coeffs= coeffs + term;
  end
end