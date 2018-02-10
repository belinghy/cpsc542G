% Newton's Method for Boundary Value Problem
tol = 1.e-10;
iter_max = 10;

figure
ax1 = subplot(3,1,1);
xlabel('x')
ylabel('y')
ax2 = subplot(3,1,2);
ylabel('ln(relative error)')
ax3 = subplot(3,1,3);
xlabel('Num. Iterations')
ylabel('ln(norm(f))')

hold(ax1,'on')
hold(ax2,'on')
hold(ax3,'on')
for n=2.^linspace(3,6,4)
  x = linspace(1, iter_max+1, iter_max+1);
  [err, norm_fy, fx, result, solution] = finite_diff(n, tol, iter_max);
  plot(ax1, fx, solution, '-')
  plot(ax1, fx, result, 'o')
  plot(ax2, x, log(err))
  plot(ax3, x, log(norm_fy))
end
legend('n=8','16','32','64')

% Actual Newton's Method 
function [r_err, r_norm_fy, fx, result, solution]=finite_diff(n, tol, iter_max)

  x = linspace(1,2,n+2);
  sol = log(x);
  sol = sol(2:n+1);
  y = zeros(n,1); % initial guess

  r_err = zeros(1, iter_max+1);
  r_norm_fy = zeros(1, iter_max+1);
  
  for i=1:iter_max
    fy = f(y); 
    Jy = J(y);
    err = sum(abs(y.' - sol)/abs(sol));
    r_err(i) = err;
    r_norm_fy(i) = norm(fy);
    fprintf('%d: %e \n', i-1, norm(fy))
    fprintf('%d: %e \n', i-1, err)
    dy = -Jy\fy; % search direction = p
    y = y + dy;

    norm_p = norm(dy);
    norm_y = norm(y);
    fprintf('\t norm(p)=%0.10f; tol*(1+norm(x))=%0.10f\n', norm_p, tol*(1+norm_y))
    if norm_p < tol*(1+norm_y) || i == iter_max
      fy = f(y);
      err = sum(abs(y.' - sol)/abs(sol));
      r_err(i+1) = err;
      r_norm_fy(i+1) = norm(fy);
      result = y.';
      solution = sol;
      fx = x(2:n+1);
      fprintf('%d: %e \n', i, norm(fy))
      fprintf('%d: %e \n', i, sum(abs(y.' - sol)/abs(sol)))
      return
    end
  end
end

function y1=f(y)
  n = length(y);
  h = 1/(n+1); % n + 1 gaps
  y1 = zeros(n, 1);
  % set first
  y1(1) = (y(2) - 2*y(1))/h^2 + (y(2)/2/h)^2 + y(1) - log(1+h);
  % set last
  y1(n) = (log(2) - 2*y(n) + y(n-1))/h^2 + ((log(2) - y(n-1))/2/h)^2 + y(n) - log(1+n*h);
  for i=2:n-1
    y1(i) = (y(i+1)-2*y(i)+y(i-1))/h^2 + ((y(i+1)-y(i-1))/2/h)^2 + y(i) - log(1+i*h);
  end
end

function J1=J(y)
  n = length(y);
  h = 1/(n+1); % n + 1 gaps
  J1 = zeros(n, n); % init
  % Manually set first row
  J1(1,1) = -2/h^2 + 1;
  J1(1,2) = 1/h^2 * (1 + 1/2 * (y(2) - 0)); % y_0 = ln(1) = 0
  for i=2:n-1
    J1(i,i-1) = 1/h^2 * (1 - 1/2 * (y(i+1) - y(i-1))); % df/dy_{i-1}
    J1(i,i) = -2/h^2 + 1; % df/dy_i
    J1(i,i+1) = 1/h^2 * (1 + 1/2 * (y(i+1) - y(i-1))); % df/dy_{i-1}
  end
  % Manually set last row
  J1(n,n) = -2/h^2 + 1;
  J1(n,n-1) = 1/h^2 * (1 - 1/2 * (log(2) - y(i-1))); % y_{n+1} = ln(2)
end