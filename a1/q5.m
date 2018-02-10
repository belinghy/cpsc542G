% Newton's Method 2D Poisson Problem
n = 2^7 - 1;
iter_max = 4;
% initial guess for u, (n*n, 1)
u = zeros(n*n, 1);
for i=1:iter_max
  fu = f(u);
  Ju = J(u);
  du = -Ju\fu; % search direction = p
  u = u + du;
end
u2d = reshape(u, [n,n]);
pu = padarray(u2d, [1,1], 0);
[X, Y] = meshgrid(0:1/(n+1):1);
surf(X, Y, pu)


function u1=f(u)
  n = sqrt(length(u)); % u is a long array
  h = 1/(n+1); % n + 1 gaps in each dimension
  
  u2d = reshape(u, [n, n]); % Get u into 2d for ease of calculation
  pu = padarray(u2d, [1,1], 0); % pad a border of 0s
  
  % y_{ij} = 1/h^2 * (-u_{i+1,j} -u_{i-1,j} +4u_{i,j}
  %     -u_{i,j+1} - u_{i,j-1}) - e^(u_{i,j})

  u1 = zeros(n+2, n+2);
  for i=2:n+1
    for j=2:n+1
      u1(i,j) = 1/h^2 * (-pu(i+1,j) -pu(i-1,j) ...
        +4*pu(i,j) -pu(i,j+1) -pu(i,j-1)) - exp(pu(i,j));
    end
  end
  
  u1 = u1(2:n+1, 2:n+1);
  u1 = reshape(u1, [n*n, 1]);
  
end

function J1=J(u)
  n = sqrt(length(u)); % u is a long array
  h = 1/(n+1); % n + 1 gaps in each dimension
  
  J1 = zeros(n^2, n^2);
  for i=1:n^2
    % diagonal
    J1(i,i) = 4/h^2 - exp(u(i));
    if i+1 <= n^2
      J1(i,i+1) = -1/h^2;
    end
    if i-1 >= 1
      J1(i,i-1) = -1/h^2;
    end
    if i+n <= n^2
      J1(i,i+n) = -1/h^2;
    end
    if i-n >= 1
      J1(i,i-n) = -1/h^2;
    end
  end
end