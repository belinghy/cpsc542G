T = 10000;
L = 2:1:6;

errors = zeros(4, length(L)); % max error in [r phi];
index = 1;
for l=L
  h = 2^-l;
  tspan = [0,T]; % T = 10
  t = tspan(1):h:tspan(2);
  N = length(t)-1;
  u = zeros(2,N+1); % u = [x y]'
  u(:,1) = [1/sqrt(2);1/sqrt(2)]; % x(0)=y(0)=1/sqrt(2)
  rk2_u = u;
  rk4_u = u;
  
  for i=1:N
    % rk2
    K1 = feval(@func, t(i),   rk2_u(:,i));
    K2 = feval(@func, t(i)+h, rk2_u(:,i)+h*K1);
    rk2_u(:,i+1) = rk2_u(:,i) + h/2*(K1 + K2);
    
    % rk4
    K1 = feval(@func, t(i),      rk4_u(:,i));
    K2 = feval(@func, t(i)+.5*h, rk4_u(:,i)+.5*h*K1);
    K3 = feval(@func, t(i)+.5*h, rk4_u(:,i)+.5*h*K2);
    K4 = feval(@func, t(i)+h,    rk4_u(:,i)+h*K3);
    rk4_u(:,i+1) = rk4_u(:,i) + h/6*(K1 + 2*K2 + 2*K3 + K4);
  end
  
  errMAR = @(x, y) max(abs(sqrt(x.^2 + y.^2)-1));
  errMAP = @(x, y) errMAPh(x,y,h);
  errors(1,index) = errMAR(rk2_u(1,:),rk2_u(2,:));
  errors(2,index) = errMAP(rk2_u(1,:),rk2_u(2,:));
  errors(3,index) = errMAR(rk4_u(1,:),rk4_u(2,:));
  errors(4,index) = errMAP(rk4_u(1,:),rk4_u(2,:));
  index = index + 1;
end

figure;
p1 = subplot(2,2,1);
plot(p1, L, errors(1,:));
p2 = subplot(2,2,2);
plot(p2, L, errors(2,:));
p3 = subplot(2,2,3); 
plot(p3, L, errors(3,:));
p4 = subplot(2,2,4);
plot(p4, L, errors(4,:));


function err=errMAPh(x,y,h)
  phi = atan(y./x);
  err = abs((phi(2:end)-phi(1:end-1))/h-1);
  err = err(err <= 1);
  err = max(err);
end

function f=func(t,u)
  rf = @(x, y) sqrt(x^2 + y^2);
  pf = @(r, mu) (r^(-2))*((1-r^2)^mu);
  qf = @(r, beta) 1+(1-r^2)^beta;
  mu = 3; beta = 2;
  pf = @(r) pf(r, mu);
  qf = @(r) qf(r, beta);
  xpf = @(p,x,q,y) p*x - q*y;
  ypf = @(p,x,q,y) q*x + p*y;
  
  x = u(1); y = u(2); r = rf(x,y); p = pf(r); q = qf(r);
  f(1) = xpf(p,x,q,y);
  f(2) = ypf(p,x,q,y);
  f=f'; % Just putting into row vector
end
