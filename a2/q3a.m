h = 0.02;
N = 120 / h;
A = [0 -1; 1 0];
u = zeros(2, N+1);  % u = [x y]'
u(:,1) = [1;0];  % u(t=0) = [1 0]'

figure;
% Implicit trapezoid
M = eye(2)-h/2*A;
M2 = eye(2)+h/2*A;
for i=1:N, y(:,i+1) = M \ (M2*y(:,i)); end
plot(y(1,:),y(2,:))
xlabel('$x$','Interpreter','latex')
ylabel('$y$','Interpreter','latex')
title('Implicit Trapezoid','Interpreter','latex')