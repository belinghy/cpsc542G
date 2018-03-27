h = 0.02;
N = 120 / h;
A = [0 -1; 1 0];
y = zeros(2, N+1);  % u = [x y]'
y(:,1) = [1;0];  % u(t=0) = [1 0]'

figure;
% Implicit trapezoid
M = eye(2)-h/2*A;
M2 = eye(2)+h/2*A;
for i=1:N, y(:,i+1) = M \ (M2*y(:,i)); end
plot(y(1,:),y(2,:))
xlabel('$x$','Interpreter','latex')
ylabel('$y$','Interpreter','latex')
title('Implicit Trapezoid','Interpreter','latex')
r1 = zeros(1,N+1);
for i=1:N+1, r1(i)=y(1,i)^2+y(2,i)^2; end

% Integrate using forward Euler 
M = eye(2)+h*A;
for i =1:N, y(:,i+1)=M*y(:,i); end
r2 = zeros(1,N+1);
for i=1:N+1, r2(i)=y(1,i)^2+y(2,i)^2; end

% Integrate using backward Euler 
M = eye(2)-h*A;
for i =1:N, y(:,i+1)=M\y(:,i); end
r3 = zeros(1,N+1);
for i=1:N+1, r3(i)=y(1,i)^2+y(2,i)^2; end

figure
subplot(1,2,1)
hold on
plot(r1)
plot(r2)
plot(r3)
ylabel('$r^2 = x^2 + y^2$','Interpreter','latex')
legend('Implicit Trapezoidal', 'Forward Euler', 'Backward Euler')
subplot(1,2,2)
plot(r1)

