figure;
x = linspace(0.1, 1, 1000);
y = x + log(x);
plot(x,y);
ylim([-2.5 1])
hold on
plot([0.5 0.5], [-2.5 1], 'r--')
plot([0.6 0.6], [-2.5 1], 'r--')
plot([0.1 1], [-0.19 -0.19], 'r--')
plot([0.1 1], [0.09 0.09], 'r--')
title('x + lnx');