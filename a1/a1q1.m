figure;
x0 = 1.2;
f0 = sin(x0);
fp = cos(x0);
i = -20:0.5:0;
h = 10.^i;
err = abs(fp - (sin(x0+h) - f0)./h);
d_err = f0/2*h;

subplot(1,2,1);
loglog (h,err,'-*');
hold on
loglog(h,d_err,'r-.');
ylim([1e-50 1])
title('[f(x+h)-f(x)]/h')
xlabel('h')
ylabel('Absolute error')


% both end
x0 = 1.2;
f0 = sin(x0);
fp = cos(x0);
i = -20:0.5:0;
h = 10.^i;
err = abs(fp - (sin(x0+h) - sin(x0-h))./(2*h));
d_err = h.^2/6*fp;

subplot(1,2,2);
loglog (h,err,'-*');
hold on
loglog(h,d_err,'r-.');
title('[f(x+h)-f(x-h)]/2h')
ylim([1e-50 1])
xlabel('h')
ylabel('Absolute error')