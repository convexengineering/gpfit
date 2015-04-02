m = 501;

u = logspace(0,log10(3),501);
u=u';
w = (u.^2 + 3)./(u+1).^2;

x = log(u);
y = log(w);

s = compare_fits(x, y, 3, 1);

