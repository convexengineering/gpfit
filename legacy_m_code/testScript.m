% x = linspace(0,10,11);
% x = x';
x1 = 0:15;
x2 = x1.^2;
x3 = x1.^3;
% x = x1';
x = [x1', x2'];
% x = [x1', x2', x3'];

% y1 = x(1:round(end/2));
% y2 = 2*x(round(end/2) + 1:end);
% y = [y1; y2];
y = [0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20]';

Ks = 2;
ntry = 1;

s = compare_fits(x,y,Ks,ntry);

disp(s.maxaffine.params{1})
disp(s.softmax_optMAinit.params{1})
disp(s.softmax_originit.params{1})
disp(s.implicit_originit.params{1})