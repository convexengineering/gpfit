[X, Y] = meshgrid(1:0.1:5,1:0.1:5);
Z = (X./Y + Y./X).^0.5;
surf(X,Y,Z);

hold all;
w = Z(:);
X = X(:);
Y = Y(:);

u = [X,Y];

x = log(u);
y = log(w);

s = compare_fits(x,y, 2,1);

[u1, u2] = meshgrid(1:5,1:5);

% Max Affine Fitting
PAR_MA = s.maxaffine.params{1};
A = PAR_MA([2,3,5,6]);
B = PAR_MA([1,4]);
w_MA = exp(B(1)) .* u1.^A(1) .* u2.^A(2);
surf(u1,u2,w_MA);
hold all;
w_MA = exp(B(2)) .* u1.^A(3) .* u2.^A(4);
surf(u1,u2,w_MA);
hold all;

% Softmax Affine Fitting
% % PAR_SMA = s.softmax_optMAinit.params{1};
% % A = PAR_SMA([2,3,5,6]);
% % B = PAR_SMA([1,4]);
% % alpha = 1/PAR_SMA(end);
% % w_SMA = (exp(alpha.*B(1)) .* u1.^(alpha*A(1)) .* u2.^(alpha*A(2)) +...
% %         exp(alpha.*B(2)) .* u1.^(alpha*A(3)) .* u2.^(alpha*A(4))...
% %         ).^(1/alpha);
% % surf(u1,u2,w_SMA);