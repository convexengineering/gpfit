[X, Y] = meshgrid(1:5,1:5);
Z = X.^2+Y.^2;
% surf(X,Y,Z);
% hold all;
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
% surf(u1,u2,w_MA);
% hold all;
w_MA = exp(B(2)) .* u1.^A(3) .* u2.^A(4);
% surf(u1,u2,w_MA);
% hold all;

% Softmax Affine Fitting
PAR_SMA = s.softmax_optMAinit.params{1};
A = PAR_SMA([2,3,5,6]);
B = PAR_SMA([1,4]);
alpha = PAR_SMA(end);
w_SMA = (exp(alpha.*B(1)) .* u1.^(alpha*A(1)) .* u2.^(alpha*A(2)) +...
        exp(alpha.*B(2)) .* u1.^(alpha*A(3)) .* u2.^(alpha*A(4))...
        ).^(1/alpha);
% surf(u1,u2,w_SMA);

% Implicit Softmax Affine Fitting
PAR_ISMA = s.implicit_originit.params{1};
% alpha = PAR_SMA(end-1:end);
% A = PAR_SMA([2,3,5,6]);
% B = PAR_SMA([1,4]);
% f_ISMA = @(w_ISMA) exp(alpha(1).*B(1))/w_ISMA.^alpha(1) .* uu1.^(alpha(1)*A(1)) .* uu2.^(alpha(1)*A(2)) +...
%     exp(alpha(2).*B(2))/w_ISMA.^alpha(2) .* uu1.^(alpha(2)*A(3)) .* uu2.^(alpha(2)*A(4)) - 1;
w_ISMA = implicit_softmax_affine(x,PAR_ISMA);
w_ISMA = exp(reshape(w_ISMA,5,5))
% surf(u1,u2,w_ISMA);