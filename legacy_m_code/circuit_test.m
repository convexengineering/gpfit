close all
vdd = linspace(1.0,2.0,30);
vth = linspace(0.2,0.4,30);
[VDD, VTH] = meshgrid(vdd, vth);


P = VDD.^2 + 30.*VDD.*exp(-(VTH-0.06.*VDD)/0.039);
surf(VDD,VTH,P)
hold on;
% surf(log(VDD),log(VTH),log(P))
w = P(:);
% [w, i] = sort(w);


VDD = VDD(:);
% VDD = VDD(i);
VTH = VTH(:);
% VTH = VTH(i);
u = [VDD,VTH];

x = log(u);
y = log(w);

% plot(x(:,2),y)
s = compare_fits(x,y, 3,1);
PAR = s.softmax_optMAinit.params{1};
% B = PAR(1:3:end-1);
% A = PAR([2,3,5,6,8,9]);
B = PAR(1:3:end-2);
A = PAR([2,3,5,6,8,9]);

alpha = 1/PAR(end)
coeff = exp(alpha*B)
powers = alpha.*A

[VDD, VTH] = meshgrid(vdd, vth);
P_approx = (...
            coeff(1).*VDD.^(powers(1)).*VTH.^(powers(2))+ ...
            coeff(2).*VDD.^(powers(3)).*VTH.^(powers(4)) + ...
            coeff(3).*VDD.^(powers(5)).*VTH.^(powers(6))...
            ).^(1/alpha);
surf(VDD,VTH,P_approx)