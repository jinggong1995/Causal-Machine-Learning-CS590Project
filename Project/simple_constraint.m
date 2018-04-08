function [c,ceq] = simple_constraint(beta)
x = data(:,1:3);
c = [k1/(1+exp(-beta(1)-x(3)))-k2/(1+exp(-beta(1)))-k3/(1+exp(-beta(3)))+m2;
    -(k6/(1+exp(-beta(1)-x(3)))-k4/(1+exp(-beta(1)))-k5/(1+exp(-beta(3)))+m1)];
ceq=0;
