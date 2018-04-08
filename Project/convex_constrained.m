

n=200;
x1 = zeros(2*n,1);
x1(1:n) = rand(n,1);
x1(n+1:2*n) = -rand(n,1);
x2 = zeros(2*n,1);
x2(1:n) = rand(n,1);
x2(n+1:2*n) = -rand(n,1);
y1 = zeros(2*n,2*n);
y2 = zeros(2*n,2*n);

for i =1:2*n 
    for j =1:2*n
        y1(i,j)= 1/(1+exp(-x1(i)-x2(j)))-1/(1+exp(-x2(j)))+1/(1+exp(-x1(i)));
        y2(i, j) = x1(i)+x2(j);
    end
end



k1=0.2;
k2=-0.09;
k3=0.5;
[X,Y] = meshgrid(-3:.1:3);
Z = k1./(1+exp(-X-Y))-k2./(1+exp(-Y))-k3./(1+exp(-X));
%%s = surf(X,Y,Z,'FaceAlpha',0.5)


yy=data(:,4);
p = data(:,5);
x = data(:,1:3);
z = data(:, 1);
w = data(:, 3);


ObjectiveFunction = @simple_fitness;
nvars = 3;    % Number of variables
LB = [-3 -3 -3];   % Lower bound
UB = [3 3 3];  % Upper bound
ConstraintFunction = @simple_constraint;
rng(1,'twister') % for reproducibility
[beta,fval] = ga(ObjectiveFunction,nvars,...
    [],[],[],[],LB,UB,ConstraintFunction)

1/(1+exp(-x*beta.'))
























