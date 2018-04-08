function y = simple_fitness(beta)
yy=data(:,4);
x = data(:,1:3);
y = -sum(yy.*(x*beta.')-log(1+exp(x*beta.')))+0.05*norm(beta);
