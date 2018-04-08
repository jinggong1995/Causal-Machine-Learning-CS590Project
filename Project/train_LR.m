function y = train_LR(X,Y)
kk=size(X,2);
w = zeros(kk,1);
count=0;
tol=1;
pred=zeros(size(X,1),1);
while (count<100) && (tol>1e-2)
    w_old=w;
    for i=1:256
        pred(i)=1/(1+exp(-X(i,:)*w));
    end
    for j=1:kk
        gradient = 0.05*w(j)-(Y-pred).'*X(:,j);
        w(j)=w(j)-0.05*gradient;
    end
    tol=norm(w-w_old);
    count=count+1;
end
y=w;

        
