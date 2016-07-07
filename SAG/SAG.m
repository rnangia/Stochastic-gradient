clear;
load rcv_bin.mat

[N,D] = size(A); 
niters = 2*N;
d = 0;
alpha = 100;
y = zeros(D,1);
x = rand(D,1);
for k = 1:niters
    i = randi(N);
    d = d - y;
    y = gradLS(A(i,:),x,b(i));
    d = d + y;
    x = x - (alpha/N)*d;
    fval = norm(A*x-b)^2;
     if mod(k,100) == 0
        fprintf ('k = %d of %d ; value = %f \r', k, niters,fval)
     end
end


