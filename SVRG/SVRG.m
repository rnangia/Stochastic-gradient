clear;
addpath('../funcs');
load ../SAG/rcv_bin.mat
[N,D] = size(A); 
niters = 50;
m = 2*N; %2n-usual choice for convex problems, 5n for non-convex
eta = 0.01;
option = 1;

%w = sparse(D,m+1);
w = rand(D,1);
w_ = rand(D,1);
for s = 1:niters
    %w_ = ws_1;
    mu_ = 0;
    for i = 1:N
        mu_ = mu_ + gradLS(A(i,:),w_,b(i));
    end
    mu_ = mu_/N;
    %w(:,1) = w_;   %Initializing w_0
    w = w_;
    for t = 2:m+1
        i_t = randi(N);
        %w(:,t) = w(:,t-1) - eta*(gradLS(A(t,:),w(:,t-1),b(t)) - gradLS(A(t,:),w_,b(t)) + mu_);
        w = w - eta*(gradLS(A(t,:),w,b(t)) - gradLS(A(t,:),w_,b(t)) + mu_);
        if mod(t,100) == 0
%            fval = norm(A*w(:,t)-b)^2;
            fval = norm(A*w-b)^2;
            fprintf ('t = %d of %d ; value = %f \r', t, m+1,fval)
        end
    end
    switch(option)
        case 1 
            w_ = w;
        case 2
            w_ = w(:,randi(m));
    end 
    %fval = norm(A*w(:,m+1)-b)^2;
    fval = norm(A*w-b)^2;
    fprintf ('s = %d of %d ; value = %f \r', s, niters,fval)
end
