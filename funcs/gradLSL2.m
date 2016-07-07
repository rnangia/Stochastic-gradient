function g = gradLSL2(a,x,b,lambda)

a = a(:);
x = x(:);
g = 2*(a'*x - b)*a + lambda*x; 