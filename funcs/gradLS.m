function g = gradLS(a,x,b)

a = a(:);
x = x(:);
g = 2*(a'*x - b)*a; 