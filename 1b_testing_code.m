N = 101;
T = 0.05;
f = @(x) exp(x) .* sin(3 * pi * x);
net = approximator(f, T);
x = linspace(0, 1, N);
y = nn(x, net);
clf
plot(x, f(x), 'b')
hold on
plot(x, y, 'r')