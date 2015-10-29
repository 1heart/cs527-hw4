function [g, e] = numeric(net, loss, xn, yn)

d = 10^-6;

% y is output from the last layer of the net
% x{l} is the input matrix to layer l
% a{l} is the activation matrix in layer l
[y, x, a] = cnn(xn, net);

L = size(net, 1);
ey = cell(L, 1);

[e, ey{L}] = loss(yn, y);

for l=L:-1:1
    curNet = net(l);
    curKernels = curNet.kernel;
    for j = size(curKernels, 3):-1:1
        curKernel = curKernels(:, :, j);
        
    end
end

g = ew(:);

end