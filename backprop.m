function [g, e] = backprop(net, loss, xn, yn)

g = zeros(1, size(net, 1));

% y is output from the last layer of the net
% x{l} is the input matrix to layer l
% a{l} is the activation matrix in layer l
[y, x, a] = cnn(xn, net);

L = size(net, 1);
ey = cell(L, 1);

[e, ey{L}] = loss(yn, y);

for l=L:-1:1
    curNet = net(l);
    curKernel = curNet.kernel;
    [yL, dhda] = curNet.h(a{l});
    curEy = ey{l};
    
    ea = [];
    ex = [];
    ew = [];
    for j = size(curKernel, 3):-1:1
        ea(:, j) = curEy(:, j) .* dhda(:, j);
        p = size(a{l}, 1);
        if isempty(ex)
            ex = convn(delta(ea(:, j), p), reverse(curKernel(:, :, j)), 'full'); 
        else
            ex = ex + convn(delta(ea(:, j), p), reverse(curKernel(:, :, j)), 'full'); 
        end
        ek = middle(convn(delta(ea(:, j), p), reverse(x{l}), 'full'), n);
        eb = sum(ea(:, j));
        ew(:, j) = [ek(:); eb];
    end
    ey{l - 1} = ex;
end

g = ew(:);

end