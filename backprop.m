function [g, e] = backprop(net, loss, xn, yn)

g = zeros(1, size(net, 1));

% y is output from the last layer of the net
% x{l} is the input matrix to layer l
% a{l} is the activation matrix in layer l
[y, x, a] = cnn(xn, net);

L = size(net, 1);
ey = cell(L, 1);

[e, ey{L}] = loss(yn, y);

for l=L:-1:2
    curNet = net(l);
    curKernel = curNet.kernel;
    [~, dhda] = curNet.h(a{l});
    curEy = ey{l};
    
    ex = [];
    ew = [];
    for j = size(curKernel, 3):-1:1
        ea = curEy(:, j) .* dhda(:, j);
        p = size(a{l}, 1);
        n = size(x{l}, 1) + 1 - p;
        if isempty(ex)
            ex = convn(dilute(ea, p), reverse(curKernel(:, :, j)), 'full'); 
        else
            ex = ex + convn(dilute(ea, p), reverse(curKernel(:, :, j)), 'full'); 
        end
        ek = middle(convn(dilute(ea, p), reverse(x{l}), 'full'), n);
        eb = sum(ea);
        ew(:, j) = [ek(:); eb];
    end
    ey{l - 1} = ex;
end

g = ew(:);

end