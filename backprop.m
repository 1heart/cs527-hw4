function [g, e] = backprop(net, loss, xn, yn)

[y, x, a] = cnn(xn, net);

L = size(net, 1);

[e, ey] = loss(yn, y);

for l=L:-1:1
    curNet = net(l);
    curKernel = curNet.kernel;
    [~, dhda] = curNet.h(a{l});
    
    ew = [];
    ex = [];
    for j = size(curKernel, 3):-1:1
        m = size(x{l}, 1);
        n = size(curKernel(j), 1);
        p = m - n + 1;
        ea = ey(:, j) .* dhda(:, j);
        exComponent = convn(dilute(ea, p), reverse(curKernel(:, :, j)), 'full');
        if isempty(ex)
            ex = exComponent;
        else
            ex = ex + exComponent;
        end
        convResult = convn(dilute(ea, p), reverse(x{l}), 'full');
        ek = middle(convResult, n);
        eb = sum(ea);
        ew(:, j) = [ek(:); eb];
    end
    ey = ex;
end

g = ew(:);

end