function [y, x, a] = cnn(x1, net)

ok(net);

x = cell(size(net, 1), 1);
a = cell(size(net, 1), 1);

for i = 1:size(net, 1)
    x{i} = x1;
    curNet = net(i);
    curKernels = curNet.kernel;
    output = [];
    curStride = curNet.stride;
    for j = 1:size(curKernels, 3)
        output(:,j) = convn(x1, curKernels(:, :, j), 'valid') + curNet.bias(j);
    end
    x1 = output(1:curStride:end, :);
    a{i} = x1;
    [x1, dhda] = curNet.h(x1);
end

y = x1;

end