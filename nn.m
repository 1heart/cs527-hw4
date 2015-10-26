function X = nn(X, net)

for i = 1:size(net, 2)
    curNet = net(i);
    W = [curNet.gain curNet.bias];
    xTilde = [X; ones(1, size(X, 2))];
    a = W*xTilde;
    X = curNet.h(a);
end

end