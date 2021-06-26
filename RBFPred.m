function output = RBFPred(W, w_RBF, data_test, ntest, size, width)
%RBFPRED Summary of this function goes here
%   Predict the labels for testing data

phi = [];
for p = 1 : size
    for q = 1 : ntest
        phi(q, p) = exp(-(dist(data_test(q, :), W(p, :)'))^2 ...
            / (2 * width^2));
    end
end

output = phi * w_RBF;

end

