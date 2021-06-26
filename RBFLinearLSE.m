function [w, width, phi] = RBFLinearLSE(W, data, label, som_size, num)
%RBFLINEARLSE Summary of this function goes here
%   Determine the weights of RBF using linear least square estimation

dmax = 0;
for i = 1 : 15
    for j = i + 1 : 16
        d = dist(W(i, :), W(j, :)');
        if d > dmax
            % find the maximum distance between the chosen centers
            dmax = d;
        end
    end
end

% calculate the width of radial basis function
width = dmax / sqrt(2 * 16);

phi = [];
for p = 1 : som_size
    for q = 1 : num
        % using Gaussian as radial basis function
        phi(q,p) = exp(-(dist(data(q, :), W(p, :)'))^2 ...
            / (2 * width^2));
    end
end

% calculate the weights using linear least square estimates
w = inv(phi' * phi) * phi' * label;

end

