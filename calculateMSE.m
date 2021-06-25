function mse = calculateMSE(num, outputs, labels)
%CALCULATEMSE Summary of this function goes here
%   Detailed explanation goes here
mse = 0;
for n = 1 : num
    error = outputs(n, 1) - labels(n, 1);
    mse = mse + error^2;
end
mse = mse / num; % using mean square error
end

