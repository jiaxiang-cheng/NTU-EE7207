clear all

load('./data_set/data_train.mat')
load('./data_set/label_train.mat')
load('./data_set/data_test.mat')
ntrain = length(data_train);
ntest = length(data_test);

sigma0 = 2.121; % initial value of the width
t1 = 1000 / log(sigma0); % time constant for neighborhood function
W = -1 + 2. * rand(16, 33, 'double');  % weights initialization

%% Self-Organizing Phase - SOM

W = SOMSelfOrganizing(W, data_train, t1, sigma0);

%% Convergence Phase - SOM

W = SOMConvergence(W, data_train);

%% Determine the weights of RBF using linear least square estimation

dmax = 0;
for i = 1 : 15
    for j = i + 1 : 16
        dd = dist(W(i, :), W(j, :)');
        if dd > dmax
            % find the maximum distance between the chosen centers
            dmax = dd;
        end
    end
end

% calculate the width of radial basis function
width_RBF = dmax / sqrt(2 * 16);

phi = [];
for p = 1 : 16
    for q = 1 : ntrain
        % using Gaussian as radial basis function
        phi(q,p) = exp(-(dist(data_train(q, :), W(p, :)'))^2 ...
            / (2 * width_RBF^2));
    end
end

% calculate the weights using linear least square estimates
w_RBF = inv(phi' * phi) * phi' * label_train;

%% Test classification accuracy of the training data

output = phi * w_RBF;

mse = calculateMSE(ntrain, output, label_train);

%% Predict the labels for testing data

phi_t = [];
for p = 1 : 16
    for q = 1 : ntest
        phi_t(q, p) = exp(-(dist(data_test(q, :), W(p, :)'))^2 ...
            / (2 * width_RBF^2));
    end
end

f_output = phi_t * w_RBF;
