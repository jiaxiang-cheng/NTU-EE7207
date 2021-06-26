clear all

% loading data and get info
load('./data_set/data_train.mat')
load('./data_set/label_train.mat')
load('./data_set/data_test.mat')
ntrain = size(data_train, 1);
ntest = size(data_test, 1);

%% Training with SOM and RBF

% size of SOM
size_1 = 16; size_2 = 33;
% initial value of the width
sigma_0 = 2.121;
% time constant for neighborhood function
t1 = 1000 / log(sigma_0);
% weights initialization
W = -1 + 2. * rand(size_1, size_2, 'double');

% find center vectors using SOM
W = SOMSelfOrganizing(W, data_train, t1, sigma_0);
W = SOMConvergence(W, data_train);

%% learn weights for RBF neural networks
[w_RBF, width_RBF, phi] = RBFLinearLSE(W, data_train, label_train, ...
    size_1, ntrain);

%% Test classification accuracy of the training data

output = phi * w_RBF;
mse = calculateMSE(ntrain, output, label_train);

%% Predict the labels for testing data

pred = RBFPred(W, w_RBF, data_test, ntest, size_1, width_RBF);
