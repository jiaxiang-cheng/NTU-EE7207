%% EE7207 ASSIGNMENT 1 - CHENG JIAXIANG G2003852A

clear all

load('data_test.mat')
load('data_train.mat')
load('label_train.mat')

%% Self-Organizing Phase - SOM

sigma0 = 2.121; % initial value of the width
t1 = 1000/log(sigma0); % time constant for neighborhood function
W = -1 + 2.*rand(16,33,'double');  % weights initialization
Delta = []; 

for itr = 1:1000    % 1000 iterations for ordering phase 
    
    %disp(itr)
    disp(W)
    ita = 0.1*exp(-itr/1000);   % learning rate updating
    sigma = sigma0*exp(-itr/t1);    % neighborhood width updating
    delta = 0;

    for num = 1:330 % 330 samples
        
        x = data_train(num,:);  % select samples sequentially
        dmin = 100000;

        for i = 1 : 16
            w = W(i,:);
            d = dist(x, w');
            if d < dmin
                dmin = d;
                imin = i;
            end
        end   % find the winning neuron with index imin

        wwinx = mod(imin,4)-1;
        if wwinx == -1
            wwinx = 3;
        end
        wwiny = floor((imin-1)/4);
        % find the location of winning neuron (wwinx, wwiny)

        for j = 1 : 16
            w = W(j,:);
            wwin = W(imin,:);
            wx = mod(j,4)-1;
            if wx == -1
                wx = 3;
            end
            wy = floor((j-1)/4);
            % find the location of neuron j (wx, wy)
            dji = sqrt((wx-wwinx)^2+(wy-wwiny)^2);
            % calculate the distance between the neuron j & winning neuron
            W(j,:) = W(j,:) + ita*exp(-dji^2/(2*sigma^2))*(x-W(j,:));
            % update the weights of neuron j
            delta = delta + abs(ita*exp(-dji^2/(2*sigma^2))*(x-W(j,:)));
        end
    end
    Delta = [Delta, delta'/(16*330)];
end

Delta = mean(Delta, 1);
plot(Delta)

% save('centre_vectors.mat','W')

%% Convergence Phase - SOM

Delta1 = [];

for itr = 1 : 500*1 %*16    % 1000 iterations for ordering phase 
    
    disp(itr)
    ita = 0.01;   % learning rate updating
    delta1 = 0;

    for num = 1 : 330 % 330 samples
        
        x = data_train(num,:);  % select samples sequentially
        dmin = 100000;

        for i = 1 : 16
            w = W(i,:);
            d = dist(x, w');
            if d < dmin
                dmin = d;
                imin = i;
            end
        end   % find the winning neuron with index imin
        W(imin,:) = W(imin,:) + ita*(x-W(imin,:));
        % only update the weights of winning neuron
        delta1 = delta1 + abs(ita*(x-W(imin,:)));
    end
    disp(W)
    Delta1 = [Delta1, delta1'/330];
end

save('centre_vectors.mat','W')

Delta1 = mean(Delta1, 1);
plot(Delta1)

%% Determine the weights of RBF using linear least square estimation

dmax = 0;
for i = 1 : 15
    for j = i+1 : 16
        dd = dist(W(i,:), W(j,:)');
        if dd > dmax
            dmax = dd;
            % find the maximum distance between the chosen centers
        end
    end
end        
width_RBF = dmax/sqrt(2*16);
% calculate the width of radial basis function

phi = [];
for p = 1 : 16
    for q = 1 : 330
        phi(q,p) = exp(-(dist(data_train(q,:),W(p,:)'))^2/(2*width_RBF^2));
        % using Gaussian as radial basis function
    end
end
        
w_RBF = inv(phi'*phi)*phi'*label_train;
% calculate the weights using linear least square estimates

%% Test classification accuracy of the training data

output = phi*w_RBF;

E = 0;
for n = 1 : 330
    e = output(n,1) - label_train(n,1);
    E = E + e^2;
end
E = E/330; % using mean square error
        
%% Predict the labels for testing data

phi_t = [];
for p = 1 : 16
    for q = 1 : 21
        phi_t(q,p) = exp(-(dist(data_test(q,:),W(p,:)'))^2/(2*width_RBF^2));
    end
end

f_output = phi_t*w_RBF;
        