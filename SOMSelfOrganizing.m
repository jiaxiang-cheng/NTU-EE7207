function W = SOMSelfOrganizing(W, data_train, t1, sigma0)
%SOMSelfOrganizing Summary of this function goes here
%   Self-Organizing Phase - SOM

Delta = [];

for itr = 1 : 1000    % 1000 iterations for ordering phase
    
    display(W)
    
    ita = 0.1 * exp(- itr / 1000);   % learning rate updating
    sigma = sigma0 * exp(- itr / t1);    % neighborhood width updating
    delta = 0;
    
    for num = 1 : 330 % 330 samples
        
        x = data_train(num, :);  % select samples sequentially
        dmin = 100000;
        
        for i = 1 : 16
            w = W(i, :);
            d = dist(x, w');
            if d < dmin
                dmin = d;
                imin = i;
            end
        end   % find the winning neuron with index imin
        
        wwinx = mod(imin, 4) - 1;
        if wwinx == -1
            wwinx = 3;
        end
        % find the location of winning neuron (wwinx, wwiny)
        wwiny = floor((imin - 1) / 4);
        
        for j = 1 : 16
            w = W(j, :);
            wwin = W(imin, :);
            wx = mod(j, 4) - 1;
            if wx == -1
                wx = 3;
            end
            
            % find the location of neuron j (wx, wy)
            wy = floor((j - 1) / 4);
            
            % calculate the distance between the neuron j & winning neuron
            dji = sqrt((wx - wwinx)^2 + (wy - wwiny)^2);
            
            % update the weights of neuron j
            W(j, :) = W(j, :) + ita * exp(- dji^2 / (2 * sigma^2)) ...
                * (x - W(j, :));
            
            % store the absolute updates on weights
            delta = delta + abs(ita * exp(- dji^2 / (2 * sigma^2)) ...
                * (x - W(j, :)));
        end
    end
    Delta = [Delta, delta' / (16 * 330)];
end

Delta = mean(Delta, 1);
plot(Delta)  % observe updates on weights

% center vectors chosen after self-organizing phase
% save('center_vectors.mat','W')
end

