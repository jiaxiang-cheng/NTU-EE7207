function W = SOMConvergence(W, data)
%SOMCONVERGENCE Summary of this function goes here
%   Convergence Phase - SOM

Delta = [];

for itr = 1 : 500  % 1000 iterations for ordering phase

    disp(W)
    ita = 0.01;  % learning rate updating
    delta1 = 0;
    
    for num = 1 : 330 % 330 samples
        
        x = data(num, :);  % select samples sequentially
        dmin = 100000;
        
        for i = 1 : 16
            w = W(i, :);
            d = dist(x, w');
            if d < dmin
                dmin = d;
                imin = i;
            end
        end   % find the winning neuron with index imin
        
        % only update the weights of winning neuron
        W(imin, :) = W(imin, :) + ita * (x - W(imin, :));
        
        % store the absolute updates on weights
        delta1 = delta1 + abs(ita * (x - W(imin, :)));
        
    end
    
    % store the absolute updates on weights of each iteration
    Delta = [Delta, delta1' / 330];
    
end

Delta = mean(Delta, 1);
plot(Delta)  % observe updates on weights

% center vectors chosen after convergence phase
save('centre_vectors.mat', 'W')

end
