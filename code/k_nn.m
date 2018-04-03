function [ predictions ] = k_nn(k, training_data, testing_data, tiebreaker)
%K_NN Classifies each test sample based on the label of the majority of its
%k nearest training sample neighbors.
%   predictions = K_NN(k, training_data, testing_data, tiebreaker) will
%   return a matrix with the predictions for each test data point. 
%
%   -----------------------------------------------------------------------
%   | Tiebreaker Mode | Description                                       |
%   -----------------------------------------------------------------------
%   | Discard         | Samples with a tie are discarded                  |
%   | Retry           | Repeat with k-1 neighbors until the tie is broken |
%   | Closest         | Choose the class with the minimum total distance  |
%   | Random          | Choose the class randomly                         |
%   -----------------------------------------------------------------------

num_classes = size(testing_data, 3);
num_training_samples_per_class = size(training_data, 2);
num_testing_samples_per_class = size(testing_data, 2);

num_samples = num_classes * num_training_samples_per_class;
if k > num_samples
    msg = ['The value of ''k'' you have selected is larger than the' ...
              'number of potential samples: %d.'];
    error(msg, num_samples);
end

predictions = zeros(num_testing_samples_per_class, num_classes);

% Compute the distance of each testing point from every sample point,
% storing the distance away and sample point class as a tuple in the
% "neighbors" matrix

for i = 1:num_classes
    for m = 1:num_testing_samples_per_class
        fprintf('Classifying class: %d, sample: %d\n', i, m);
        idx = 1;
        neighbors = zeros(num_classes * num_training_samples_per_class, 2);
        for j = 1:num_classes
            for n = 1:num_training_samples_per_class
                v = testing_data(:, m, i) - training_data(:, n, j);
                d = sqrt(v' * v);
                neighbors(idx, :) = [d j];
                idx = idx + 1;
            end
        end
        
        % Sort the tuples by the distance entry
        neighbors = sortrows(neighbors, 1);
        
        % Determine the class count for the k closest samples
        count = zeros(num_classes, 2);
        for l = 1:k
            distance = neighbors(l, 1);
            class = neighbors(l, 2);
            count(class, 1) = count(class, 1) + 1;
            count(class, 2) = count(class, 2) + distance;
        end
        
        % Assign the label of the largest class count
        majority = 0;
        for l = 1:num_classes
            if count(l, 1) > majority
                tie = false;
                majority = count(l, 1);
                predictions(m, i) = l;
            elseif count(l, 1) == majority
                tie = true;
            end
        end
        
        % In the event of a tie, use tiebreaker parameter
        if tie
            switch tiebreaker  
                % If discard is chosen, assign label of -1
                case 'discard'
                    predictions(m, i) = -1;
                    
                % If retry is chosen, repeat algorithm with k-1 neighbors 
                case 'retry'
                    while tie
                        k = k - 1;

                        count = zeros(num_classes, 1);
                        for l = 1:k
                            class = neighbors(l, 2);
                            count(class) = count(class) + 1;
                        end

                        majority = 0;
                        for l = 1:num_classes
                            if count(l) > majority
                                tie = false;
                                majority = count(l);
                                predictions(m, i) = l;
                            elseif count(l) == majority
                                tie = true;
                            end
                        end
                    end
                    
                % If distance is chosen, select the tied label with the
                % minimum total distance
                case 'closest'
                    majority = max(count(:, 1));
                    indices = find(count(:, 1) == majority);
                    closest = min(count(indices, 2));
                    predictions(m, i) = indices(count(indices, 2) == closest);
                    
                % If random is chosen, select one the tied labels randomly 
                case 'random'
                    majority = max(count(:, 1));
                    indices = find(count(:, 1) == majority);
                    predictions(m, i) = datasample(indices, 1);
                 
                % Throw an error if an invalid tiebreaker is selected    
                otherwise
                    msg = 'You have selected an invalid tiebreaker';
                    error(msg);
            end
        end  
    end
end

    