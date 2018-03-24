function [ predictions ] = k_nn( k, training_data, testing_data )
%K_NN Classifies each test sample based on the label of the majority of its
%k nearest training sample neighbors.
%   predictions = K_NN(k, training_data, testing_data) will return a matrix
%   with the prediction for each test data point.

num_classes = size(testing_data, 3);
num_training_samples_per_class = size(training_data, 2);
num_testing_samples_per_class = size(testing_data, 2);

if mod(k, 2) == 0
    msg = 'Please select an odd value for ''k''.';
    error(msg);
end

num_samples = num_classes * num_training_samples_per_class;
if k > num_samples
    msg = 'The value of ''k'' you have selected is larger than the number of potential samples: %d.';
    error(msg, num_samples);
end

predictions = zeros(num_testing_samples_per_class, num_classes);

% Compute the distance of each testing point from every sample point,
% storing the distance away and sample point class as a tuple in the
% "neighbors" matrix

for i = 1:num_classes
    for m = 1:num_testing_samples_per_class
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
        
        count = zeros(num_classes, 1);
        for l = 1:k
            class = neighbors(l, 2);
            count(class) = count(class) + 1;
        end
        
        % Assign the label of the largest class count
        
        max = 0;
        for l = 1:num_classes
            if count(l) > max
                max = count(l);
                predictions(m, i) = l;
            end
        end
        
    end
end

    