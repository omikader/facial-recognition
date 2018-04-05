function [ predictions ] = bayes(params, testing_data, distribution)
%BAYES Classifies the testing data according to the provided distribution 
%using the parameters derived from the maximum likelihood estimation of the
%training data.
%   predictions = BAYES(params, testing_data, distribution) will return a 
%   matrix with the Bayes' classifier prediction for each test data point.

num_features = size(testing_data, 1);
num_classes = size(testing_data, 3);
num_samples_per_class = size(testing_data, 2);

predictions = zeros(num_samples_per_class, num_classes);

switch distribution
    case 'normal'
        mu = params{1};
        sigma = params{2};
        
        % Compute the general case Gaussian distribution discriminant 
        % function for each testing point and assign the label of the 
        % largest value

        W = zeros(num_features, num_features, num_classes);
        w = zeros(num_features, num_classes);
        w_0 = zeros(num_classes, 1);

        for i = 1:num_classes
            inverse = inv(sigma(:, :, i));
            determinant = det(sigma(:, :, i));

            W(:, :, i) = (-1/2) * inverse;
            w(:, i) = mu(:, i)' * inverse;
            w_0(i) = ((-1/2) * mu(:, i)' * inverse * mu(:, i)) - ((1/2) * log(determinant));
        end

        for i = 1:num_classes
            for n = 1:num_samples_per_class
                fprintf('Classifying class: %d, sample: %d\n', i, n);
                max = intmin;
                for j = 1:num_classes
                    g = (testing_data(:, n, i)' * W(:, :, j) * testing_data(:, n, i)) + (w(:, j)' * testing_data(:, n, i)) + w_0(j);
                    if g > max
                        max = g;
                        predictions(n, i) = j;
                    end
                end
            end
        end
    otherwise
        msg = 'You have selected an invalid or unsupported distribution';
        error(msg);
end