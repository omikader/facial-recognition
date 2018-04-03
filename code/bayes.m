function [ predictions ] = bayes(mu, sigma, testing_data)
%BAYES Computes the class conditional probabilities using the mu and sigma
%parameters. Since the priors and evidence probabilities are equal, we can
%safely compare the values of the class conditional to classify.
%   predictions = BAYES(mu, sigma, testing_data) will return a matrix with
%   the Bayes' classifier prediction for each test data point.

num_features = size(testing_data, 1);
num_classes = size(testing_data, 3);
num_samples_per_class = size(testing_data, 2);

predictions = zeros(num_samples_per_class, num_classes);

% Compute the class conditional probability for each testing point and
% assign the label of the largest probability

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
        fprintf('Computing Bayes'' probability for class %d, sample, %d\n', i, n);
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