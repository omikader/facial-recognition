function [ predictions ] = bayes( mu, sigma, testing_data )
%BAYES Computes the class conditional probabilities using the mu and sigma
%parameters. Since the priors and evidence probabilities are equal, we can
%safely compare the values of the class conditional to classify.
%   predictions = bayes(mu, sigma, testing_data) will return a matrix with
%   the Bayes' classifier prediction for each test data point.

num_features = size(testing_data, 1);
num_classes = size(testing_data, 3);
num_samples_per_class = size(testing_data, 2);

predictions = zeros(num_samples_per_class, num_classes);

% Compute the class conditional probability for each testing point and
% assign the label of the largest probability

for i = 1:num_classes
    for n = 1:num_samples_per_class
        max = 0;
        for j = 1:num_classes
            prob = (1/(((2*pi)^(num_features/2))*(sqrt(det(sigma(:, :, j))))))*(exp((-1/2)*(testing_data(:, n, i) - mu(:, j))'*(inv(sigma(:, :, j)))*(testing_data(:, n, i) - mu(:, j))));
            if prob > max
                max = prob;
                predictions(n, i) = j;
            end
        end
    end
end