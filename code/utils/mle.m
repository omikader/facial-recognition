function [ params ] = mle(training_data, distribution)
%MLE Computes the sample mean and variance of data according to the
%provided distribution
%   [mu, sigma] = MLE(training_data, distribution) will use maximum 
%   likelihood esimtation to determine the optimal parameters that fit the
%   data to the provided distrubtion.

num_features = size(training_data, 1);
num_classes = size(training_data, 3);
num_samples_per_class = size(training_data, 2);

switch distribution
    case 'normal'
        mu = zeros(num_features, num_classes);
        sigma = zeros(num_features, num_features, num_classes);

        for i = 1:num_classes
            fprintf('Computing mu and sigma for class %d\n', i);
            sum = zeros(num_features, 1);
            for n = 1:num_samples_per_class
                sum = sum + training_data(:, n, i);
            end
            mu(:, i) = sum / n;
            sum = zeros(num_features, num_features);
            for n = 1:num_samples_per_class
                sum = sum + (training_data(:, n, i) - mu(:, i)) * ...
                    (training_data(:, n, i) - mu(:, i))';
            end

            % If n < d, add a regularization term to make sigma positive-definite
            if num_samples_per_class < num_features
                sigma(:, :, i) = (sum / n) + eye(num_features);
            else
                sigma(:, :, i) = sum / n;
            end
        end
        params = {mu, sigma};
    otherwise
        msg = 'You have selected an invalid or unsupported distribution';
        error(msg);
end