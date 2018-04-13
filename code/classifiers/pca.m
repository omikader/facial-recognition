function [ W ] = pca(training_data, alpha)
%PCA Returns the principal components in the form of a matrix according to
%the second parameter, alpha, which dictates the energy willing to
%sacrifice.
%   W = PCA(training_data, alpha) will return the principal components of
%   the training data.

num_features = size(training_data, 1);
num_classes = size(training_data, 3);
num_samples_per_class = size(training_data, 2);

% Define total mean vector of the data

m = mean(mean(training_data, 3), 2);

% Center the datapoints

centered = zeros(size(training_data));

for i = 1:num_classes
    for n = 1:num_samples_per_class
        centered(:, n, i) = training_data(:, n, i) - m;
    end
end

% Compute the scatter matrix

C = zeros(num_features, num_features);

for i = 1:num_classes
    for n = 1:num_samples_per_class
        C = C + (centered(:, n, i) * centered(:, n, i)');
    end
end

C = C / (num_classes * num_samples_per_class);

% Identify the largest eigenvalues and their corresponding eigenvectors

[V, D] = eig(C);
[d, ind] = sort(diag(D), 'descend');
Vs = V(:, ind);

% Select as many eigenvectors as possible, while maintaining the energy
% restriction imposed by alpha

num = 0;
den = sum(d);
dimensions = 0;

for i = 1:size(d)
    dimensions = dimensions + 1;
    num = num + d(i);
    if num / den >= 1 - alpha
        break
    end
end

W = Vs(:, 1:dimensions);
