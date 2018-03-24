function [ projected ] = pca( data, alpha )
%PCA Reduces the data parameter down to a lower dimension according to the
%second parameter alpha, the amount of energy willing to sacrifice.
%   projected = PCA(data, alpha) will return a dataset with a lower first
%   dimension.

num_features = size(data, 1);
num_classes = size(data, 3);
num_samples_per_class = size(data, 2);

% Center the datapoints

centered = zeros(size(data));
mean = zeros(size(data, 1), 1);

for i = 1:num_classes
    for n = 1:num_samples_per_class
        mean = mean + data(:, n, i);
    end
end

mean = mean / (num_classes * num_samples_per_class);

for i = 1:num_classes
    for n = 1:num_samples_per_class
        centered(:, n, i) = data(:, n, i) - mean;
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

% Project the dataset onto the eigenvectors corresponding to the largest
% eigenvalues

W = Vs(1:dimensions, :);
projected = zeros(dimensions, num_samples_per_class, num_classes);

for i = 1:num_classes
    for n = 1:num_samples_per_class
        projected(:, n, i) = W * data(:, n, i);
    end
end