function [ projected ] = mda( training_data )
%MDA Reduces the data parameter down to a lower dimension set of size c-1,
%where c is the number of classes.
%   projected = MDA(data) will return a dataset with a lower first
%   dimension of num_classes - 1.

num_features = size(training_data, 1);
num_classes = size(training_data, 3);
num_samples_per_class = size(training_data, 2);

Sw = zeros(num_features, num_features);
m = zeros(num_features, 1);

% Compute the within class scatter matrix and total mean vector

for i = 1:num_classes
    sum = zeros(num_features, 1);
    for n = 1:num_samples_per_class
        sum = sum + training_data(:, n, i);
        m = m + training_data(:, n, i);
    end
    m_i = sum / n;
    
    Sw_i = zeros(num_features, num_features);
    for n = 1:num_samples_per_class
        Sw_i = Sw_i + ((training_data(:, n, i) - m_i) * (training_data(:, n, i) - m_i)');
    end
    Sw = Sw + Sw_i;
end

m = m / (num_classes * num_samples_per_class);

% Using the total mean vector, compute the total scatter matrix

St = zeros(num_features, num_features);
for i = 1:num_classes
    for n = 1:num_samples_per_class
        St = St + ((training_data(:, n, i) - m) * (training_data(:, n, i) - m)');
    end
end

% The between scatter matrix is the difference of the total scatter and
% within scatter matrices

Sb = St - Sw;

% Solve the generalized eigenvalue problem and select the eigenvalues of
% the "num_classes - 1" largest eigenvalues.

[V, D] = eig(Sb, Sw);
[~, ind] = sort(diag(D), 'descend');
Vs = V(:, ind);

W = Vs(1:(num_classes - 1), :);
projected = zeros(num_classes - 1, num_samples_per_class, num_classes);

% Project the original data onto the selected "num_classes - 1"
% eigenvectors

for i = 1:num_classes
    for n = 1:num_samples_per_class
        projected(:, n, i) = W * training_data(:, n, i);
    end
end

end