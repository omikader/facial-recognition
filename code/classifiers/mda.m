function [ W ] = mda(training_data, num_dimensions)
%MDA Returns a transformation matrix of size num_dimensions, with max size
%c-1, where c is the total number of classes, using Fisher's multiple 
%discriminant analysis.
%   W = MDA(training_data, num_dimensions) will return an MDA 
%   transformation matrix of size of num_dimensions.

num_features = size(training_data, 1);
num_classes = size(training_data, 3);
num_samples_per_class = size(training_data, 2);

if num_dimensions >= num_classes
    msg = ['The transformation matrix size you have selected is too' ...
              ' large, please retry with size c-1 = %d, or smaller.'];
    error(msg, num_classes - 1);
end

% Define total mean vector of the data

m = mean(mean(training_data, 3), 2);

% Compute between and within scatter matrices

S_w = zeros(num_features, num_features);
S_b = zeros(num_features, num_features);

for i = 1:num_classes
    m_i = mean(training_data(:, :, i), 2);
    S_b = S_b + (num_samples_per_class * ((m_i - m) * (m_i - m)'));
    S_i = zeros(num_features, num_features);
    for n = 1:num_samples_per_class
        S_i = S_i + (training_data(:, n, i) - m_i) * (training_data(:, n, i) - m_i)';
    end
    S_w = S_w + S_i;
end

% Add a regularization term to make S_w positive-definite

S_w = S_w + eye(num_features);

% Solve the generalized eigenvalue problem and select the eigenvalues of
% the "num_dimensions" largest eigenvalues

[V, D] = eig(S_b, S_w);
[~, ind] = sort(diag(D), 'descend');
Vs = V(:, ind);

W = Vs(:, 1:(num_dimensions));

end