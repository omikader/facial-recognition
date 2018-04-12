function [ transformation_matrix ] = mda(training_data)
%MDA Returns a transformation matrix to c-1 classes, where c is the total 
%number of classes in the data, using Fisher's multiple discriminant 
%analysis.
%   W = MDA(training_data) will return a transformation matrix of size c-1,
%   where c is the number of classes.

num_features = size(training_data, 1);
num_classes = size(training_data, 3);
num_samples_per_class = size(training_data, 2);

% S_w = zeros(num_features, num_features);
% S_b = zeros(num_features, num_features);
% m = mean(mean(training_data, 3), 2);
% 
% for i = 1:num_classes
%     m_i = mean(training_data(:, :, i), 2);
%     S_b = S_b + (num_samples_per_class * ((m_i - m) * (m_i - m)'));
%     S_i = zeros(num_features, num_features);
%     for n = 1:num_samples_per_class
%         S_i = S_i + (training_data(:, n, i) - m_i) * (training_data(:, n, i) - m_i)';
%     end
%     S_w = S_w + S_i;
% end
% 
% [V, D] = eig(S_b, S_w);
% [~, ind] = sort(diag(D), 'descend');
% Vs = V(:, ind);
% 
% transformation_matrix = Vs(:, 1:(num_classes - 1));









% m_i = zeros(num_features, num_classes);
% S_i = zeros(num_features, num_features, num_classes);
% for i = 1:num_classes
%     for n = 1:num_samples_per_class
%         m_i(:, i) = m_i(:, i) + training_data(:, n, i);
%     end
%     m_i(:, i) = m_i(:, i) / num_samples_per_class;
%     
%     for n = 1:num_samples_per_class
%         S_i(:, :, i) = S_i(:, :, i) + (training_data(:, n, i) - m_i(:, i)) * (training_data(:, n, i) - m_i(:,i ))';
%     end
% end
% 
% S_w = sum(S_i, 3);
% m = sum(m_i, 2) / num_classes;
% 
% S_t = zeros(num_features, num_features);
% for i = 1:num_classes
%     for n = 1:num_samples_per_class
%         S_t = S_t + (training_data(:, n, i) - m) * (training_data(:, n, i) - m)';
%     end
% end
% 
% S_b = S_t - S_w;
% 
% [V, D] = eig(S_b, S_w);
% [~, ind] = sort(diag(D), 'descend');
% Vs = V(:, ind);
% 
% transformation_matrix = Vs(:, 1:(num_classes - 1));








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
    
    S_i = zeros(num_features, num_features);
    for n = 1:num_samples_per_class
        S_i = S_i + ((training_data(:, n, i) - m_i) * (training_data(:, n, i) - m_i)');
    end
    Sw = Sw + S_i;
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

transformation_matrix = Vs(:, 1:(num_classes - 1));

end