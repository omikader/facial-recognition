function [ training_proj, testing_proj ] = project(W, training_data, testing_data)
%PROJECT Projects the training and testing data onto vectors in W.
%   [training_proj, testing_proj] = PROJECT(W, training_data, testing_data)
%   will return the projected training and testing data.

num_principal_components = size(W, 1);
num_classes = size(training_data, 3);
num_samples_per_training_class = size(training_data, 2);
num_samples_per_testing_class = size(testing_data, 2);

training_proj = zeros(num_principal_components, num_samples_per_training_class, num_classes);
testing_proj = zeros(num_principal_components, num_samples_per_testing_class, num_classes);

for i = 1:num_classes
    for n = 1:num_samples_per_training_class
        training_proj(:, n, i) = W * training_data(:, n, i);
    end
end

for i = 1:num_classes
    for n = 1:num_samples_per_testing_class
        testing_proj(:, n, i) = W * testing_data(:, n, i);
    end
end

end

