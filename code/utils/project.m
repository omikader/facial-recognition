function [ proj_data ] = project(W, input_data)
%PROJECT Projects the training and testing data onto vectors in W.
%   [training_proj, testing_proj] = PROJECT(W, training_data, testing_data)
%   will return the projected training and testing data.

num_principal_components = size(W, 1);
num_classes = size(input_data, 3);
num_samples_per_class = size(input_data, 2);

proj_data = zeros(num_principal_components, num_samples_per_class, num_classes);

for i = 1:num_classes
    for n = 1:num_samples_per_class
        proj_data(:, n, i) = W * input_data(:, n, i);
    end
end

end

