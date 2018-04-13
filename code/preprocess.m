function [ training_data, testing_data ] = preprocess(dataset, training_ratio)
%PREPROCESS Reshapes image data into vector form and divide into training
%and testing data according to the training ratio.
%   [training_data, testing_data] = PREPROCESS(dataset, training_ratio)
%   will return the training and testing data split for the given dataset
%   according to the training ratio provided.

switch dataset
    case 'face'
        input = load('data/face.mat');
        data = to_vector(dataset, input.face);
    case 'pose'
        input = load('data/pose.mat');
        data = to_vector(dataset, input.pose);
    case 'illum'
        input = load('data/illumination.mat');
        data = input.illum;
end

num_samples_per_class = size(data, 2);
elem = round(num_samples_per_class * training_ratio);

training_data = data(:, 1:elem, :);
testing_data = data(:, (elem+1):end, :);

end

