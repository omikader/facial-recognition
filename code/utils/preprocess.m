function [ training_data, testing_data ] = preprocess(dataset, ratio)
%PREPROCESS Summary of this function goes here
%   Detailed explanation goes here

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
elem = round(num_samples_per_class * ratio(1));

training_data = data(:, 1:elem, :);
testing_data = data(:, (elem+1):end, :);

end

