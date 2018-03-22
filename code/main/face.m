%% Project 1
% Omar Abdelkader

% # of classes: 200
% # of features/dimensions: 504

%% Load Variables
% Images loaded in .mat format

load('Data/face.mat')

%% Preprocess Data
% Separate images by class and reshape images into vector form. Split data
% into training (2/3) and testing (1/3)

data = zeros(size(face, 1) * size(face, 2), 3, 200);

for n = 1:200
    data(:, 1, n) = reshape(face(:, :, 3*n-2), [504, 1]);
    data(:, 2, n) = reshape(face(:, :, 3*n-1), [504, 1]);
    data(:, 3, n) = reshape(face(:, :, 3*n), [504, 1]);
end

training_data = data(:, 1:2, :);
testing_data = data(:, 3, :);

%% Maximum Likelihood
% Use maximum likelihood estimation with Gaussian assumption to estimate
% parameters mu and sigma

[mu, sigma] = mle(training_data);

%% Bayesian Classification
% Use Bayes' classifier to classify the photos

predictions = bayes(mu, sigma, testing_data);

%% K-Nearest Neighbors Classification

%% Principal Component Analysis (PCA)

%% Fisher's Linear Discriminant Analysis (LDA)