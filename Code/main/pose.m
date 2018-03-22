%% Project 1
% Omar Abdelkader

% # of classes: 68
% # of features/dimensions: 1920

%% Load Variables
% Images loaded in .mat format

load('Data/pose.mat')

%% Preprocess Data
% Reshape images into vector form. Split data into training (2/3) and
% testing (1/3)

data = zeros(size(pose, 1) * size(pose, 2), 13, 68);

for n = 1:size(pose, 3)
    for i = 1:size(pose, 4)
        data(:, n, i) = reshape(pose(:, :, n, i), [1920, 1]);
    end
end
        
training_data = data(:,1:9,:);
testing_data = data(:,10:13,:);

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