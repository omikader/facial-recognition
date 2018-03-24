%% Project 1
% Omar Abdelkader

% # of classes: 68
% # of features/dimensions: 1920

%% Load Variables
% Images loaded in .mat format

load('Data/pose.mat')

%% Preprocess Data
% Reshape images into vector form. Split data into training (~2/3) and
% testing (~1/3)

data = zeros(size(pose, 1) * size(pose, 2), 13, 68);

for n = 1:size(pose, 3)
    for i = 1:size(pose, 4)
        data(:, n, i) = reshape(pose(:, :, n, i), [1920, 1]);
    end
end
        
training_data = data(:,1:9,:);
testing_data = data(:,10:13,:);

%% Bayesian Classification
% Use maximum likelihood estimation with Gaussian assumption to estimate
% parameters mu and sigma. Then use Bayes' classifier to classify the
% photos in the pose dataset.

[mu, sigma] = mle(training_data);

bayesian_predictions = bayes(mu, sigma, testing_data);
bayesian_accuracy = get_accuracy(bayesian_predictions, testing_data);

%% K-Nearest Neighbors Classification
% Use K-Nearest Neighbors to classify the photos in the pose dataset

k = 1;
k_nn_predictions = k_nn(k, training_data, testing_data);
k_nn_accuracy = get_accuracy(k_nn_predictions, testing_data);

%% Principal Component Analysis (PCA)
% Use principal component analysis to reduce the photos down to a lower
% dimension feature set. Parameter alpha to choose how much energy willing
% to sacrifice.

alpha = 0.05;
pca_projected = pca(data, alpha);

%% Fisher's Multiple Discriminant Analysis (MDA)
% Use Fisher's linear discriminant analysis technique (generalized for 'c'
% classes) for dimensionality reduction.

mda_projected = mda(data);