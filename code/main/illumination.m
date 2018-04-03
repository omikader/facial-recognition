%% Project 1
% Omar Abdelkader

% # of classes: 68
% # of features: 1920

%% Load Variables
% Images loaded in .mat format

load('data/illumination.mat')

%% Divide Data
% Split data into training (~3/4) and testing (~1/4).

training_data = illum(:, 1:16, :);
testing_data = illum(:, 17:21, :);

%% Bayesian Classification
% Use maximum likelihood estimation with Gaussian assumption to estimate
% parameters mu and sigma. Then use Bayes' classifier to classify the
% photos in the illumination dataset.

params = mle(training_data, 'gaussian');
bayesian_predictions = bayes(params, testing_data, 'gaussian');
bayesian_accuracy = get_accuracy(bayesian_predictions, testing_data);

%% K-Nearest Neighbors Classification
% Use K-Nearest Neighbors to classify the photos in the illumination
% dataset

k = 1;
k_nn_predictions = k_nn(k, training_data, testing_data);
k_nn_accuracy = get_accuracy(k_nn_predictions, testing_data);

%% Principal Component Analysis (PCA)
% Use principal component analysis to reduce the images in the training set
% down to a lower dimension feature set. Parameter alpha to choose how much
% energy willing to sacrifice.

alpha = 0.05;
W_pca = pca(training_data, alpha);

%% Fisher's Multiple Discriminant Analysis (MDA)
% Use Fisher's linear discriminant analysis technique (generalized for 'c'
% classes) for dimensionality reduction.

W_mda = mda(training_data);