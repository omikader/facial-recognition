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

params = mle(training_data, 'normal');
bayesian_predictions = bayes(params, testing_data, 'normal');
bayesian_accuracy = get_accuracy(bayesian_predictions, testing_data);

%% K-Nearest Neighbors Classification
% Use K-Nearest Neighbors to classify the photos in the illumination
% dataset

k = 1;
k_nn_predictions = k_nn(k, training_data, testing_data, 'discard');
k_nn_accuracy = get_accuracy(k_nn_predictions, testing_data);

%% Principal Component Analysis (PCA)
% Use principal component analysis to reduce the images in the training set
% down to a lower dimension feature set. Parameter alpha to choose how much
% energy willing to sacrifice.

alpha = 0.05;
W_pca = pca(training_data, alpha);

% Project the original dataset onto the principal components

training_proj = zeros(size(W_pca, 1), size(training_data, 2), size(training_data, 3));
testing_proj = zeros(size(W_pca, 1), size(testing_data, 2), size(testing_data, 3));

for j = 1:size(training_data, 3)
    for k = 1:size(training_data, 2)
        training_proj(:, k, j) = W_pca * training_data(:, k, j);
    end
end

for j = 1:size(testing_data, 3)
    for k = 1:size(testing_data, 2)
        testing_proj(:, k, j) = W_pca * testing_data(:, k, j);
    end
end

% Post PCA Bayesian Classification

params = mle(training_proj, 'normal');
bayesian_predictions = bayes(params, testing_proj, 'normal');
bayesian_accuracy = get_accuracy(bayesian_predictions, testing_proj);

% Post PCA K-NN Classification

k = 1;
k_nn_predictions = k_nn(k, training_proj, testing_proj, 'discard');
k_nn_accuracy = get_accuracy(k_nn_predictions, testing_proj);

%% Fisher's Multiple Discriminant Analysis (MDA)
% Use Fisher's linear discriminant analysis technique (generalized for 'c'
% classes) for dimensionality reduction.

W_mda = mda(training_data);