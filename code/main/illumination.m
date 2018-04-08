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

n = 1;
k_nn_predictions = k_nn(n, training_data, testing_data, 'discard');
k_nn_accuracy = get_accuracy(k_nn_predictions, testing_data);

%% Principal Component Analysis (PCA)
% Use principal component analysis to reduce the images in the training set
% down to a lower dimension feature set. Parameter alpha to choose how much
% energy willing to sacrifice.

alpha = 0.05;
W_pca = pca(training_data, alpha);

num_classes = size(illum, 3);
num_samples_per_testing_class = size(testing_data, 2);
num_samples_per_training_class = size(training_data, 2);
num_principal_components = size(W_pca, 1);

% Project the original dataset onto the principal components

training_proj = zeros(num_principal_components, num_samples_per_training_class, num_classes);
testing_proj = zeros(num_principal_components, num_samples_per_testing_class, num_classes);

for i = 1:num_classes
    for n = 1:num_samples_per_training_class
        training_proj(:, n, i) = W_pca * training_data(:, n, i);
    end
end

for i = 1:num_classes
    for n = 1:num_samples_per_testing_class
        testing_proj(:, n, i) = W_pca * testing_data(:, n, i);
    end
end

% Post PCA Bayesian Classification

pca_params = mle(training_proj, 'normal');
pca_bayesian_predictions = bayes(pca_params, testing_proj, 'normal');
pca_bayesian_accuracy = get_accuracy(pca_bayesian_predictions, testing_proj);

% Post PCA K-NN Classification

k = 1;
pca_k_nn_predictions = k_nn(k, training_proj, testing_proj, 'discard');
pca_k_nn_accuracy = get_accuracy(pca_k_nn_predictions, testing_proj);

%% Fisher's Multiple Discriminant Analysis (MDA)
% Use Fisher's linear discriminant analysis technique (generalized for 'c'
% classes) for dimensionality reduction.

W_mda = mda(training_data);