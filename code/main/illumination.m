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
bayesian_accuracy = get_accuracy(bayesian_predictions);

%% K-Nearest Neighbors Classification
% Use K-Nearest Neighbors to classify the photos in the illumination
% dataset

k = 1;
k_nn_predictions = k_nn(k, training_data, testing_data, 'closest');
k_nn_accuracy = get_accuracy(k_nn_predictions);

%% Principal Component Analysis (PCA)
% Use principal component analysis to reduce the images in the training set
% down to a lower dimension feature set. Parameter alpha to choose how much
% energy willing to sacrifice.

alpha = 0.05;
W_pca = pca(training_data, alpha)';

% Project the original dataset onto the principal components

pca_training_proj = project(W_pca, training_data);
pca_testing_proj = project(W_pca, testing_data);

% Post PCA Bayesian Classification

pca_params = mle(pca_training_proj, 'normal');
pca_bayesian_predictions = bayes(pca_params, pca_testing_proj, 'normal');
pca_bayesian_accuracy = get_accuracy(pca_bayesian_predictions);

% Post PCA K-NN Classification

k = 1;
pca_k_nn_predictions = k_nn(k, pca_training_proj, pca_testing_proj, 'closest');
pca_k_nn_accuracy = get_accuracy(pca_k_nn_predictions);

%% Fisher's Multiple Discriminant Analysis (MDA)
% Use Fisher's linear discriminant analysis technique (generalized for 'c'
% classes) for dimensionality reduction.

W_mda = mda(training_data)';

% Project the original dataset onto the eigenvectors in W

mda_training_proj = project(W_mda, training_data);
mda_testing_proj = project(W_mda, testing_data);

% Post MDA Bayesian Classification

mda_params = mle(mda_training_proj, 'normal');
mda_bayesian_predictions = bayes(mda_params, mda_testing_proj, 'normal');
mda_bayesian_accuracy = get_accuracy(mda_bayesian_predictions);

% Post MDA K-NN Classification

k = 1;
mda_k_nn_predictions = k_nn(k, mda_training_proj, mda_testing_proj, 'closest');
mda_k_nn_accuracy = get_accuracy(mda_k_nn_predictions);