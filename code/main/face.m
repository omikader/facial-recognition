%% Project 1
% Omar Abdelkader

% # of classes: 200
% # of features/dimensions: 504

%% Load Variables
% Images loaded in .mat format

load('Data/face.mat')

%% Preprocess Data
% Separate images by class and reshape images into vector form. Split data
% into training (~2/3) and testing (~1/3)

data = zeros(size(face, 1) * size(face, 2), 3, 200);

for n = 1:200
    data(:, 1, n) = reshape(face(:, :, 3*n-2), [504, 1]);
    data(:, 2, n) = reshape(face(:, :, 3*n-1), [504, 1]);
    data(:, 3, n) = reshape(face(:, :, 3*n), [504, 1]);
end

training_data = data(:, 1:2, :);
testing_data = data(:, 3, :);

%% Bayesian Classification
% Use maximum likelihood estimation with Gaussian assumption to estimate
% parameters mu and sigma. Then use Bayes' classifier to classify the
% photos in the face dataset.

[mu, sigma] = mle(training_data);

bayesian_predictions = bayes(mu, sigma, testing_data);
bayesian_accuracy = get_accuracy(predictions, testing_data);

%% K-Nearest Neighbors Classification
% Use K-Nearest Neighbors to classify the photos in the face dataset

k = 1;
k_nn_predictions = k_nn(k, training_data, testing_data);
k_nn_accuracy = get_accuracy(predictions, testing_data);

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