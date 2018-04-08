%% Project 1
% Omar Abdelkader

% # of classes: 200
% # of features/dimensions: 504

%% Load Variables
% Images loaded in .mat format

load('data/face.mat')

%% Preprocess Data
% Separate images by class and reshape images into vector form.

data = zeros(size(face, 1) * size(face, 2), 3, 200);

for n = 1:200
    data(:, 1, n) = reshape(face(:, :, 3*n-2), [504, 1]);
    data(:, 2, n) = reshape(face(:, :, 3*n-1), [504, 1]);
    data(:, 3, n) = reshape(face(:, :, 3*n), [504, 1]);
end

%% Divide Data
% Split data into training (~2/3) and testing (~1/3).

training_data = data(:, 1:2, :);
testing_data = data(:, 3, :);

%% Bayesian Classification
% Use maximum likelihood estimation with Gaussian assumption to estimate
% parameters mu and sigma. Then use Bayes' classifier to classify the
% photos in the face dataset.

params = mle(training_data, 'normal');
bayesian_predictions = bayes(params, testing_data, 'normal');
bayesian_accuracy = get_accuracy(bayesian_predictions, testing_data);

%% K-Nearest Neighbors Classification
% Use K-Nearest Neighbors to classify the photos in the face dataset

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