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
num_classes = size(data, 3);

for n = 1:200
    data(:, 1, n) = reshape(face(:, :, 3*n-2), [504, 1]);
    data(:, 2, n) = reshape(face(:, :, 3*n-1), [504, 1]);
    data(:, 3, n) = reshape(face(:, :, 3*n), [504, 1]);
end

%% Divide Data
% Split data into training (~2/3) and testing (~1/3).

training_data = data(:, 1:2, :);
num_samples_per_training_class = size(training_data, 2);

testing_data = data(:, 3, :);
num_samples_per_testing_class = size(testing_data, 2);

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
k_nn_predictions = k_nn(k, training_data, testing_data, 'closest');
k_nn_accuracy = get_accuracy(k_nn_predictions, testing_data);

%% Principal Component Analysis (PCA)
% Use principal component analysis to reduce the images in the training set
% down to a lower dimension feature set. Parameter alpha to choose how much
% energy willing to sacrifice.

alpha = 0.05;
W_pca = pca(training_data, alpha);
num_principal_components = size(W_pca, 1);

% Project the original dataset onto the principal components

pca_training_proj = zeros(num_principal_components, num_samples_per_training_class, num_classes);
pca_testing_proj = zeros(num_principal_components, num_samples_per_testing_class, num_classes);

for i = 1:num_classes
    for n = 1:num_samples_per_training_class
        pca_training_proj(:, n, i) = W_pca * training_data(:, n, i);
    end
end

for i = 1:num_classes
    for n = 1:num_samples_per_testing_class
        pca_testing_proj(:, n, i) = W_pca * testing_data(:, n, i);
    end
end

% Post PCA Bayesian Classification

pca_params = mle(pca_training_proj, 'normal');
pca_bayesian_predictions = bayes(pca_params, pca_testing_proj, 'normal');
pca_bayesian_accuracy = get_accuracy(pca_bayesian_predictions, pca_testing_proj);

% Post PCA K-NN Classification

k = 1;
pca_k_nn_predictions = k_nn(k, pca_training_proj, pca_testing_proj, 'closest');
pca_k_nn_accuracy = get_accuracy(pca_k_nn_predictions, pca_testing_proj);

%% Fisher's Multiple Discriminant Analysis (MDA)
% Use Fisher's linear discriminant analysis technique (generalized for 'c'
% classes) for dimensionality reduction.

W_mda = mda(training_data);
num_principal_components = size(W_mda, 1);

% Project the original dataset onto the principal components

mda_training_proj = zeros(num_principal_components, num_samples_per_training_class, num_classes);
mda_testing_proj = zeros(num_principal_components, num_samples_per_testing_class, num_classes);

for i = 1:num_classes
    for n = 1:num_samples_per_training_class
        mda_training_proj(:, n, i) = W_mda * training_data(:, n, i);
    end
end

for i = 1:num_classes
    for n = 1:num_samples_per_testing_class
        mda_testing_proj(:, n, i) = W_mda * testing_data(:, n, i);
    end
end

% Post MDA Bayesian Classification

mda_params = mle(mda_training_proj, 'normal');
mda_bayesian_predictions = bayes(mda_params, mda_testing_proj, 'normal');
mda_bayesian_accuracy = get_accuracy(mda_bayesian_predictions, mda_testing_proj);

% Post MDA K-NN Classification

k = 1;
mda_k_nn_predictions = k_nn(k, mda_training_proj, mda_testing_proj, 'closest');
mda_k_nn_accuracy = get_accuracy(mda_k_nn_predictions, mda_testing_proj);