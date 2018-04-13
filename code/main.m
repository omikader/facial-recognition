%% Project 1
% Omar Abdelkader

%% Preprocess Data
% Load images from .mat files, reshape images into feature vector form, and
% divide into training and testing data.

dataset = 'face';
training_ratio = (2/3);

[training_data, testing_data] = preprocess(datset, training_ratio);

% Initialize Table Variables

MODE = {};
ACCURACY = {};
PARAMS = {};

%% Bayesian Classification
% Use maximum likelihood estimation with Gaussian assumption to estimate
% parameters mu and sigma. Then use Bayes' classifier to classify the
% images in the dataset.

distribution = 'normal';

params = mle(training_data, distribution);
predictions = bayes(params, testing_data, distribution);
accuracy = get_accuracy(predictions);

% Add elements to table variables

MODE = vertcat(MODE, {'Bayes'''});
PARAMS = vertcat(PARAMS, strcat('distribution=', distribution));
ACCURACY = vertcat(ACCURACY, accuracy);

%% K-Nearest Neighbors Classification
% Use K-Nearest Neighbors to classify the images in the dataset.

k = 1;
tiebreaker = 'closest';

predictions = k_nn(k, training_data, testing_data, tiebreaker);
accuracy = get_accuracy(predictions);

% Add elements to table variables

MODE = vertcat(MODE, {'K-NN'});
PARAMS = vertcat(PARAMS, strcat('k=', num2str(k), ', tiebreaker=', tiebreaker));
ACCURACY = vertcat(ACCURACY, accuracy);

%% Principal Component Analysis (PCA)
% Use principal component analysis to reduce the images in the training set
% down to a lower dimension feature set. Parameter alpha to choose how much
% energy willing to sacrifice.

alpha = 0.05;
W = pca(training_data, alpha)';

% Project the original dataset onto the principal components

training_proj = project(W, training_data);
testing_proj = project(W, testing_data);

% Post PCA Bayesian Classification

distribution = 'normal';

params = mle(training_proj, distribution);
predictions = bayes(params, testing_proj, distribution);
accuracy = get_accuracy(predictions);

% Add elements to table variables

MODE = vertcat(MODE, {'PCA Bayes'''});
PARAMS = vertcat(PARAMS, strcat('alpha=', num2str(alpha), ', distribution=', distribution));
ACCURACY = vertcat(ACCURACY, accuracy);

% Post PCA K-NN Classification

k = 1;
tiebreaker = 'closest';

predictions = k_nn(k, training_proj, testing_proj, tiebreaker);
accuracy = get_accuracy(predictions);

% Add elements to table variables

MODE = vertcat(MODE, {'PCA K-NN'});
PARAMS = vertcat(PARAMS, strcat('alpha=', num2str(alpha), ', k=', num2str(k), ', tiebreaker=', tiebreaker));
ACCURACY = vertcat(ACCURACY, accuracy);

%% Fisher's Multiple Discriminant Analysis (MDA)
% Use Fisher's linear discriminant analysis technique (generalized for 'c'
% classes) for dimensionality reduction.

num_dimensions = 200;
W = mda(training_data, num_dimensions)';

% Project the original dataset onto the eigenvectors in W

training_proj = project(W, training_data);
testing_proj = project(W, testing_data);

% Post MDA Bayesian Classification

distribution = 'normal';

params = mle(training_proj, distribution);
predictions = bayes(params, testing_proj, distribution);
accuracy = get_accuracy(predictions);

% Add elements to table variables

MODE = vertcat(MODE, {'MDA Bayes'''});
PARAMS = vertcat(PARAMS, strcat('num_dimensions=', num2str(num_dimensions), ', distribution=', distribution));
ACCURACY = vertcat(ACCURACY, accuracy);

% Post MDA K-NN Classification

k = 1;
tiebreaker = 'closest';

predictions = k_nn(k, training_proj, testing_proj, tiebreaker);
accuracy = get_accuracy(predictions);

% Add elements to table variables

MODE = vertcat(MODE, {'MDA K-NN'});
PARAMS = vertcat(PARAMS, strcat('num_dimensions=', num2str(num_dimensions), ', k=', num2str(k), ', tiebreaker=', tiebreaker));
ACCURACY = vertcat(ACCURACY, accuracy);

%% Create Table
% Create a table to display the results

T = table(MODE, PARAMS, ACCURACY)