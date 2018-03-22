%% Project 1
% Omar Abdelkader

% # of classes: 68
% # of features: 1920

%% Load Variables
% Images loaded in .mat format

load('Data/illumination.mat')

%% Preprocess Data
% No need, images have alread been reshaped to vector form. Split data into
% training (2/3) and testing (1/3)

training_data = illum(:,1:14,:);
testing_data = illum(:,15:21,:);

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