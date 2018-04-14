# facial_recognition
Implement basic classifiers (Bayes', K-Nearest Neighbors, PCA, LDA) in MATLAB to achieve facial recognition.
 
## Introduction
 
For a formal definition of the assignment, please see the project [description](docs/proj01.pdf). For a summary of the results, please see my [final report](docs/final_report.pdf).

## How to Run the Code

In order to keep the code readable and modular, each of the classifier and dimension reduction techniques have been implemented in separate MATLAB functions located in the [classifiers](code/classifiers) directory. Helper functions are conveniently located in the [utils](code/utils/) directory. The [main](code/main.m) script is located in the top level of the code directory.

I have divided the script into the following sections:

* Preprocess Data
  * Load Variables
  * Divide Data (Training and Testing)
* Bayesian Classification
* K-Nearest Neighbors Classification
* Principal Component Analysis (PCA)
  * Bayesian Classification after PCA
  * K-Nearest Neighbors Classification after PCA
* Fisher's Linear Discriminant Analysis (LDA)
  * Bayesian Classification after LDA
  * K-Nearest Neighbors Classification after LDA
  
The main file contains initial conditions and parameters for different circumstances. To test different functionality, simply modify those state conditions variables.
  
The script can be run its entirety, or one section at a time to observe and analyze the results for the given section. At the end of the script, the results of each classification technique are displayed in a table. All input data can be found in the [data](data/) directory in the form of `.mat` files. For any help or clarification, please review the help text to understand how each of the functions are meant to be used.

## Extra Credit

The extra credit [script](code/extra_credit.m) is written in the same format as the main script. In this script, I split the [face](data/face.mat) dataset 3 different classes corresponding to the states of "Happy", "Neutral" and "Illuminated". I used this data to train the classifiers to detect the state of a given subject, as opposed to recognizing the subject themselves.
