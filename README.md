# facial_recognition
Implement basic classifiers (Bayes', K-Nearest Neighbors, PCA, LDA) in MATLAB to achieve facial recognition.
 
## Introduction
 
For a formal definition of the assignment, please see the project [description](docs/proj01.pdf). For a summary of the results, please see my [final report](docs/final_report.pdf).

## How to Run the Code

In order to keep the code readable and modular, each of the classifier and dimension reduction techniques have been implemented in separate MATLAB functions located at the top level of the [code](code/) directory. Helper functions are conveniently located in the [utils](code/utils/) directory. Each dataset is processed in separate MATLAB scripts located in the [main](code/main/) directory.

For each dataset, I divide the script into the following sections:

* Load Variables
* Preprocess Data
* Divide Data (Training and Testing)
* Bayesian Classification
* K-Nearest Neighbors Classification
* Principal Component Analysis (PCA)
  * Bayesian Classification after PCA
  * K-Nearest Neighbors Classification after PCA
* Fisher's Linear Discriminant Analysis (LDA)
  * Bayesian Classification after LDA
  * K-Nearest Neighbors Classification after LDA
  
Each script can be run its entirety, or run one section at a time to observe and analyze the results for the given section. All input data can be found in the [data](data/) directory in the form of `.mat` files.
