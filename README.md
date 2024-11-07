# multilayer_perceptron
Implementation from scratch of a customizable multilayer perceptron

<img src="https://github.com/user-attachments/assets/fef334da-4ab0-4ff3-8cac-82021430f463" width="900">
<br /><br />

This is a work based on the first 2 courses of the deep learning specialization by Andrew Ng.<br>
https://www.coursera.org/specializations/deep-learning#courses
<br /><br />

The case study presented here is based on a dataset of breast cancer diagnoses (malignant or benign) derived from breast mass sampling, with 30 different features describing characteristics of the cell nuclei.<br>
https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names
<br>
This is primarily a binary classification problem, though a multiclass approach has also been implemented for generalization.<br>
<br /><br />

This project is divided into 3 parts:<br>
1. Raw data analysis.<br>
2. Model training.<br>
3. Prediction.<br>

## Data analysis and preparation (split.py)
<br />

Data cleaning:
- removal of non relevant data (patients ID in the current study)
- numerization of non-numerical data
- (optional) removal of highly correlated data features.

There were no missing feature values for the current study.

<img src="https://github.com/user-attachments/assets/9a4062c5-953e-48ce-a40a-00279926d164" width="49.5%"> <img src="https://github.com/user-attachments/assets/48418179-3325-40a4-9ce7-10e2f65ea554" width="49.5%">
*Figure 2: Features correlation heatmaps*

Data split:
- Prepared data was then split into 3 datasets: training, validation and test.

Usage: &emsp;	`./split.py <data.csv> [column_number_to_remove...]`

<br />

## Training (train.py)
<br />
Datasets were normalized using min-max (or standard) normalization based on the training data. The model was then trained on the training set to evaluate its weights. These were consecutively validated using the validation dataset to avoid overfitting.

<br /><br />
![metrics](https://github.com/user-attachments/assets/e15522cd-c27c-4db7-baee-b8d4ff5372c8)
*Figure 3: Training and validation losses and accuracy for one run.*
<br /><br />

Training phase is customizable to explore the effects of model parameters on training:
- hidden layer available activation functions: sigmoid, tanh, relu or leaky relu.
- output layer available activation functions: sigmoid or softmax.
- weights initializations: random or Xavier.
- optimization algorithms: momentum, RMSprop or Adam.
- early stopping with configurable patience.
- multiple runs can be done consecutively.


#### Multiple runs examples:

<img src="https://github.com/user-attachments/assets/5bdb7d4d-efcc-4c3e-9608-8a3e1aa18b21" width="75%"><br>
*Figure 4: Multiple runs validation losses for different learning rates and network topology.*

<img src="https://github.com/user-attachments/assets/275f35df-2b6e-43f7-b1c8-84b4fad8f959" width="75%"><br>
*Figure 5: Multiple runs validation losses for different hidden layers activation functions.*

<img src="https://github.com/user-attachments/assets/eb7a5b3c-aed3-46c1-9eba-bd2bb6dc3b1b" width="75%"><br>
*Figure 6: Multiple runs validation losses with different optimization algorithms and early stopping.*

To access function usage: &emsp;	`./train.py -h`

<br />

## Prediction (prediction.py)
<br />
Test data was normalized using the same normalization function coefficients previously applied on training and validation datasets.<br>
Results returns accuracy score and test loss for the current test dataset, as well as model parameters.
<br /><br />

![Screenshot from 2024-11-07 10-43-53](https://github.com/user-attachments/assets/d0f73c9f-4411-43c9-84ec-e9e50bc76841)


Usage: &emsp;	`./prediction.py <nn_model.pkl>`

