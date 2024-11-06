# multilayer_perceptron
Implementation from scratch of a customizable multilayer perceptron

<img src="https://github.com/user-attachments/assets/1d51224a-bfba-4c43-944e-ffe02a65607f" width="1000">
<br /><br />

This is a work based on the first 2 courses of the deep learning specialization by Andrew Ng.<br>
https://www.coursera.org/specializations/deep-learning#courses
<br /><br />

The case study presented here is based on a dataset of breast cancer diagnoses (malignant or benign). This is primarily a binary classification problem, though a multiclass implementation has also been done for generalization.
https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names
<br /><br />

This project has been done in 3 parts:<br>
1. Raw Data analysis and preparation.<br>
2. Model training.<br>
3. Prediction.<br>

## Data analysis and preparation (split.py)
<br />

Data cleaning:
- removal of non relevant data (patient ID in the current study)
- numerization of non-numerical data
- (optional)



Prepared data was then split into 3 datasets: training, validation and test.

## Training (train.py)
<br />
After normalization of the datasets based on min-max or standard normalization of the training dataset, model training was done on the training dataset to evaluate model weights which are consecutively validated using the validation dataset to avoid overfitting.

<br /><br />
![metrics](https://github.com/user-attachments/assets/e15522cd-c27c-4db7-baee-b8d4ff5372c8)
<br /><br />


## Prediction (prediction.py)
<br />
Test data was normalized using the same normalization function coefficients previously applied on training and validation datasets.