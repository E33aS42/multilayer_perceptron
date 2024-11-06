# multilayer_perceptron
Implementation from scratch of a customizable multilayer perceptron

<img src="https://github.com/user-attachments/assets/fef334da-4ab0-4ff3-8cac-82021430f463" width="800">
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
- removal of non relevant data (patient ID in the current study)
- numerization of non-numerical data
- (optional) removal of highly correlated data features.
There were no missing feature values for the current study.

<img src="https://github.com/user-attachments/assets/9a4062c5-953e-48ce-a40a-00279926d164" width="49.5%"> <img src="https://github.com/user-attachments/assets/48418179-3325-40a4-9ce7-10e2f65ea554" width="49.5%">

Data split:
- Prepared data was then split into 3 datasets: training, validation and test.

Usage:
	./split.py \<data.csv\> [column_number...]

optional

## Training (train.py)
<br />
Datasets were normalized using min-max (or standard) normalization based on the training data. The model was then trained on the training set to evaluate its weights. These were consecutively validated using the validation dataset to avoid overfitting.

<br /><br />
![metrics](https://github.com/user-attachments/assets/e15522cd-c27c-4db7-baee-b8d4ff5372c8)
<br /><br />



## Prediction (prediction.py)
<br />
Test data was normalized using the same normalization function coefficients previously applied on training and validation datasets.
