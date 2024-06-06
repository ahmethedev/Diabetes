# Diabetes Prediction using Random Forest Classifier

This code aims to predict whether a patient has diabetes or not based on various features such as the number of pregnancies, glucose level, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age. The prediction is done using a Random Forest Classifier, which is a machine learning algorithm that combines multiple decision trees to improve accuracy and robustness.

## Requirements

The following Python libraries are required to run this code:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Data

The dataset used in this code is the `diabetes.csv` file, which should be located in the same directory as the code. The dataset contains information about various features related to diabetes and the target variable, which indicates whether the patient has diabetes or not.

## Code Structure

1. **Data Loading and Preprocessing**
   - The code loads the `diabetes.csv` dataset using pandas.
   - It checks for missing values and fills them with the median value of the respective column.
   - It scales the features using `StandardScaler` from scikit-learn.
   - It separates the features (X) and the target variable (y) from the scaled dataset.

2. **Data Splitting**
   - The dataset is split into training and test sets using `train_test_split` from scikit-learn.

3. **Model Training**
   - The code performs a grid search with cross-validation to find the best hyperparameters for the Random Forest Classifier.
   - The best model is trained on the training data using the optimal hyperparameters.

4. **Model Evaluation**
   - The trained model is evaluated on the test data.
   - The code calculates and prints the accuracy score, confusion matrix, classification report, and ROC AUC score.
   - It also visualizes the ROC curve and the confusion matrix using matplotlib and seaborn.

5. **User Input and Prediction**
   - The code prompts the user to enter values for each feature.
   - It scales the user input using the same `StandardScaler` object used for the training data.
   - The scaled user input is fed into the trained model, and the prediction (diabetes or not) is printed.

## Usage

1. Make sure you have the required Python libraries installed.
2. Place the `diabetes.csv` dataset in the same directory as the code file.
3. Run the code.
4. When prompted, enter the values for the features (number of pregnancies, glucose level, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age).
5. The code will print the prediction (whether the patient has diabetes or not) based on the user input.

Note: The code assumes that the `diabetes.csv` dataset is in the correct format and contains all the required features.
