# Obesity Classification — ML Failure Analysis

## 📌 Project Overview

This project explores a machine learning approach to classify individuals into five obesity-related categories using demographic, physical, and lifestyle features.

The goal of this project was not only to build a machine learning model, but also to understand what happens when a model performs poorly and how to investigate the reasons behind that failure.

## 🎯 Objective

Predict the `Obesity_Level` of an individual using:

- Age
- Height
- Weight
- BMI
- Physical activity frequency
- Water intake
- Gender
- Family history of obesity
- Dietary habits
- Smoking habits
- Alcohol consumption

The target variable contains five categories:

- Normal Weight
- Overweight
- Obese Type I
- Obese Type II
- Obese Type III

## 📊 Dataset

The dataset contains **1,000 records** and includes both numerical and categorical features.

### Data preparation

The following preprocessing steps were performed:

- Checked for missing values
- Checked for duplicate records
- Converted height from meters to centimeters
- Created BMI as a new feature
- Separated features (`X`) from the target (`y`)
- Applied one-hot encoding to categorical variables
- Split the dataset into 80% training and 20% testing data

No duplicate rows or missing values were found during the initial data-cleaning checks.

## 🧠 Model

A **Random Forest Classifier** was used as the baseline model.

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```


LEARNINGS 
This project taught me that machine learning is not just about training a model and looking for a high accuracy score.

### Key lessons

- Learned the difference between classification and regression.
- Learned how to identify numerical and categorical features.
- Learned how to perform basic data cleaning and validation.
- Learned how feature engineering can create useful variables such as BMI.
- Learned how one-hot encoding converts categorical variables into numerical features.
- Learned how to split data into training and testing sets.
- Learned how a Random Forest classifier works using multiple decision trees.
- Learned how to evaluate a classification model using accuracy and a confusion matrix.
- Learned how to interpret feature importance.
- Learned that feature importance does not imply causation.
- Learned that poor model performance can originate from the data and target labels, not necessarily from the algorithm itself.
- Learned to investigate a model's errors before blindly changing hyperparameters.
- Learned that inconsistent or overlapping target classes can make supervised learning difficult.


## ❌ Why Did the Baseline Model Fail?

The Random Forest baseline achieved only **17.5% accuracy** on the test set.

Further investigation showed substantial overlap in BMI values across all five obesity categories. The dataset also contains examples where BMI and the assigned `Obesity_Level` appear inconsistent.

For example:

| BMI | Assigned Obesity Level |
|---:|---|
| 20.55 | Obese Type I |
| 37.22 | Overweight |
| 35.16 | Normal Weight |
| 21.53 | Obese Type II |

Because the target labels do not show a clean relationship with BMI and other available features, the model has difficulty learning reliable decision boundaries between the five classes.

The failure therefore became an opportunity to investigate **data quality, feature-target relationships, and model behavior** rather than simply increasing the number of trees or changing algorithms.

