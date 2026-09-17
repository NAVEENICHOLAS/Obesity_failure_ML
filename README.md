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
