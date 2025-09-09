# Machine Learning Project 1 – Classification Algorithms

Course: DASC 41103 – Machine Learning  
Members: Coy Morris, Hunter Hill

## Project Overview

This project applies foundational classification algorithms to the UCI Adult Income dataset to predict whether an individual earns more than $50K/year.

We implemented and compared Perceptron, Adaline (SGD), Logistic Regression, and Support Vector Machines (SVM) models. Our pipeline covers data preprocessing, algorithm implementation, hyperparameter tuning, visualization of decision boundaries, and accuracy evaluation on a validation dataset.

## Repository Structure

```
├── data/
│   ├── project_adult.csv
│   └── project_validation_inputs.csv
├── scripts/
│   ├── preprocess.py          # Data cleaning, encoding, scaling
│   ├── perceptron_adaline.py  # Custom Perceptron & Adaline implementations
│   ├── logistic_svm.py        # Logistic Regression and SVM models
│   └── utils.py               # Helper functions, plotting utilities
├── outputs/
│   ├── Group_#_Perceptron_PredictedOutputs.csv
│   ├── Group_#_Adaline_PredictedOutputs.csv
│   ├── Group_#_LogisticRegression_PredictedOutputs.csv
│   └── Group_#_SVM_PredictedOutputs.csv
├── notebooks/                 # (Optional) for exploratory analysis
├── figures/                   # Misclassification/MSE plots & decision boundaries
└── README.md                  # This file
```

