# Predicting House Prices: Advanced Regression

## Data
Predicting final sales price of houses using 79 explanatory variables describing almost every aspect of residential homes in Ames, Iowa, downloaded from the [Kaggle website](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).
### Data Files
- *data/train.csv*: the training set
- *data/test.csv*: the test set
- *data/data_description.txt*: full description of each variable

## Summary of Analysis
My analysis includes the following steps
- Imputation on the missing values 
- Data Exploration and Cleaning
- Prediction using multiple models
- Ensemble learning on the model results

## Imputation on Missing Values
(*Data Cleaning and Imputation.ipynb*)
The about 30 predictor variables include missing values to various extents.

<center><b>Figure 1. Percentage Missingness of Predictors with Missing Values</b></center>
![Missing data](https://cloud.githubusercontent.com/assets/9686980/23108119/59b8e322-f6d6-11e6-9eba-8d8a6c988190.JPG)

1. Predictors with more than 40% missingness are eliminated from the analysis;
2. Remaining missingness are imputed:
    - Quantitative variable: better model between lasso or ridge regression (regularization parameter tuned)
    - Categorical variable: logistic regression (regularization parameter tuned)

## Data Cleaning


## Prediction


## Model Comparison
