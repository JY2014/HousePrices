# Predicting House Prices: Advanced Regression

This project attempts to predict sales prices for residential homes using information such as physical structure, location and condition of the home, in order to guide the estimation of individual house prices and assist the studies on housing market trends.  


## Data
Predicting final sales price of houses using 79 explanatory variables describing almost every aspect of residential homes in Ames, Iowa, downloaded from the [Kaggle website](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

**Data Files**:
- ***data/train.csv***: the training set
- ***data/test.csv***: the test set
- ***data/data_description.txt***: full description of each variable

## Summary of Analysis
My analysis includes the following steps (python code in the ipynb files):
- Imputation on the missing values: ***Data Cleaning and Imputation.ipynb***
- Data transformation and cleaning: ***Data Cleaning and Imputation.ipynb***
- Prediction using multiple models: ***Prediction.ipynb, Prediction_PCA.ipynb, XGBoost.ipynb, XGBoost_PCA.ipynb***
- Ensemble learning on the model results: ***Model_Ensemble.ipynb***

## 1. Imputation on Missing Values

About 30 predictor variables contain missing values.

<center><b>Figure 1. Percentage Missingness of Predictors with Missing Values</b></center>

![Missing data](https://cloud.githubusercontent.com/assets/9686980/23108119/59b8e322-f6d6-11e6-9eba-8d8a6c988190.JPG)

**Steps**:

1. Predictors with more than 40% missingness were eliminated from the analysis;
2. Remaining missing values were imputed:
    - Quantitative variable: better model between lasso or ridge regression (regularization parameter tuned)
    - Categorical variable: logistic regression (regularization parameter tuned)

## 2. Data Transformation and Cleaning

1. **Response Variable**:

    Since the price is highly right skewed, log-transformation was employed to transform the variable. 
    
 <center><b>Figure 2. Log-transformation on the response variable (sales price)</b></center>   

![transformation](https://cloud.githubusercontent.com/assets/9686980/23108947/2caadd2a-f6e2-11e6-9eac-df6160682a67.JPG)

2. **Quantitative Predictors**:

    - log-transformed if highly right skewed;
    - if includes a lot of zero values, a binary variable was added to indicate zero vs non-zero.


## 3. Prediction

Multiple regressing models were applied to predict the log-transformed sales price using the predictors:

- Lasso regression
- Ridge regression
- Random forest regressor
- XGBoost

Since the predictor variables are highly correlated, transformation of the predictors using principal component analysis was also examined. However, the PCA transformed data did not yield better prediction.  

## 4. Ensemble Learning

The prediction of each model on both the training and testing sets was saved from the previous step. 
- Training set: cross-validation prediction
- Testing set: prediction using the training set

The data were further fed into predictive models to generate better prediction on the testing set.

Models include
- Lasso regression
- Ridge regression
- Random forest regressor
- XGBoost

## Model Comparison
Root Mean Squared Logarithmic Error (RMSLE) is used to evaluate each model.

| Basic Model |   RMSLE   |               
|-------------|-----------|               
|   Lasso     |   0.1326  |               
|   Ridge     |   0.1366  |               
|Random Forest|   0.1462  |               
|  XGBoost    |**0.1188** |           


| Ensemble    |   RMSLE   |
|-------------|-----------|
|   Lasso     | **0.1194**|
|   Ridge     |   0.1195  |
|Random Forest|   0.1219  |
|  XGBoost    | **0.1180**|

### Summary
- Lasso regression and XGBoost in general perform the best on this dataset;
- Ensemble learning is able to boost the prediction precision by integrating individual models.
