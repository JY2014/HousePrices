'''
# Data exploration for the Housing Pricing Kaggle Project
'''
#%%
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge as Ridge_Reg
from sklearn.linear_model import Lasso as Lasso_Reg
#%%

### import training and testing data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
n1,n2 = train.shape
#%%
'''
## Missing values: training set
### explore the number of missingness in each variable in the training set
### drop columns with more than 40% missing values (natural cut of the data)
'''
### remove some columns
# remove SalePrice from training data
train_dropped = train.drop("SalePrice", axis = 1)
# drop Utilities because it has only one value
train_dropped = train_dropped.drop('Utilities', axis = 1)
test_dropped = test.drop('Utilities', axis = 1)

### check missing data in each variable
# store number of missing values 
missing_num_train = train_dropped.isnull().sum()
# find columns with missingness
missing_column_train_index = (missing_num_train != 0)
missing_column_train_names = train_dropped.columns[missing_column_train_index]
missing_num_train[missing_column_train_index]/n1
# drop columns with more than 40% missingness
dropped_columns = train_dropped.columns[missing_num_train >= 0.4 * n1]
train_dropped = train_dropped.drop(dropped_columns, axis = 1)
test_dropped = test_dropped.drop(dropped_columns, axis = 1)

# check dimension
print "New training dimension: ", train_dropped.shape
print "New testing dimension: ", test_dropped.shape
#%%
'''
## Missing values: testing set
### explore the number of missingness in each variable in the testing set
### columns with more than 40% missingness have been dropped
'''
missing_num_test = test_dropped.isnull().sum()
missing_column_test_index = (missing_num_test != 0)
missing_column_test_names = test_dropped.columns[missing_column_test_index]
#%%
'''
## Combine columns with missing values from both training and testing sets
'''
missing_column_names = set().union(missing_column_test_names, 
missing_column_train_names)
#%%
'''
## Determine categorical or quantitative predictors of the remaining columns
### variables with less than 15 values are set to be categorical
### non-numerical values also indicate categorical variables
'''
### extract variables with less than 10 values and non-numerical values
value_num = [] #store number of unique values in each column
non_numerical_names = []

for name in train_dropped.columns:
    # number of unique values in each column
    value_num.append(len(train_dropped[name].unique()))
    # check if the first value is numerical
    if not (train_dropped[name].dtype == np.dtype('int64') or 
    train_dropped[name].dtype == np.dtype('float64')):
        non_numerical_names.append(name)
# extract columns with less than 15 values       
value_num = np.asarray(value_num)
categorical_names = train_dropped.columns[value_num <= 15]

# the categorial predictors are the union of the two lists
categorical_names = list(categorical_names)
categorical_final = set().union(categorical_names, non_numerical_names)
categorical_final = list(categorical_final)

#%%
### Check which variables with missing values are categorical
# it is the intersect of categorical_final and missing_column_names
missing_cat_names = list(set(missing_column_names) & set(categorical_final))
# the rest are quantitative
missing_quant_names = list(set(missing_column_names)-set(missing_cat_names)
-set(dropped_columns))
#%%
'''
## Impute missing values
### Using variables without missing values as training set
### to impute missing values

### Categorical variables: regularized logistic
### Quantitative variables: try lasso and Ridge 
#### tune the parameter and use the best model to impute
'''
#%%
### training data for imputation 
# extract training data without missing values to use for imputation
complete_column_names = list(set(train_dropped.columns)
-set(missing_column_names))
complete_train = train_dropped[complete_column_names]
complete_test = test_dropped[complete_column_names]
# need to combine the complete data from training and testing sets
complete_data = pd.concat((complete_train,complete_test))
print "Dimension of the complete data: ", complete_data.shape
# turn quant variables into float numbers
quant_complete_names = list(set(complete_data.columns)-set(categorical_final))
for name in quant_complete_names:
    complete_data[name] = complete_data[name].astype(float)
# one-hot_encode the categorical columns
cat_complete_names = list(set(complete_train.columns)-set(quant_complete_names))
dummies = pd.get_dummies(complete_data[cat_complete_names])
# combine the cat and quant columns
data_for_impute = pd.concat([complete_data[quant_complete_names],
                              dummies], axis = 1)
# separate the training and testing data from data_for_impute
train_for_impute = data_for_impute.iloc[:n1]
test_for_impute = data_for_impute.iloc[n1:]
#%%
### function to impute missing values for each column
## input 1: complete columns from training set
## input 2: complete columns from testing set
## input 3: column to impute in training data as a Pandas Series
## input 4: column to impute in testing data as a Pandas Series
## input 5: variable type: 1 = categorical; 2 = quantitative
#
## output: training and testing columns with missing data filled
def data_impute(train_complete, test_complete, train_impute, test_impute,
               variable_type):
    index1 = train_impute.isnull()
    index2 = test_impute.isnull()
    # response variable
    y_complete = np.concatenate((train_impute[index1 == False].values,
                              test_impute[index2 == False].values))
    y_train = train_impute[index1 == True].values
    y_test = test_impute[index2 == True].values
    # predictors
    x_complete = np.concatenate((train_complete[index1 == False].values,
                              test_complete[index2 == False].values), axis = 0)
    x_train = train_for_impute[index1 == True].values
    x_test = test_for_impute[index2 == True].values
    
    # categorical
    if variable_type == 1:
        # tune logistic regression model
        param, score = tune_logistic(x_complete, y_complete)
        print "The best param is {} with accuracy of {}.".format(param, score)
        # imputation
        model = LogisticRegression(C = param)
        model.fit(x_complete, y_complete)
    
    # quantitative
    elif variable_type == 2:
        # tune ridge or lasso regression
        penalty, param, score = tune_ridge_lasso(x_complete, y_complete)
        print "The penalty is L{} with best param of {} and accuracy of {}.".format(
        penalty, param, score)
        # imputation
        if penalty == 1:
            model = Lasso_Reg(alpha = param)
        elif penalty == 2:
            model = Ridge_Reg(alpha = param)
        model.fit(x_complete, y_complete)
    
    # impute the missing data 
    if y_train.shape[0] > 0:
        y1 = model.predict(x_train)
        train_impute[index1 == True] = y1
    if y_test.shape[0] > 0:
        y2 = model.predict(x_test)
        test_impute[index2 == True] = y2
    
    return train_impute, test_impute
 
   
### function to tune regularization parameter for logistic regression
## input: predictors x and response y
## output: best parameter and accuracy
## selection based on classification accuracy
def tune_logistic(x, y):
    #shuffe the data
    perm = range(y.shape[0])
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]
    # split train and test sets (7:3)
    [x_train, x_test, y_train, y_test] = train_test_split(x, y, train_size=0.7)
    # set of paramters to test
    param_list = np.power(0.1, np.arange(-4, 4))
    # tune the model
    best_param = 0
    best_score = 0
    
    for param in param_list:
        log = LogisticRegression(C = param)
        log.fit(x_train, y_train)
        score = log.score(x_test, y_test)
        # update
        if score > best_score:
            best_score = score
            best_param = param
    
    return best_param, best_score
    
### function to tune ridge or lasso regression
## input: predictors x and response y
## output: penalty type (1: lasso, 2: ridge), best parameter and accuracy
## selection based on R^2
def tune_ridge_lasso(x, y):
    #shuffe the data
    perm = range(y.shape[0])
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]
    # split train and test sets (7:3)
    [x_train, x_test, y_train, y_test] = train_test_split(x, y, train_size=0.7)
    # set of paramters to test
    param_list = np.power(0.1, np.arange(-7, 7))
    
    # tune the model
    best_param = 0
    best_score = 0
    best_penalty = 0
    
    for param in param_list:
        ridge = Ridge_Reg(alpha = param)
        ridge.fit(x_train, y_train)
        score1 = ridge.score(x_test, y_test)
        
        lasso = Lasso_Reg(alpha = param)
        lasso.fit(x_train, y_train)
        score2 = lasso.score(x_test, y_test)
    # compare between the two scores first
        if score1 > score2:
            penalty = 2
            score = score1
        else:
            penalty = 1
            score = score2
        # update
        if score > best_score:
            best_score = score
            best_param = param
            best_penalty = penalty
        
        return best_penalty, best_param, best_score
#%%
### impute missing values for categorical variables
print "CATEGORICAL:"
for name in missing_cat_names:
    print name
    train_dropped[name], test_dropped[name] = data_impute(
    train_for_impute,test_for_impute, train_dropped[name], test_dropped[name],1)
  
#%%  
### impute missing values for categorical variables
print "QUANTITATIVE:"
for name in missing_quant_names:
    print name
    train_dropped[name], test_dropped[name] = data_impute(
    train_for_impute,test_for_impute, train_dropped[name], test_dropped[name],2)
    
#%%
### Save the imputed data into a csv file
# add back SalePrice to the training data
train_dropped = pd.concat((train_dropped, train['SalePrice']), axis = 1)
train_dropped.to_csv("imputed_train.csv", sep = ',')
test_dropped.to_csv("imputed_test.csv", sep = ',')

### Save names of the categorical columns for later use
df_categorical = pd.DataFrame(categorical_final)
df_categorical.to_csv("categorical_col.csv", sep = ',')
#%%