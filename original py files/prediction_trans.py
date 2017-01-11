'''
# Predicting SalePrice for the Housing Pricing Kaggle Project

### SalePrice is log-transformed
'''
#%%
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge as Ridge_Reg
from sklearn.linear_model import Lasso as Lasso_Reg
from sklearn.linear_model import ElasticNetCV as ElasticNet
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler as Standardize
from sklearn.metrics import r2_score
from scipy.stats import skew
#%%
'''
## import data
'''
### import imputed training and testing data
train = pd.read_csv('imputed_train.csv')
test = pd.read_csv('imputed_test.csv')
# check dimensions
print "Training dimension: ", train.shape
print "Testing dimension: ", test.shape
### import names of categorical columns
categorical_names = pd.read_csv('categorical_col.csv')['0'].values
quant_names = list(set(train.columns)-set(categorical_names)) 
#%%
'''
## clean the data
### convert quantitative values to float
'''
# convert quant values to float
for name in quant_names:
    train[name] = train[name].astype(float)
    if name != 'SalePrice':
        test[name] = test[name].astype(float)
        
#%%
'''
### combine the predictors from training and testing sets for one-hot-encoding
'''
# combine train and test predictors
predictors = pd.concat((train[train.columns[:-1]], test))
# one-hot_encode the categorical columns
dummies = pd.get_dummies(predictors[categorical_names])
# combine the cat and quant columns
# remember to remove SalePrice from the quant names
quant_names.remove('SalePrice') 
predictors_expanded = pd.concat([predictors[quant_names],dummies], axis = 1)
#%%
'''
Log transformation on SalePrice --> normal dist
'''
# the response variable for the training set
y_train = train['SalePrice'].values

### check distribution of SalePrice
fig = plt.figure(figsize = (10, 5))
ax1 = fig.add_subplot(121)
ax1.hist(y_train)

ax2 = ax1 = fig.add_subplot(122)
ax2.hist(np.log(y_train))

y_train_log = np.log(y_train)
   #%%
'''
### Check relationship between response variable with each quantitative variable
### transformation quantitative variables:
#### 1. add a binary variable to indicate non-zero values if many zero-values
#### 2. log transform non-zero values if highly skewed
'''
n_train = train.shape[0]

# check relationship between response and each quant predictor
for name in quant_names:
    print name
    fig = plt.figure(figsize = (5, 5))
    ax1 = fig.add_subplot(111)
    ax1.scatter(predictors_expanded[name].values[:n_train, ],y_train_log)

# add binary variable to indicate non-zero values
add_binary_index = [0, 1, 2, 5, 6, 9, 11, 12, 13, 14, 17, 18, 19, 20, 21]

for i in add_binary_index:
    name = quant_names[i]
    # add a binary column
    binary_col = np.copy(predictors_expanded[name].values)
    binary_col[binary_col > 0] = 1
    # combine to the predictor dataframe
    predictors_expanded[name+'-binary'] = binary_col
    
# log-transform skewed quant predictors
log_transform_index = [0, 1, 3, 4, 5, 6, 10, 11, 13, 18]

for i in log_transform_index:
    name = quant_names[i]
    # log transform
    predictors_expanded[name] = np.log(predictors_expanded[name].values+0.001)
#%%  
# split into training and testing predictors
x_train = predictors_expanded.values[:n_train, ]
x_test = predictors_expanded.values[n_train:, ]
#%%
'''
## Simple linear regression
'''
# use K-fold validation
skf = StratifiedKFold(n_splits = 5, shuffle = True)

linear = LinearRegression()
y_pred = cross_val_predict(linear, x_train, y_train, cv = skf)
# calculate R_square
print "Score of the simple linear regression model ", r2_score(y_train, y_pred)
#%%
'''
## Standardize predictors
'''
std = Standardize(with_mean=False)
x_train_std = std.fit_transform(x_train)
x_test_std = std.transform(x_test)
#%%
'''
## Lasso regression
'''
# use K-fold validation
skf = StratifiedKFold(n_splits = 5, shuffle = True)
# parameters
param = np.power(10.0, range(-4, 5, 1))
# tune lasso regression
model = Lasso_Reg()
grid_model = GridSearchCV(model, param_grid = {'alpha': param}, cv  = skf)
grid_model.fit(x_train_std, y_train_log)
# best model
lasso = grid_model.best_estimator_
print "The score of the best lasso model is ", grid_model.best_score_

#%%
'''
## Ridge regression
'''
# tune lasso regression using the same parameter list
model = Ridge_Reg()
grid_model = GridSearchCV(model, param_grid = {'alpha': param}, cv  = skf)
grid_model.fit(x_train_std, y_train_log)
# best model
ridge = grid_model.best_estimator_
print "The score of the best ridge model is ", grid_model.best_score_
#%%
''' 
## Elastic Net
### using the sklearn function
'''
# parameters
l1_ratio_list = np.asarray([.1, .5, .7, .9, .95, .99, 1])

# tune elastic net
model = ElasticNet()
grid_model = GridSearchCV(model, 
                          param_grid = {'l1_ratio': l1_ratio_list}, cv  = skf)
grid_model.fit(x_train_std, y_train_log)
# best model
elastic = grid_model.best_estimator_
print "The score of the best elastic net model is ", grid_model.best_score_
#%%
''' 
## Elastic Net
### implement: tune alpha for both models and the ratio
'''
# alpha for lasso
alpha1_list = np.power(10.0, range(-4, 5, 1))
# alpha for ridge
alpha2_list = np.power(10.0, range(-4, 5, 1))
# ratio
ratio_list = np.arange(0, 1.1, 0.1) 

### function to find the best ratio of lasso and ridge
## input: prediction from ridge and lasso
## output: best R_squared and best ratio (number 1:2)
def find_best_ratio(y_pred_1, y_pred_2, ratio_list):
    # find the best ratio
        best_ratio = -1
        best_score = 0
        for ratio in ratio_list:
            y_pred_ratio = ratio * y_pred_1 + (1-ratio)*y_pred_2
            y_pred_ratio = np.exp(y_pred_ratio)
            score = r2_score(y_train, y_pred_ratio)
            
            if score > best_score:
                best_score = score
                best_ratio = ratio
        
        return best_score, best_ratio


best_score_1 = 0
best_alpha1 = 0
    
for alpha1 in alpha1_list:
    # lasso
    elastic_l1 = Lasso_Reg(alpha = alpha1)
    y_pred_l1 = cross_val_predict(elastic_l1, x_train_std, y_train_log, 
                                  cv = skf)
    
    # tune alpha2
    best_score_2 = 0
    best_alpha2 = 0
    
    for alpha2 in alpha2_list:
        # ridge
        elastic_l2 = Ridge_Reg(alpha = alpha2)
        y_pred_l2 = cross_val_predict(elastic_l2, x_train_std, y_train_log, 
                                      cv = skf)
        
        # tune the ratio
        best_score_3, best_ratio = find_best_ratio(y_pred_l1, 
                                                   y_pred_l2, ratio_list)
    
        if best_score_3 > best_score_2:
            best_score_2 =  best_score_3
            best_alpha2 = alpha2
    
    # tune alpha1
    if best_score_2 > best_score_1:
        best_score_1 =  best_score_2
        best_alpha1 = alpha1

print "The best model is lasso (alpha = {}) to ridge (alpha = {}) by {}:{}.".format(
best_alpha1, best_alpha2, best_ratio, (1-best_ratio))
print "The best score is ", best_score_1
#%%
'''
## store the prediction of testing sets
'''
# store models into a list
model_list = [lasso, ridge, elastic]
model_names = ['lasso', 'ridge', 'elastic']
submission = pd.read_csv('sample_submission.csv')

# repeat for each model
for i in range(len(model_list)):
    model = model_list[i]
    model_name = model_names[i]
    
    # prediction on testing data
    model.fit(x_train_std, y_train_log)
    y_pred_test = np.exp(model.predict(x_test_std))
    #y_pred_test = y_pred_test.reshape((n_test, 1))
    # save the data
    submission['SalePrice'] = y_pred_test
    filename = "results/{}_test.csv".format(model_name)
    submission.to_csv(filename, sep = ',')
    
# elastic net
elastic_l1 = Lasso_Reg(alpha = best_alpha1)
elastic_l1.fit(x_train_std, y_train_log)
y_pred_l1 = elastic_l1.predict(x_test_std)

elastic_l2 = Ridge_Reg(alpha = best_alpha2)
elastic_l2.fit(x_train_std, y_train_log)
y_pred_l2 = elastic_l2.predict(x_test_std)

y_pred = best_ratio * y_pred_l1 + (1-best_ratio)*y_pred_l2
filename = "results/elastic_test_2.csv".format(model_name)
# save the data
submission['SalePrice'] = np.exp(y_pred)
submission.to_csv(filename, sep = ',')
#%%
'''
## learn from all the model results using linear regression
'''
### calculate the training y_pred from all the models
n_train = x_train.shape[0]
y_train_results = np.asarray(np.arange(1,n_train+1, 1)).reshape((n_train, 1))
# repeat for each model
for i in range(len(model_list)):
    model = model_list[i]
    model_name = model_names[i]
    # predict training data using each model
    y_pred_train = cross_val_predict(model, x_train_std, y_train_log, cv = skf)
    y_pred_train = np.exp(y_pred_train)
    y_pred_train = y_pred_train.reshape((n_train, 1))
    y_train_results = np.concatenate([y_train_results, y_pred_train], axis = 1)

# elastic net
y_pred_l1 = cross_val_predict(elastic_l1, x_train_std, y_train_log, cv = skf)
y_pred_l2 = cross_val_predict(elastic_l2, x_train_std, y_train_log, cv = skf)
y_pred_train = best_ratio * y_pred_l1 + (1-best_ratio)*y_pred_l2
y_pred_train = np.exp(y_pred_train)
y_pred_train = y_pred_train.reshape((n_train, 1))
y_train_results = np.concatenate([y_train_results, y_pred_train], axis = 1)
#%%
### using linear regression to learn from the model
linear = LinearRegression()
y_pred = cross_val_predict(linear, y_train_results[:, 1:], y_train, cv = skf)
# calculate R_square
print "Score of the simple linear regression model ", r2_score(y_train, y_pred)
#%%
'''
XGBoost
'''
import xgboost as xgb
#%%
### test
dtrain = xgb.DMatrix(x_train, label = y_train_log)
dtest = xgb.DMatrix(x_test)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
#%%
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
#%%
model_xgb = xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.1)
model_xgb.fit(x_train_reduced, y_train_log)
y_pred = model_xgb.predict(x_test_reduced)
y_pred = np.exp(y_pred)

# save the data
submission['SalePrice'] = y_pred
filename = "results/xgb.csv"
submission.to_csv(filename, sep = ',')

#%%
'''
function to compute RMSE
'''
def RMSE(y_true, y_pred):
    n = len(y_true)
    rmse = sum((y_true - y_pred)^2)/n
    
    return rmse
#%%