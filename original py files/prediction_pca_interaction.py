'''
# Predicting SalePrice for the Housing Pricing Kaggle Project

### SalePrice is log-transformed
### Using PCA 100 components
### Adding interaction and quandratic terms
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
from sklearn.preprocessing import PolynomialFeatures as Poly
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
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
### combine the predictors from training and testing sets for one-hot-encoding
### then split into training and testing sets again
'''
# convert quant values to float
for name in quant_names:
    train[name] = train[name].astype(float)
    if name != 'SalePrice':
        test[name] = test[name].astype(float)
    
# combine train and test predictors
predictors = pd.concat((train[train.columns[:-1]], test))
# one-hot_encode the categorical columns
dummies = pd.get_dummies(predictors[categorical_names])
# combine the cat and quant columns
# remember to remove SalePrice from the quant names
quant_names = list(set(train.columns[:-1])-set(categorical_names)) 
predictors_expanded = pd.concat([predictors[quant_names],dummies], axis = 1)
# split into training and testing predictors
n_train = train.shape[0]
x_train = predictors_expanded.values[:n_train, ]
x_test = predictors_expanded.values[n_train:, ]
# the response variable for the training set
y_train = train['SalePrice'].values

#%%
'''
## Log transformation on SalePrice --> normal dist
'''
### check distribution of SalePrice
fig = plt.figure(figsize = (10, 5))
ax1 = fig.add_subplot(121)
ax1.hist(y_train)

ax2 = ax1 = fig.add_subplot(122)
ax2.hist(np.log(y_train))

y_train_log = np.log(y_train)
#%%
'''
## Principal Component Analysis
### Use the number of components that yield similar R^2 as the original 
### training data by linear regression for further analysis
'''
pca = PCA()
x_train_pca = pca.fit_transform(x_train)

### check the R-squared of linear regression on original training data
linear = LinearRegression()
linear.fit(x_train, y_train_log)
linear.score(x_train, y_train_log) 

### check the value of linear regression on different numbers of PCs
for i in np.arange(61, 101, 1):
    linear.fit(x_train_pca[:,:i], y_train_log)
    score = linear.score(x_train_pca[:,:i], y_train_log)
    
    print "{}: {}".format(i, score)

#%%
'''
### Decide to use the first 100 PCs
'''
x_train_reduced = x_train_pca[:, :101]
x_test_reduced = pca.transform(x_test)[:, :101]
#%%
'''
### Adding quandratic terms and interaction terms
'''
poly = Poly(degree = 2, include_bias = False)
x_train_poly = poly.fit_transform(x_train_reduced)
x_test_poly = poly.transform(x_test_reduced)
#%%
'''
## Standardize predictors
'''
std = Standardize(with_mean=False)
x_train_reduced_std = std.fit_transform(x_train_poly)
x_test_reduced_std = std.transform(x_test_poly)
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
grid_model.fit(x_train_reduced_std, y_train_log)
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
grid_model.fit(x_train_reduced_std, y_train_log)
# best model
ridge = grid_model.best_estimator_
print "The score of the best ridge model is ", grid_model.best_score_
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
            score = r2_score(y_train_log, y_pred_ratio)
            
            if score > best_score:
                best_score = score
                best_ratio = ratio
        
        return best_score, best_ratio


best_score_1 = 0
best_alpha1 = 0
    
for alpha1 in alpha1_list:
    # lasso
    elastic_l1 = Lasso_Reg(alpha = alpha1)
    y_pred_l1 = cross_val_predict(elastic_l1, x_train_reduced_std, y_train_log,
                                  cv = skf)
    
    # tune alpha2
    best_score_2 = 0
    best_alpha2 = 0
    
    for alpha2 in alpha2_list:
        # ridge
        elastic_l2 = Ridge_Reg(alpha = alpha2)
        y_pred_l2 = cross_val_predict(elastic_l2, x_train_reduced_std, 
                                      y_train_log,
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
# ID of the testing data
n_test = x_test.shape[0]
ID = np.arange(1461, 2920, 1).astype(int)
ID = ID.reshape((n_test, 1))

# store models into a list
model_list = [lasso, ridge]
model_names = ['lasso', 'ridge']

# repeat for each model
for i in range(len(model_list)):
    model = model_list[i]
    model_name = model_names[i]
    
    # prediction on testing data
    model.fit(x_train_reduced_std, y_train_log)
    y_pred_test = np.exp(model.predict(x_test_reduced_std))
    y_pred_test = y_pred_test.reshape((n_test, 1))
    # save the data
    y_pred = np.concatenate([ID, y_pred_test], axis = 1)
    filename = "results/{}_test.csv".format(model_name)
    np.savetxt(filename, y_pred, fmt='%.i', delimiter=',',  header='Id,SalePrice')
    
# elastic net
elastic_l1 = Lasso_Reg(alpha = best_alpha1)
elastic_l1.fit(x_train_reduced_std, y_train_log)
y_pred_l1 = elastic_l1.predict(x_test_reduced_std)

elastic_l2 = Ridge_Reg(alpha = best_alpha2)
elastic_l2.fit(x_train_reduced_std, y_train_log)
y_pred_l2 = elastic_l2.predict(x_test_reduced_std)

y_pred = best_ratio * y_pred_l1 + (1-best_ratio)*y_pred_l2
# transform back
y_pred = np.exp(y_pred).reshape((n_test,1))
y_pred = np.concatenate([ID, y_pred], axis = 1)
filename = "results/elastic_test.csv".format(model_name)
np.savetxt(filename, y_pred, fmt='%.i', delimiter=',',  header='Id,SalePrice')
#%%

