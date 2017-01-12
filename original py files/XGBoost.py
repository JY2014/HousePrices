#%%
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.grid_search import GridSearchCV
#%%
# read the data
x_train = np.loadtxt('x_train_cleaned.txt', delimiter=',')
x_test = np.loadtxt('x_test_cleaned.txt', delimiter=',')
y_train_log = np.loadtxt('y_train_log.txt', delimiter=',')
#%%
xgb_model = xgb.XGBRegressor().fit(x_train,y_train_log)
y_pred = xgb_model.predict(x_test)

# convert to original scale and save
submission = pd.read_csv('sample_submission.csv')
submission['SalePrice'] = np.exp(y_pred)
submission.to_csv('results/xgb.csv', sep = ',', index=False)
#%%
dtrain = xgb.DMatrix(x_train, label=y_train_log)
dtest = xgb.DMatrix(x_test)

param = {'max_depth':2, 'eta':1, 'silent':0, 'objective':'reg:linear'}
num_round = 200
xgb.cv(param, dtrain, num_round, nfold=5,
       metrics={'rmse'}, seed = 0,
       callbacks=[xgb.callback.print_evaluation(show_stdv=False)])
       
cvresult = xgb.cv(param, dtrain, num_boost_round=1000, nfold=5,
       metrics='rmse', early_stopping_rounds=50)

#%%
def modelfit(alg, x_train, y_train, useTrainCV=False, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(x_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='rmse', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
 
    #Fit the algorithm on the data
    alg.fit(x_train, y_train, eval_metric='rmse')
     
    #Predict training set:
    dtrain_predictions = alg.predict(x_train)
    dtrain_predprob = alg.predict_proba(x_train)[:,1]
     
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.rmse(x_train, y_train_log)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(x_train, dtrain_predprob)
     
    #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')
#%%
xgb1 = xgb.XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'reg:linear',
 nthread=4,
 seed=27)
modelfit(xgb1, x_train, y_train_log)
#%%
cv_folds = 5
early_stopping_rounds = 50

xgb_param = xgb1.get_xgb_params()
xgtrain = xgb1.DMatrix(x_train, label=y_train_log)
cvresult = xgb1.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
    metrics='rmse', early_stopping_rounds=early_stopping_rounds)
print cvresult.shape[0]
alg.set_params(n_estimators=cvresult.shape[0])
 
