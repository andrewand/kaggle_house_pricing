#!/usr/bin/env python

import numpy as np
import pandas as pd
import xgboost as xgb
import math
import random
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold


# We may also consider using RandomizedSearchCV


train, test = load_data()

param = {'silent': 1, 'objective': 'reg:linear'}

result = []
best = [None, None, None, None, None, 1]

random_params = {
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5, 6],
    'eta': [0.01, 0.02, 0.05, 0.075, 0.1]}

cv_param = [
    [subsample, colsample_bytree, max_depth, eta]
    for subsample in random_params['subsample']
    for colsample_bytree in random_params['colsample_bytree']
    for max_depth in random_params['max_depth']
    for eta in random_params['eta']]

for i in random.sample(range(len(cv_param)), 15):
    subsample, colsample_bytree, max_depth, eta = cv_param[i]
    param['subsample'] = subsample
    param['colsample_bytree'] = colsample_bytree
    param['max_depth'] = max_depth
    param['eta'] = eta
    cv = xgb.cv(param, dtrain, 10000, nfold=5, early_stopping_rounds=100, verbose_eval=1)
    result.append(cv_param[i])
    result[-1].append(cv.shape[0])
    result[-1].append(cv.iloc[-1]['test-rmse-mean'])
    if best[-1] > cv.iloc[-1]['test-rmse-mean']:
        best = result[-1]

result = pd.DataFrame(result)
result.columns = ['subsample', 'colsample_bytree', 'max_depth', 'eta', 'num_tree', 'error']
print result
print best

param['subsample'] = best[0]
param['colsample_bytree'] = best[1]
param['max_depth'] = best[2]
param['eta'] = best[3]
num_tree = best[-2]
model = xgb.train(param, dtrain, cv.shape[0])
prediction = model.predict(dtest)

result = pd.concat([data.loc[data['data_type'] == 'test']['Id'], pd.Series(prediction)], axis=1)
result.columns = ['Id', 'logSalePrice']
result['SalePrice'] = result['logSalePrice'].apply(math.exp)
result = result.drop(['logSalePrice'], axis=1)
result.to_csv('new_attempt.csv', index=False)
