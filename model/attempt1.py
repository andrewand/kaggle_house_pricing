#!/usr/bin/env python

import numpy as np
import pandas as pd
import xgboost as xgb
import math
import random
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold


train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
test['SalePrice'] = np.nan
train['data_type'] = 'train'
test['data_type'] = 'test'
data = pd.concat([train, test])

data['logSalePrice'] = data['SalePrice'].apply(math.log)
data = data.drop('SalePrice', axis=1)

data['sale_date'] = data['YrSold'].map(str) + '-' + data['MoSold'].map(str)

one_hot_fea = [
    'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',  'Neighborhood', 
    'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
    'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'Heating', 'CentralAir', 'Electrical', 'Functional',
    'GarageType', 'GarageFinish', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'sale_date']

other_fea = [
    'LotShape', 'YearBuilt', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'YrSold', 'MoSold']

drop_list = []

drop_list += one_hot_fea + other_fea

for fea in one_hot_fea:
    data_dummy = pd.get_dummies(data[fea], prefix=fea)
    data = pd.concat((data, data_dummy), axis=1)

data['LotShape_ordinal'] = data['LotShape'].map({'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3})
data['ExterQual_ordinal'] = data['ExterQual'].map({'EX': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
data['ExterCond_ordinal'] = data['ExterCond'].map({'EX': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
data['BsmtQual_ordinal'] = data['BsmtQual'].map({'EX': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
data['BsmtCond_ordinal'] = data['BsmtCond'].map({'EX': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
data['BsmtExposure_ordinal'] = data['BsmtExposure'].map({'Gd': 5, 'Av': 4, 'Mn': 3, 'No': 2, 'NA': 1})
data['BsmtFinType1_ordinal'] = data['BsmtFinType1'].map({'GLQ': 5, 'ALQ': 4, 'BLQ': 3, 'Rec': 2, 'LwQ': 1, 'Unf': 0, 'NA': -1})
data['BsmtFinType2_ordinal'] = data['BsmtFinType2'].map({'GLQ': 5, 'ALQ': 4, 'BLQ': 3, 'Rec': 2, 'LwQ': 1, 'Unf': 0, 'NA': -1})
data['HeatingQC_ordinal'] = data['HeatingQC'].map({'EX': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
data['KitchenQual_ordinal'] = data['KitchenQual'].map({'EX': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
data['FireplaceQu_ordinal'] = data['FireplaceQu'].map({'EX': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
data['GarageQual_ordinal'] = data['GarageQual'].map({'EX': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
data['GarageCond_ordinal'] = data['GarageCond'].map({'EX': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
data['PavedDrive_ordinal'] = data['PavedDrive'].map({'Y': 2, 'P': 1, 'N': 0})
data['PoolQC_ordinal'] = data['PoolQC'].map({'EX': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
data['age'] = 2018 - data['YearBuilt']
data['Garage_age'] = 2018 - data['GarageYrBlt']

data = data.drop(drop_list, axis=1)

y_train = data.loc[data['data_type'] == 'train']['logSalePrice']
x_train = data.loc[data['data_type'] == 'train']
x_train = x_train.drop(['Id', 'logSalePrice', 'data_type'], axis=1)
x_test = data.loc[data['data_type'] == 'test'].drop(['Id', 'logSalePrice', 'data_type'], axis=1)

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

###############################################################################
# Approach 1: using xgboost.cv

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





###############################################################################
# Approach 2: using RandomizedSearchCV
skf = KFold(n_splits=5, shuffle = True)
bst = xgb.XGBClassifier(n_estimators=10000, objective='reg:linear', silent=1)
random_search = RandomizedSearchCV(bst, param_distributions=random_params, n_iter=1, scoring='neg_mean_squared_error', cv=skf.split(x_train, y_train), verbose=3, n_jobs=1)
random_search.fit(x_train, y_train, early_stopping_rounds=100)
results = pd.DataFrame(random_search.cv_results_)



num_round = 1000



