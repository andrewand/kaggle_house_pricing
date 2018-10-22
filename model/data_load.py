#!/usr/bin/env python

import numpy as np
import pandas as pd
import xgboost as xgb
import math


def load_raw_data():
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    test['SalePrice'] = np.nan
    train['data_type'] = 'train'
    test['data_type'] = 'test'
    data = pd.concat([train, test])

    data['logSalePrice'] = data['SalePrice'].apply(math.log)
    data = data.drop('SalePrice', axis=1)
    data['sale_date'] = data['YrSold'].map(str) + '-' + data['MoSold'].map(str)

    return data


def data_pre_processing(data):
    one_hot_fea = [
        'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour', 'Utilities',
        'LotConfig', 'LandSlope',  'Neighborhood', 'Condition1', 'Condition2',
        'BldgType', 'HouseStyle', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'Heating',
        'CentralAir', 'Electrical', 'Functional','GarageType', 'GarageFinish',
        'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'sale_date']

    other_fea = [
        'LotShape', 'YearBuilt', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual',
        'FireplaceQu', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'YrSold', 'MoSold']

    drop_list = one_hot_fea + other_fea

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

    return dtrain, dtest
