# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 15:03:59 2018

@author: shiro
"""
from sklearn import datasets
from CrossValidation import RMSE, CrossValClassifier,CrossValRegressor
import pandas as pd
import numpy as np
import copy
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
import pickle
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier, AdaBoostRegressor
from xgboost.sklearn import XGBRegressor, XGBClassifier
from lightgbm.sklearn import LGBMRegressor, LGBMClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
def load_train(filename='train.csv'):
  train_csv = pd.read_csv(filename)
  y_train_csv = train_csv["target"]
  train_csv = train_csv.drop(["ID", "target"], axis=1)
  return train_csv, y_train_csv

def add_Features(train, features=['SumZeros', 'SumValues'], RF=False):
    Train = copy.deepcopy(train)
    
    flist = [x for x in train.columns if not x in ['ID','target']]
    
    Train.replace(0, np.nan, inplace=True)
    Train.insert(0, 'SumNaN', Train[flist].isnull().sum(axis=1))
        
    if 'SumValues' in features:
        Train.insert(1, 'SumValues', Train[flist].sum(axis=1))
        
    if 'OtherAgg' in features:
        #print(Train[flist].mode(axis=1))
        #Train.add(Train[flist].mode(axis=1))
        Train['Mean']   = Train[flist].mean(axis=1)
        Train['Median'] = Train[flist].median(axis=1)
        Train['Max']    = Train[flist].max(axis=1)
        Train['Min']    = Train[flist].min(axis=1)
        Train['Var']    = Train[flist].var(axis=1)
        Train['Std']    = Train[flist].std(axis=1)
        Train['Kurtosis'] = Train[flist].kurtosis(axis=1)
        #Train['NoNan'] = Train[flist].notnull().sum(axis=1)
        Train['Mad'] = Train[flist].mad(axis=1)
        #Train['the_sum'] = Train[flist].sum(axis=1)
    if RF:
        Train.replace(np.nan, 0, inplace=True)
    
    


    return Train


### regression
train_csv, y_train_csv = load_train(filename='train.csv')

X_train_post = add_Features(train_csv, ['SumZeros', 'SumValues', 'OtherAgg'])  

cols = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1',
           '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9',
           'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b', 
           '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212',  '66ace2992',
           'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd', 
           '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a',
           '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2',  '0572565c2',
           '190db8488',  'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98'] 

others = [ 'SumValues', 'Mean', 'Median',  'Max', "Min",  'SumNaN', 'Kurtosis', 'Var',  'Std', 'Mad'] 
    
    
    #print(X_train_post[cols+others].shape, train_leak.shape)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_post, y_train_csv, test_size=0.1, random_state=10)

### CatBoostClassifier
catboost_params ={
    'loss_function': 'RMSE',
    'learning_rate': 0.5,
    'iterations': 5000,
    'depth': 20,
    #'class_weights': [1, 2],
    'bootstrap_type': 'Bernoulli',
    'random_seed': 3,
    'verbose': False,
    'eval_metric':'RMSE',
    #'device_type':'CPU',
}

single_model = CatBoostRegressor(**catboost_params)
single_model.fit(X_train.values, np.log1p(Y_train.values), eval_set=[(X_val.values, np.log1p(Y_val.values))], use_best_model=True, early_stopping_rounds=15, verbose=True)
preds_val = single_model.predict(X_val.values)
print('Evaluation CatBoost Single Model :', RMSE(np.log1p(Y_val.values), preds_val))

model = CrossValRegressor(
         CatBoostRegressor(**catboost_params), n_split=3)
model.fit(X_train.values, np.log1p(Y_train.values), X_val.values, np.log1p(Y_val.values), eval_metric=RMSE)
model.save_models('test_catboostcv.pkl')
del model

with open('test_catboostcv.pkl', 'rb') as f:
    model = pickle.load(f)
preds = model.predict(X_val.values)
print('Evaluation CatBoost CV :', RMSE(np.log1p(Y_val.values), preds))



## XGB regression 
xgb_params = { 'max_depth':100, 'random_state':10, 'n_estimators':1500, 'learning_rate':0.1, 'silent':False,
                  'booster':'gbtree', 'min_child_weight':57, 'gamma':1.45, 'alpha':0.0,
                   'subsample':0.67, 'colsample_bytree':0.054, 'colsample_bylevel':0.5, 'metric': 'rmse'}
model = CrossValRegressor(XGBRegressor(**xgb_params),  n_split=10)
_, _ = model.fit(X_train.values, np.log1p(Y_train.values), X_val.values, np.log1p(Y_val.values), eval_metric=RMSE)
model.save_models('test_regressionxgbcv.pkl')
del model

with open('test_regressionxgbcv.pkl', 'rb') as f:
    model = pickle.load(f)
preds = model.predict(X_val.values)
print(RMSE(np.log1p(Y_val.values), preds))
model_single = XGBRegressor(**xgb_params)
model_single.fit(X_train, np.log1p(Y_train), eval_set=[(X_val, np.log1p(Y_val))], verbose=False,
                 early_stopping_rounds=50)
preds_single = model_single.predict(X_val)
print("Model single", RMSE(np.log1p(Y_val.values), preds_single))

## lgb regression
lgb_params = {
        'objective': 'regression',
        'num_leaves': 58,#58
        'subsample': 0.6143,
        'colsample_bytree': 0.6453,
        'min_split_gain': np.power(10, -2.5988),
        'reg_alpha': np.power(10, -2.2887),
        'reg_lambda': np.power(10, 1.7570),
        'min_child_weight': np.power(10, -0.1477),
        'verbose': 0,
        'seed': 3,
        'boosting_type': 'gbdt',
        'max_depth': -1,
        'learning_rate': 0.05,
        'metric': 'rmse',}

model = CrossValRegressor(LGBMRegressor(**lgb_params), n_split=10)
model.fit(X_train.values, np.log1p(Y_train.values), X_val.values, np.log1p(Y_val.values), eval_metric=RMSE)
model.save_models('test_regressionlgbcv.pkl')
del model

with open('test_regressionlgbcv.pkl', 'rb') as f:
    model = pickle.load(f)
preds = model.predict(X_val.values)

print("Model CV", RMSE(np.log1p(Y_val.values), preds))

model_single = LGBMRegressor(**lgb_params)
model_single.fit(X_train, np.log1p(Y_train), eval_set=[(X_val, np.log1p(Y_val))], verbose=False,
                 early_stopping_rounds=50)
preds_single = model_single.predict(X_val)
print("Model single", RMSE(np.log1p(Y_val.values), preds_single))


## rf REGRESSION

model = CrossValRegressor(
        RandomForestRegressor(bootstrap=True, max_depth=10, random_state=10, max_features=0.4, 
                                            min_samples_leaf=4, min_samples_split=2, n_estimators=50, n_jobs=-1)
                                        , n_split=10)
model.fit(np.nan_to_num(X_train.values), np.log1p(Y_train.values), np.nan_to_num(X_val.values), np.log1p(Y_val.values), eval_metric=RMSE)
model.save_models('test_regressionrfcv.pkl')
del model

with open('test_regressionrfcv.pkl', 'rb') as f:
    model = pickle.load(f)
preds = model.predict(np.nan_to_num(X_val.values))

print(RMSE(np.log1p(Y_val.values), preds))

model_single = RandomForestRegressor(bootstrap=True, max_depth=10, random_state=10, max_features=0.4, 
                                            min_samples_leaf=4, min_samples_split=2, n_estimators=50, n_jobs=-1)
model_single.fit(np.nan_to_num(X_train), np.log1p(Y_train))
preds_single = model_single.predict(np.nan_to_num(X_val))
print("Model single", RMSE(np.log1p(Y_val.values), preds_single))

## adaboost regression

model = CrossValRegressor(
         AdaBoostRegressor(DecisionTreeRegressor(max_depth=15), random_state=10, n_estimators=25)
                                        , n_split=10)
model.fit(np.nan_to_num(X_train.values), np.log1p(Y_train.values), np.nan_to_num(X_val.values), eval_metric=RMSE)
model.save_models('test_adaboostcv.pkl')
del model

with open('test_adaboostcv.pkl', 'rb') as f:
    model = pickle.load(f)
preds = model.predict(np.nan_to_num(X_val.values))

print(RMSE(np.log1p(Y_val.values), preds))

model_single = AdaBoostRegressor(DecisionTreeRegressor(max_depth=15), random_state=10, n_estimators=25)
model_single.fit(np.nan_to_num(X_train), np.log1p(Y_train))
preds_single = model_single.predict(np.nan_to_num(X_val))
print("Model single", RMSE(np.log1p(Y_val.values), preds_single))