# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 12:14:16 2018

@author: shiro
"""
import pickle
import numpy as np


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import skew
import time
import gc
import copy
gc.enable()

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
import numpy as np
import pandas as pd
import copy
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import lightgbm as lgb
np.random.seed(10)
from sklearn.metrics import mean_squared_error

def load_train(filename='train.csv'):
  train_csv = pd.read_csv(filename)
  y_train_csv = train_csv["target"]
  train_csv = train_csv.drop(["ID", "target"], axis=1)
  return train_csv, y_train_csv

def add_Features(train, features=['SumZeros', 'SumValues']):
    Train = copy.deepcopy(train)
    
    flist = [x for x in train.columns if not x in ['ID','target']]
    #print(len(flist))
    Train.replace(0, np.nan, inplace=True)
    Train.insert(0, 'SumNaN', Train[flist].isnull().sum(axis=1))
    #if 'SumZeros' in features:
        #Train.insert(0, 'SumZeros', (train[flist] == 0).astype(int).sum(axis=1))
        #print(Train[flist].isnull().sum(axis=1))
        
    if 'SumValues' in features:
        Train.insert(1, 'SumValues', Train[flist].sum(axis=1))
        
    #flist = [x for x in train.columns if not x in ['ID','target']]
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
        #Train['the_sum'] = Train[flist].sum(axis=1)

    return Train
def RMSLE(y_pred, y_true):
  #root ( 1/N sum [( log 1+y_true - log 1°y_pred)²])
  if y_true.shape == y_pred.shape:
    N = len(y_pred)
  else:
    raise("Error dimension prediction and true value!")
  log_true = y_true
  log_pred = y_pred
  logval = np.square(log_true - log_pred)
  loss = np.sqrt(logval.sum()/N)
  return ('RMSLE', loss)
class CrossVal():
    def __init__(self, model, n_split=10):
        self.n_split = n_split
        self.model_base = model
        self.models = []
        
    def predict(self, data, method='average'):
        if len(self.models) > 0:
            if method == 'average':
                for i, model in enumerate(self.models):
                    if i == 0:
                        prediction = model.predict(data)/len(self.models)
                    else:
                        prediction += model.predict(data)/len(self.models)
        return prediction
    def save_models(self, name='save.pkl'):
        with open(name, 'wb') as f:
            pickle.dump(self.models, f)
    
    
    def train(self, X_data, y_data, X_test, params, number_iteration=1000, early_stopping_rounds=20, eval_metric=RMSLE):
        """"
            train a model lightgbm or XGBoost. Put params
        """
        folds = KFold(n_splits=self.n_split)
        # saves predictions 
        test_preds = np.zeros(X_test.shape[0])
        val_preds = np.zeros(X_data.shape[0])
        
        if self.model_base == lgb:
            print('### LGB model ###')
            dtrain = lgb.Dataset(data=X_data, label=np.log1p(y_data), free_raw_data=False)
            dtrain.construct()
            for train_idx, val_idx in folds.split(X_data):
                
                model = self.model_base.train(params=params, train_set=dtrain.subset(train_idx),
                                                  num_boost_round=number_iteration, valid_sets=dtrain.subset(val_idx),
                                                  early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
                preds = model.predict(X_data[val_idx])
                print('Evaluation kfold : ', eval_metric(preds, np.log1p(y_data[val_idx])))#dtrain.label.iloc[val_idx]))
                val_preds[val_idx] = preds
                test_preds += model.predict(X_test)/self.n_split
                self.models.append(model)
            
            print('Average Evaluation k-fold :', eval_metric(val_preds, np.log1p(y_data)))
        elif self.model_base == xgb:
            print('### XGB model ###')
            dtest = xgb.DMatrix(X_test) # xgboost
            #datas = xgb.DMatrix(X_data, label=np.log1p(y_data))
            for train_idx, val_idx in folds.split(X_data):
                dtrain =  xgb.DMatrix(X_data[train_idx], label=np.log1p(y_data[train_idx])) # datas.slice(train_idx)
                dval = xgb.DMatrix(X_data[val_idx], label=np.log1p(y_data[val_idx]))
                
                model = self.model_base.train(params=params, dtrain=dtrain,
                                                  num_boost_round=number_iteration, evals=[(dval, 'eval')],
                                                  early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
                preds = model.predict(dval)
                print('Evaluation kfold : ', eval_metric(preds, np.log1p(y_data[val_idx])))
                val_preds[val_idx] = preds
                test_preds += model.predict(dtest)/self.n_split
                self.models.append(model)
            
            print('Average Evaluation k-fold :', eval_metric(val_preds, np.log1p(y_data)))
        return val_preds, test_preds
    
    def fit(self, X_data, y_data, X_test, eval_metric):
        folds = KFold(n_splits=self.n_split)
        
        # saves predictions 
        test_preds = np.zeros(X_test.shape[0])
        val_preds = np.zeros(X_data.shape[0])
        
        for train_idx, val_idx in folds.split(X_data):
            model = copy.deepcopy(self.model_base)
            model.fit(X_data[train_idx], np.log1p(y_data[train_idx]))
            preds = model.predict(X_data[val_idx])
            print('Evaluation kfold : ', eval_metric(preds, y_data[val_idx]))
            val_preds[val_idx] = preds
            test_preds += model.predict(X_test)/self.n_split
            self.models.append(model)
        
        print('Average Evaluation k-fold :', eval_metric(val_preds, y_data))
        return val_preds, test_preds
    


def main():    
    train_csv, y_train_csv = load_train('train.csv')  
    X_train_post = add_Features(train_csv, ['SumZeros', 'SumValues', 'OtherAgg']) # 
    #print(X_train_post)
    X_train, X_val, y_train, y_val = train_test_split(X_train_post, y_train_csv, test_size=0.1, random_state=10)

    ## lgb 
    
    
    lgb_params = {
        'objective': 'regression',
        'num_leaves': 58,
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
        'metric': 'rmse',

    }
    cv = CrossVal(lgb, n_split=30)
    #
    #print(dtrain.data.shape, dtrain.label.shape)
    crossval_preds, val_preds = cv.train(X_data=X_train.values, y_data=y_train.values, X_test=X_val.values, params=lgb_params)
    print('Validation results !', RMSLE(val_preds, np.log1p(y_val)))
    cv.save_models('lightgbm_cv30.pkl')

def mainXGB():
    train_csv, y_train_csv = load_train('train.csv')  
    X_train_post = add_Features(train_csv, ['SumZeros', 'SumValues', 'OtherAgg']) # 
    #print(X_train_post)
    X_train, X_val, y_train, y_val = train_test_split(X_train_post, y_train_csv, test_size=0.1, random_state=10)
    
    xgb_params = {'max_depth':100, 'random_state':10, 'n_estimators':1000, 'learning_rate':0.1, 'silent':False,
                  'booster':'gbtree', 'min_child_weight':57, 'gamma':1.45, 'alpha':0.0,
                   'subsample':0.67, 'colsample_bytree':0.054, 'colsample_bylevel':0.5}
    
    cv = CrossVal(xgb, n_split=10)
    #
    #print(dtrain.data.shape, dtrain.label.shape)
    crossval_preds, val_preds = cv.train(X_data=X_train.values, y_data=y_train.values, X_test=X_val.values, params=xgb_params)
    print('Validation results !', RMSLE(val_preds, np.log1p(y_val)))
    cv.save_models('xgb_cv30.pkl')
    
#mainXGB()