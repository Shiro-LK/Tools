# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 18:57:27 2018

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
from CrossValidation import CrossVal, load_train,  RMSLE, add_Features


class Bagging():
    def __init__(self, models=[], params=[],n=10):
        """
            model is a list of model we want to train in the bagging (choose randomly) and params are their parameters
        """
        self.n = n
        self.model_base = models
        self.params = params
        self.models = []
        if len(params) != len(self.model_base) or len(params)==0 or len(self.model_base)==0:
            raise('Error in params or model_base parameters. Check there are the same number of model and parameter')
        
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
    
    
    def train(self, X_data, y_data, X_test, number_iteration=1000, early_stopping_rounds=20, eval_metric=RMSLE):
        """"
            train a model lightgbm or/and XGBoost. Put params
        """
        
        # saves predictions 
        test_preds = np.zeros(X_test.shape[0])
        max_ = X_data.shape[0]
        number_of_model = len(self.model_base)
        idx_model = 0
        validation_results = []
        dtest = xgb.DMatrix(X_test)
        for cpt in range(self.n):
            train_idx = np.random.choice(max_, max_)
            val_idx = np.setdiff1d([i for i in range(max_)], train_idx)
            if number_of_model > 1:
                idx_model = np.random.randint(number_of_model)
                
            if self.model_base[idx_model]  == lgb:
                print('### LGB model ###')
                dtrain = lgb.Dataset(data=X_data, label=np.log1p(y_data), free_raw_data=False)
                dtrain.construct()
                
                # hard code parameters, change parameters for bagging
                self.params[idx_model]['num_leaves'] = np.random.randint(50, 90)
                self.params[idx_model]['subsample'] = np.random.uniform(0.6,1.0)
                
                model = self.model_base[idx_model].train(params=self.params[idx_model], train_set=dtrain.subset(train_idx),
                                                  num_boost_round=number_iteration, valid_sets=dtrain.subset(val_idx),
                                                  early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
                preds = model.predict(X_data[val_idx])
                results_preds = eval_metric(preds, np.log1p(y_data[val_idx]))
                print('Evaluation validation lgb iteration {} : {}'.format(cpt ,results_preds))
                validation_results.append(results_preds[1])
                test_preds += model.predict(X_test)
                self.models.append(model)
                
                
                
            elif self.model_base[idx_model]  == xgb:
                print('### XGB model ###')
                
                # parameters random hard code
                self.params[idx_model]['max_depth'] = np.random.randint(5, 105)
                self.params[idx_model]['subsample'] = np.random.uniform(0.6,1.0)
                self.params[idx_model]['min_child_weight'] = np.random.randint(50, 90)
            
                dtrain =  xgb.DMatrix(X_data[train_idx], label=np.log1p(y_data[train_idx])) # datas.slice(train_idx)
                dval = xgb.DMatrix(X_data[val_idx], label=np.log1p(y_data[val_idx]))
                model = self.model_base[idx_model].train(params=self.params[idx_model], dtrain=dtrain,
                                                      num_boost_round=number_iteration, evals=[(dval, 'eval')],
                                                      early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
                preds = model.predict(dval)
                results_preds = eval_metric(preds, np.log1p(y_data[val_idx]))
                print('Evaluation validation xgb iteration {} : {}'.format(cpt, results_preds))
                validation_results.append(results_preds[1])
                test_preds += model.predict(dtest)
                self.models.append(model)
                
            print('Average Evaluation Bagging intermediate:', np.mean(np.asarray(validation_results) ) )
        return test_preds/self.n
    
    def fit(self, X_data, y_data, X_test, eval_metric):
        
        
        # saves predictions 
        test_preds = np.zeros(X_test.shape[0])
        
        max_ = X_data.shape[0]
        number_of_model = len(self.model_base)
        idx_model = 0
        validation_results = []
        
        
        for cpt in range(self.n):
            train_idx = np.random.choice(max_, max_)
            val_idx = np.setdiff1d([i for i in range(max_)], train_idx)
            if number_of_model > 1:
                idx_model = np.random.randint(number_of_model)
                
            model = copy.deepcopy(self.model_base[idx_model])
            model.fit(X_data[train_idx], np.log1p(y_data[train_idx]))
            
            preds = model.predict(X_data[val_idx])
            results_preds = eval_metric(preds, np.log1p(y_data[val_idx]))
            print('Evaluation validation lgb iteration {} : {}'.format(cpt), results_preds)
            validation_results.append(results_preds)
            test_preds += model.predict(X_test)
            self.models.append(model)
        
            print('Average Evaluation Bagging intermediate:', np.mean(np.asarray(results_preds) ) )
        return test_preds/self.n
    
    
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
    bagging = Bagging(models=[lgb], params=[lgb_params], n=30)
    #
    #print(dtrain.data.shape, dtrain.label.shape)
    val_preds = bagging.train(X_data=X_train.values, y_data=y_train.values, X_test=X_val.values)
    print('Validation results !', RMSLE(val_preds, np.log1p(y_val)))
    bagging.save_models('lightgbm_bagging30.pkl')
    
def mainXGB():
    train_csv, y_train_csv = load_train('train.csv')  
    X_train_post = add_Features(train_csv, ['SumZeros', 'SumValues', 'OtherAgg']) # 
    #print(X_train_post)
    X_train, X_val, y_train, y_val = train_test_split(X_train_post, y_train_csv, test_size=0.1, random_state=10)
    
    xgb_params = {'max_depth':100, 'random_state':10, 'n_estimators':1000, 'learning_rate':0.1, 'silent':False,
                  'booster':'gbtree', 'min_child_weight':57, 'gamma':1.45, 'alpha':0.0,
                   'subsample':0.67, 'colsample_bytree':0.054, 'colsample_bylevel':0.5}
    #xgb_params = {'max_depth':100, 'random_state':10, 'n_estimators':1000, 'learning_rate':0.1, 'silent':False,
     #             'booster':'gbtree', 'min_child_weight':57, 'gamma':1.45, 'alpha':0.0,
      #             'subsample':0.67, 'colsample_bytree':0.054, 'colsample_bylevel':0.5}
    bagging = Bagging(models=[xgb], params=[xgb_params], n=100)
    #
    #print(dtrain.data.shape, dtrain.label.shape)
    val_preds = bagging.train(X_data=X_train.values, y_data=y_train.values, X_test=X_val.values)
    print('Validation results !', RMSLE(val_preds, np.log1p(y_val)))
    bagging.save_models('xgb_bagging10.pkl')
    
mainXGB()