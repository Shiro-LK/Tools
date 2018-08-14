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
from xgboost.sklearn import XGBRegressor, XGBClassifier
from lightgbm.sklearn import LGBMRegressor, LGBMClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
gc.enable()

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
import numpy as np
import pandas as pd
import copy
import pickle

from sklearn.model_selection import train_test_split

import lightgbm as lgb
np.random.seed(10)
from matplotlib import pyplot as plt

def RMSE(y_pred, y_true):
  #root ( 1/N sum [( log 1+y_true - log 1°y_pred)²])
  if y_true.shape == y_pred.shape:
    N = len(y_pred)
  else:
    raise("Error dimension prediction and true value!")
  log_true = y_true
  log_pred = y_pred
  logval = np.square(log_true - log_pred)
  loss = np.sqrt(logval.sum()/N)
  return ('RMSE', loss)
  
class CrossValClassifier():
    def __init__(self, model, n_split=10):
        self.n_split = n_split
        self.model_base = model
        self.models = []
        try:
            self.xgb_type = type(XGBClassifier())
        except:
            print('XGBoost not installed')
        try:
            self.lgbm_type = type(LGBMClassifier())
        except:
            print('LightGBM not installed')
        try:
            self.catboost_type = type(CatBoostClassifier())
        except:
            print('CatBoost not installed')
        
    def predict(self, data, method='average'):
        """
            predict the data from the different model using an average method.
            For getting probability, use predict_proba function.
        """
        if len(self.models) > 0:
            if method == 'average':

                    for i, model in enumerate(self.models):
                        if type(model) == self.xgb_type:
                            if i == 0:
                                prediction = model.predict(data, ntree_limit=model.best_iteration+1)
                            else:
                                prediction += model.predict(data, ntree_limit=model.best_iteration+1)
                        else:
                            if type(model) != self.lgbm_type:
                                if type(model) != self.catboost_type:
                                    data = np.nan_to_num(data.astype('float32'))
                                else:
                                    data = np.nan_to_num(data)
                            if i == 0:
                                prediction = model.predict(data)
                            else:
                                prediction += model.predict(data)
        return prediction/len(self.models)
    
    def predict_proba(self, data, method='average'):
        """
            Function use to get the probability for classification task.
            Excepted for lightgbm and xgboost which return by default the probability with the predict function
            method : average the prediction of each model of the cross validation
            data : input to be predicted
        """
        if len(self.models) > 0:
            if method == 'average':

                    for i, model in enumerate(self.models):
                        if type(model) == self.xgb_type:
                            if i == 0:
                                if self.binary:
                                    prediction = model.predict_proba(data, ntree_limit=model.best_iteration+1)[:,1]
                                else:
                                    prediction = model.predict_proba(data, ntree_limit=model.best_iteration+1)
                            else:
                                if self.binary:
                                    prediction += model.predict_proba(data, ntree_limit=model.best_iteration+1)[:,1]
                                else:
                                    prediction += model.predict_proba(data, ntree_limit=model.best_iteration+1)
                        else:
                            if type(model) != self.lgbm_type:
                                if type(model) != self.catboost_type:
                                    data = np.nan_to_num(data.astype('float32'))
                                else:
                                    data = np.nan_to_num(data)
                                    
                            if i == 0:
                                if self.binary:
                                    prediction = model.predict_proba(data)[:,1]
                                else:
                                    prediction = model.predict_proba(data)
                            else:
                                if self.binary:
                                    prediction += model.predict_proba(data)[:,1]
                                else:
                                    prediction += model.predict_proba(data)
        return prediction/len(self.models)
    
    def save_models(self, name='save.pkl'):
        """
            name : name of the models' saved
            model base is put to None because it is impossible to save lgb and xgb library.
        """
        self.model_base=None
        with open(name, 'wb') as f:
            pickle.dump(self, f)
    
    
    def fit(self, X, y, X_test=None, y_test=None, eval_metric=None, number_iteration=1000, early_stopping_rounds=50, verbose=False, custom_eval_metric=None, verbose_eval=True, binary=False):
        '''
            predicts : None or 'proba'. Usefull for certain model where we need to use predict_proba to get probabilities
            classification : if true, fix binary variable. If logistic regression, binary equal True, else False.
            params : use only for xgb and lgb model
            X_data, y_data : data which will be decomposed in train and val for the cross validation
            eval metric : metric use to computed the performance of the validation/test. It can be a custom or sklearn metric.
                          Be cautious, depending of the output of your model you will need probably to create your own metric.
                          For the classification, the model output the probability for the performance calculus. 
            custom_eval_metric: custom metric used during the validation process for XGBoost or LightGBM (with the early stopping parameters)
            early stopping rounds : parameters used for XGB. Else use it when you initiliaze the model in the dic
            binary : use for binary classification
        '''
        
        folds = KFold(n_splits=self.n_split)
        self.binary = binary
        # saves predictions 
        if binary:
            if X_test is not None:
                test_preds = np.zeros((X_test.shape[0], ))
            val_preds = np.zeros((X.shape[0],)) # number of class 
        else:
            if X_test is not None:
                test_preds = np.zeros((X_test.shape[0], len(np.unique(y))))
            val_preds = np.zeros((X.shape[0],len(np.unique(y)))) # number of class
        

        if type(self.model_base) == self.lgbm_type:
            print('### LGB model ###')

            for train_idx, val_idx in folds.split(X):
                
                model = copy.deepcopy(self.model_base)
                model.fit(X[train_idx], y[train_idx], eval_set=[(X[val_idx], y[val_idx])], eval_metric=custom_eval_metric,
                          verbose=verbose)
                
                if binary:
                    preds = model.predict_proba(X[val_idx])[:,1]
                else:
                    preds = model.predict_proba(X[val_idx])

                val_preds[val_idx] = preds
                if eval_metric is not None and verbose_eval:
                    print('Evaluation kfold : ', eval_metric(y[val_idx], preds))
                    
                if X_test is not None:
                    if binary:
                        test_preds += model.predict_proba(X_test)[:,1]
                    else:
                        test_preds += model.predict_proba(X_test)
                self.models.append(model)
                print(test_preds.shape)
            #print('Average Evaluation k-fold :', eval_metric(val_preds, y_data))
        elif type(self.model_base) == self.xgb_type:
            print('### XGB model ###')
           
            for train_idx, val_idx in folds.split(X):
                                
                model = copy.deepcopy(self.model_base) 
                model.fit(X[train_idx], y[train_idx], eval_set=[(X[val_idx], y[val_idx])], early_stopping_rounds=early_stopping_rounds,
                            eval_metric=custom_eval_metric, verbose=verbose)
                
                if binary:
                    preds = model.predict_proba(X[val_idx], ntree_limit=model.best_iteration+1)[:,1]
                else:
                    preds = model.predict_proba(X[val_idx], ntree_limit=model.best_iteration+1)
               
                val_preds[val_idx] = preds
                if eval_metric is not None and verbose_eval:
                    print('Evaluation kfold : ', eval_metric(y[val_idx], preds))
                
                if X_test is not None:
                    if binary:
                        test_preds += model.predict_proba(X_test, ntree_limit=model.best_iteration+1)[:,1]
                    else:
                        test_preds += model.predict_proba(X_test, ntree_limit=model.best_iteration+1)
                    
                self.models.append(model)
    
            #print('Average Evaluation k-fold :', eval_metric(val_preds, y_data))
            
        elif type(self.model_base) == self.catboost_type:
            print('### CatBoost model ###')
            X = np.nan_to_num(X)
            if X_test is not None:
                X_test = np.nan_to_num(X_test)
            for train_idx, val_idx in folds.split(X):
                                
                model = copy.deepcopy(self.model_base) 
                model.fit(X[train_idx], y[train_idx], eval_set=[(X[val_idx], y[val_idx])],
                            verbose=verbose, use_best_model=True)
                
                if binary:
                    preds = model.predict_proba(X[val_idx])[:,1]
                else:
                    preds = model.predict_proba(X[val_idx])

                val_preds[val_idx] = preds
                if eval_metric is not None and verbose_eval:
                    print('Evaluation kfold : ', eval_metric(y[val_idx], preds))
                
                if X_test is not None:
                    if binary:
                        test_preds += model.predict_proba(X_test)[:,1]
                    else:
                        test_preds += model.predict_proba(X_test)
                self.models.append(model)    
        else:
            print('### Sklearn model ###')
            X = np.nan_to_num(X.astype('float32'))
            if X_test is not None:
                X_test = np.nan_to_num(X_test.astype('float32'))
            for train_idx, val_idx in folds.split(X):
                model = copy.deepcopy(self.model_base)
                model.fit(X[train_idx], y[train_idx])
                
 
                if binary:
                    preds = model.predict_proba(X[val_idx])[:,1]
                else:
                    preds = model.predict_proba(X[val_idx])
                    
                if eval_metric is not None and verbose_eval:
                    print('Evaluation kfold : ', eval_metric(y[val_idx], preds))
                val_preds[val_idx] = preds
                
                if X_test is not None:
                    if binary:
                        test_preds += model.predict_proba(X_test)[:,1]
                    else:
                        test_preds += model.predict_proba(X_test)
                self.models.append(model)
                
        if eval_metric is not None and verbose_eval:         
            print('Average Evaluation k-fold :', eval_metric(y, val_preds))
        ### training finished
        if y_test is not None:
            print('Evaluation Test : ', eval_metric(y_test, test_preds/self.n_split))
                
        if X_test is not None:      
            return val_preds, test_preds/self.n_split
        else:
            return val_preds
    

class CrossValRegressor():
    def __init__(self, model, n_split=10):
        self.n_split = n_split
        self.model_base = model
        self.models = []
        try:
            self.xgb_type = type(XGBRegressor())
        except:
            print('XGBoost not installed')
        try:
            self.lgbm_type = type(LGBMRegressor())
        except:
            print('LightGBM not installed')
        try:
            self.catboost_type = type(CatBoostRegressor())
        except:
            print('CatBoost not installed')
        
    def predict(self, data, method='average'):
        """
            data : must be numpy array
            predict the data from the different model using an average method.
        """
        if len(self.models) > 0:
            if method == 'average':
                
                    for i, model in enumerate(self.models):
                        if type(model) == self.xgb_type:
                            if i == 0:
                                prediction = model.predict(data, ntree_limit=model.best_iteration+1)
                            else:
                                prediction += model.predict(data, ntree_limit=model.best_iteration+1)
                        else:
                            if type(model) != self.lgbm_type:
                                if type(model) == self.catboost_type:
                                     data = np.nan_to_num(data)
                                else:
                                    data = np.nan_to_num(data.astype('float32'))

                            if i == 0:
                                prediction = model.predict(data)
                            else:
                                prediction += model.predict(data)
        return prediction/len(self.models)

    def save_models(self, name='save.pkl'):
        """
            name : name of the models' saved
            model base is put to None because it is impossible to save lgb and xgb library.
        """
        self.model_base=None
        with open(name, 'wb') as f:
            pickle.dump(self, f)
    
    
    def fit(self, X, y, X_test=None, y_test=None, eval_metric=None, number_iteration=1000, early_stopping_rounds=50, verbose=False, verbose_eval=True, custom_eval_metric=None):
        '''
            params : use only for xgb and lgb model
            X_data, y_data : data which will be decomposed in train and val for the cross validation
            eval metric : metric use to computed the performance of the validation/test. It can be a custom or sklearn metric.
                          Be cautious, depending of the output of your model you will need probably to create your own metric.
                          For the classification, the model output the probability for the performance calculus. 
        '''
        
        folds = KFold(n_splits=self.n_split)
        
        # saves predictions 
        if X_test is not None:
            test_preds = np.zeros(X_test.shape[0])
        val_preds = np.zeros(X.shape[0])
        

        if type(self.model_base) == self.lgbm_type:
            print('### LGB model ###')
            
            for train_idx, val_idx in folds.split(X):
                
                model = copy.deepcopy(self.model_base)
                model.fit(X[train_idx], y[train_idx], eval_set=[(X[val_idx], y[val_idx])],
                          eval_metric=custom_eval_metric, verbose=verbose)
                preds = model.predict(X[val_idx])
                val_preds[val_idx] = preds
                if eval_metric is not None and verbose_eval:
                    print('Evaluation kfold LGBM: ', eval_metric(y[val_idx], preds))
                
                
                
                if X_test is not None:
                    test_preds += model.predict(X_test)
                self.models.append(model)
                

        elif type(self.model_base) == self.xgb_type:
            print('### XGB model ###')
            for train_idx, val_idx in folds.split(X):
                
                
                model = copy.deepcopy(self.model_base)
                model.fit(X[train_idx], y[train_idx], eval_set=[(X[val_idx], y[val_idx])],
                          early_stopping_rounds=early_stopping_rounds, eval_metric=custom_eval_metric, verbose=verbose)
                
                preds = model.predict(X[val_idx], ntree_limit=model.best_iteration+1)
                val_preds[val_idx] = preds
                if eval_metric is not None and verbose_eval:
                    print('Evaluation kfold XGB: ', eval_metric(y[val_idx], preds))
                
                
                if X_test is not None:
                    test_preds += model.predict(X_test, ntree_limit=model.best_iteration+1)
                self.models.append(model)
                
        elif type(self.model_base) == self.catboost_type:
            print('### CatBoost model ###')
           
            X = np.nan_to_num(X)
            if X_test is not None:
                X_test = np.nan_to_num(X_test)
            for train_idx, val_idx in folds.split(X):
                                
                model = copy.deepcopy(self.model_base) 
                model.fit(X[train_idx], y[train_idx], eval_set=[(X[val_idx], y[val_idx])],
                            verbose=verbose, use_best_model=True)
                
                preds = model.predict(X[val_idx])
                val_preds[val_idx] = preds
                if eval_metric is not None and verbose_eval:
                    print('Evaluation kfold : ', eval_metric(y[val_idx], preds))
                
                if X_test is not None:
                    test_preds += model.predict(X_test)
                self.models.append(model)         
            
        else:
            print('### Sklearn Model ###')
            X = np.nan_to_num(X.astype('float32'))
            if X_test is not None:
                X_test = np.nan_to_num(X_test.astype('float32'))
            for train_idx, val_idx in folds.split(X):
                model = copy.deepcopy(self.model_base)
                model.fit(X[train_idx], y[train_idx])

                preds = model.predict(X[val_idx])
                if eval_metric is not None and verbose_eval:
                    print('Evaluation kfold : ', eval_metric(y[val_idx], preds))
                val_preds[val_idx] = preds
                
                if X_test is not None:
                    test_preds += model.predict(X_test)
                self.models.append(model)
                   
        if eval_metric is not None and verbose_eval:
            print('Average Evaluation k-fold Validation:', eval_metric(y, val_preds))
        if y_test is not None and X_test is not None and eval_metric is not None:
            print('Evaluation Test : ', eval_metric(y_test, test_preds/self.n_split))
                
        if X_test is None:
            return val_preds
        else:
            return val_preds, test_preds/self.n_split
    


