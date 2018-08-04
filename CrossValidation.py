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
  
class CrossVal():
    def __init__(self, model, name_model, n_split=10):
        self.n_split = n_split
        self.model_base = model
        self.name_model = name_model
        self.models = []
        
    def predict(self, data, method='average'):
        """
            predict the data from the different model using an average method.
            For getting probability, use predict_proba function.
        """
        if len(self.models) > 0:
            if method == 'average':
                
                    # special format for xgb
                    if self.name_model == 'xgb':
                        ddata = xgb.DMatrix(data)
                    for i, model in enumerate(self.models):
                        if self.name_model == 'xgb':
                            if i == 0:
                                prediction = model.predict(ddata)/len(self.models)
                            else:
                                prediction += model.predict(ddata)/len(self.models)
                        else:

                            if i == 0:
                                prediction = model.predict(data)/len(self.models)
                            else:
                                prediction += model.predict(data)/len(self.models)
        return prediction
    
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
                        if i == 0:
                            prediction = model.predict_proba(data)/len(self.models)
                        else:
                            prediction += model.predict_proba(data)/len(self.models)
            return prediction
    def save_models(self, name='save.pkl'):
        """
            name : name of the models' saved
            model base is put to None because it is impossible to save lgb and xgb library.
        """
        self.model_base=None
        with open(name, 'wb') as f:
            pickle.dump(self, f)
    
    
    def fit(self, X_data, y_data, X_test, y_test=None, classification=False, binary=False, eval_metric=RMSE, params=None, number_iteration=1000, early_stopping_rounds=20):
        '''
            predicts : None or 'proba'. Usefull for certain model where we need to use predict_proba to get probabilities
            classification : if true, fix binary variable. If logistic regression, binary equal True, else False.
            params : use only for xgb and lgb model
            X_data, y_data : data which will be decomposed in train and val for the cross validation
            eval metric : metric use to computed the performance of the validation/test. It can be a custom or sklearn metric.
                          Be cautious, depending of the output of your model you will need probably to create your own metric.
                          For the classification, the model output the probability for the performance calculus. 
        '''
        
        folds = KFold(n_splits=self.n_split)
        
        # saves predictions 
        if classification and binary==False:
            test_preds = np.zeros((X_test.shape[0], len(np.unique(y_data))))
            val_preds = np.zeros((X_data.shape[0], len(np.unique(y_data))))
        else:
            test_preds = np.zeros(X_test.shape[0])
            val_preds = np.zeros(X_data.shape[0])
        if params is not None:
            if self.model_base == lgb and self.name_model=="lgb":
                print('### LGB model ###')
                dtrain = lgb.Dataset(data=X_data, label=y_data, free_raw_data=False)
                dtrain.construct()
                for train_idx, val_idx in folds.split(X_data):
                    
                    model = self.model_base.train(params=params, train_set=dtrain.subset(train_idx),
                                                      num_boost_round=number_iteration, valid_sets=dtrain.subset(val_idx),
                                                      early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
                    preds = model.predict(X_data[val_idx])
                    print('Evaluation kfold : ', eval_metric(y_data[val_idx], preds))
                    val_preds[val_idx] = preds
                    test_preds += model.predict(X_test)/self.n_split
                    self.models.append(model)
                    
                #print('Average Evaluation k-fold :', eval_metric(val_preds, y_data))
            elif self.model_base == xgb and self.name_model=="xgb":
                print('### XGB model ###')
                dtest = xgb.DMatrix(X_test) # xgboost
                #datas = xgb.DMatrix(X_data, label=np.log1p(y_data))
                for train_idx, val_idx in folds.split(X_data):
                    dtrain =  xgb.DMatrix(X_data[train_idx], label=y_data[train_idx]) # datas.slice(train_idx)
                    dval = xgb.DMatrix(X_data[val_idx], label=y_data[val_idx])
                    
                    model = self.model_base.train(params=params, dtrain=dtrain,
                                                      num_boost_round=number_iteration, evals=[(dval, 'eval')],
                                                      early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
                    preds = model.predict(dval)
                    print(y_data[val_idx].shape, preds.shape)
                    print('Evaluation kfold : ', eval_metric(y_data[val_idx], preds))
                    val_preds[val_idx] = preds
                    test_preds += model.predict(dtest)/self.n_split
                    self.models.append(model)

                #print('Average Evaluation k-fold :', eval_metric(val_preds, y_data))
            
            
        else:
            if self.name_model == 'rf' or self.name_model =="adaboost" or self.name_model =='svm':
                for train_idx, val_idx in folds.split(X_data):
                    model = copy.deepcopy(self.model_base)
                    model.fit(X_data[train_idx], y_data[train_idx])
                    
                    if classification:
                        preds = model.predict_proba(X_data[val_idx])
                    else:
                        preds = model.predict(X_data[val_idx])
                    print('Evaluation kfold : ', eval_metric(y_data[val_idx], preds))
                    val_preds[val_idx] = preds
                    if classification:
                        test_preds += model.predict_proba(X_test)/self.n_split
                    else:
                        test_preds += model.predict(X_test)/self.n_split
                    self.models.append(model)
                   
        print('Average Evaluation k-fold :', eval_metric(y_data, val_preds))
        if y_test is not None:
            print('Evaluation Test : ', eval_metric(y_test, test_preds))
                
                
        return val_preds, test_preds
    

