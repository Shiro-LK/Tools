#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stacking Function.
"""
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from CrossValidation import CrossValClassifier, CrossValRegressor
import numpy as np
import time
import copy
import pickle
import pandas as pd
class StackingClassifierCV():
    def __init__(self, models=[], meta_models=[], custom_metric_model=None, custom_metric_meta=None, n_split=10):
        if models is None or meta_models is None:
            raise('No models or meta models initialiaze !')
        self.models = models
        self.CVmodels = []
        self.meta_models = meta_models
        self.CVmeta_models = []
        self.CVmeta_models_of = [] #train only on new features
        self.n_split=n_split
        self.only_features=False
        
        self.custom_metric_model = custom_metric_model
        if self.custom_metric_model is None:
            self.custom_metric_model = [None]*len(self.models)
            
        self.custom_metric_meta = custom_metric_meta
        if self.custom_metric_meta is None:
            self.custom_metric_meta = [None]*len(self.meta_models)
            
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
    
    def fit(self, X, y, X_test=None, y_test=None, binary=False, eval_metric=None, early_stopping_rounds=50, num_iterations=10000, verbose=False, only_features=False, save='features.csv', names=None, verbose_eval_cv = False):
        """"
            Fit method using cross validation. 
            only features : train the model only on new features else create two meta learner : one which learn 
            on new features, the second on data+new features.
        """
        
        self.binary=binary
        ### train models simple Learner
        self.only_features = only_features
        for idx in range(len(self.models)):
            cvmodel = CrossValClassifier(self.models[idx], n_split=self.n_split)
            if idx == 0:
                new_features_train, new_features_test = cvmodel.fit(X=X, y=y, X_test=X_test, y_test=y_test, eval_metric=eval_metric, 
                                                                    early_stopping_rounds=early_stopping_rounds, verbose=verbose,
                                                                    verbose_eval=verbose_eval_cv, binary=binary, custom_eval_metric=self.custom_metric_model[idx])
                # = cvmodel.predict_proba(X, method='average')
                if binary:
                    new_features_train = new_features_train.reshape((new_features_train.shape[0], 1))
                    if new_features_test is not None:
                        new_features_test = new_features_test.reshape((new_features_test.shape[0], 1))
                
            else:
                temp, temp2= cvmodel.fit(X=X, y=y, X_test=X_test, y_test=y_test, eval_metric=eval_metric, early_stopping_rounds=early_stopping_rounds, 
                            verbose=verbose, verbose_eval=verbose_eval_cv, binary=binary, custom_eval_metric=self.custom_metric_model[idx])
                 #= cvmodel.predict_proba(X, method='average')
                new_features_train = np.column_stack((new_features_train, temp))
                        
                if new_features_test is not None:
                    new_features_test = np.column_stack((new_features_test, temp2))
            self.CVmodels.append(cvmodel)
            
        if eval_metric is not None:
            print("Validation K-fold averaging : ", eval_metric(y, np.mean(new_features_train.reshape( (new_features_train.shape[0], len(self.CVmodels), -1)), axis=1)))
        
        if X_test is not None and y_test is not None:
            pred_test = new_features_test.reshape((new_features_test.shape[0], len(self.CVmodels), -1))
            pred_test = pred_test.transpose(1,0,2)
            print("Evaluation Test simple learner averaging :", eval_metric(y_test, np.mean(pred_test, axis=0)))
            #_, preds_test = self.predict_proba(X_test, False, True)
            #print(np.not_equal(pred_test, preds_test))
            #print(eval_metric(y_test, np.mean(preds_test, axis=0)))
        
        ### train meta models ###
        if names is not None:
            if len(names) != new_features_train.shape[1]:
                print('number of colums names and number of features differents, create artificial names')
            
                names = ['features_'+str(i+1) for i in range(new_features_train.shape[1]) ]
        else:
            names = ['features_'+str(i+1) for i in range(new_features_train.shape[1]) ]
            
        new_features_train = np.round(new_features_train, 4)
        if new_features_test is not None:
            new_features_test = np.round(new_features_test, 4)
            
        ftrain = pd.DataFrame(new_features_train, columns=names)
        ftrain.to_csv('train_'+save, index=False)
        ftest = pd.DataFrame(new_features_test, columns=names)
        ftest.to_csv('test_'+save, index=False)
        
        ## train meta  only on new features
        print('### Meta Learner on new features ###')
        if binary:
            preds_meta_val_of = np.zeros((len(self.meta_models), X.shape[0]))
            if X_test is not None:
                preds_meta_test_of = np.zeros((len(self.meta_models), X_test.shape[0]))
        else:
            preds_meta_val_of = np.zeros((len(self.meta_models), X.shape[0], temp.shape[-1]))
            if X_test is not None:
                preds_meta_test_of = np.zeros((len(self.meta_models), X_test.shape[0], pred_test.shape[-1]))
            
        begin = time.time()
        for idx in range(len(self.meta_models)):
            cvmetaof = CrossValClassifier(copy.deepcopy(self.meta_models[idx]), n_split=self.n_split)
            preds_val, preds_test = cvmetaof.fit(new_features_train, y, X_test=new_features_test, y_test=y_test,
                       eval_metric=eval_metric, early_stopping_rounds=early_stopping_rounds, verbose=verbose,
                       verbose_eval=verbose_eval_cv, binary=binary, custom_eval_metric=self.custom_metric_meta[idx])
            if binary:
                preds_meta_val_of[idx,:] = preds_val
            else:
                preds_meta_val_of[idx,:,:] = preds_val
                
            if X_test is not None:
                preds_meta_test_of[idx,:] = preds_test
            else:
                preds_meta_test_of[idx,:,:] = preds_test
            self.CVmeta_models_of.append(cvmetaof)
            
            
        print('End training Meta only feature:', time.time()-begin)
        if eval_metric is not None:
            print("Validation K-fold averaging Meta Only new features: ", eval_metric(y, np.mean(preds_meta_val_of, axis=0)))
        if X_test is not None and y_test is not None:
            #preds_meta_test  = preds_test.reshape((preds_test.shape[0], len(self.CVmeta_models), -1))
            #preds_meta_test = preds_meta_test.transpose(1,0,2) # (model, rowdata, probability)
            print("Evaluation Meta learner averaging only new feature:", eval_metric(y_test, np.mean(preds_meta_test_of, axis=0) ))
        
        ## train meature  on  data + new features
        if only_features == False:
            print('### Meta Learner on data + new features ###')
            new_features_train = np.column_stack((X, new_features_train))
            if new_features_test is not None:
                new_features_test = np.column_stack((X_test, new_features_test))
        
            preds_meta_val = np.zeros(preds_meta_val_of.shape)
            if X_test is not None:
                preds_meta_test = np.zeros(preds_meta_test_of.shape)
                
            print('shape new data with meta features', new_features_train.shape)        
            begin= time.time()
            for idx in range(len(self.meta_models)):
                cvmeta = CrossValClassifier(self.meta_models[idx], n_split=self.n_split)
                preds_val, preds_test = cvmeta.fit(new_features_train, y, X_test=new_features_test, y_test=y_test,
                           eval_metric=eval_metric, early_stopping_rounds=early_stopping_rounds, verbose=verbose,
                           verbose_eval=verbose_eval_cv, binary=binary, custom_eval_metric=self.custom_metric_meta[idx])
                self.CVmeta_models.append(cvmeta)
                
                if binary:
                    preds_meta_val[idx,:] = preds_val
                else:
                  
                    preds_meta_val[idx,:,:] = preds_val
                
                if X_test is not None:
                    if binary:
                        preds_meta_test[idx,:] = preds_test
                    else:
                        preds_meta_test[idx,:,:] = preds_test
                    
                    
            print('End training Meta all data :', time.time()-begin)
            
            if eval_metric is not None:
                print("Validation K-fold averaging Meta all data: ", eval_metric(y, np.mean( preds_meta_val, axis=0)))
            if X_test is not None and y_test is not None:
                #preds_meta_test  = preds_test.reshape((preds_test.shape[0], len(self.CVmeta_models), -1))
                #preds_meta_test = preds_meta_test.transpose(1,0,2) # (model, rowdata, probability)
                print("Evaluation Meta learner averaging all data:", eval_metric(y_test, np.mean(preds_meta_test, axis=0) ))
                #preds_proba_meta_test, _ = self.predict_proba(X_test, average=False, return_features=False)
                #print(np.array_equiv(preds_meta_test, preds_proba_meta_test))
                #print(preds_proba_meta_test.shape, preds_meta_test.shape)
                #print(np.not_equal(preds_meta_test, preds_proba_meta_test).sum())
    
    def save(self, name='stack.pkl'):
        self.models = None
        self.meta_models = None
        with open(name, 'wb') as f:
            pickle.dump(self, f)
        
   
    def predict(self, data, average=True, return_features=False, only_features=False):
        # predict simple model
        return self.predict_proba(data, average, return_features, only_features)
    
    def predict_proba(self, data, average=True, return_features=False, only_features=False):
        # predict simple model
        if only_features == False and self.only_features == True:
            raise('Model train only with new features, put only_features parameters as True')
        predictions_data = -1
        for idx_models in range(len(self.CVmodels)):
            if idx_models == 0:
                temp = self.CVmodels[idx_models].predict_proba(data, method='average')
                new_features_data = temp
            else:
                temp = self.CVmodels[idx_models].predict_proba(data, method='average')
                new_features_data = np.column_stack((new_features_data, temp))
        #if only_features == False:
         #   new_data = np.column_stack((data, new_features_data))
        #else:
         #   new_data = new_features_data
        
        new_features_data = np.round(new_features_data, 4)
        # predict with meta learners
        if only_features:
            # prediction only on new features
            new_data = new_features_data
            for idx_meta in range(len(self.CVmeta_models_of)):
                if idx_meta == 0:
                    temp = self.CVmeta_models_of[idx_meta].predict_proba(new_data, method='average')
                    predictions_data = np.zeros((len(self.CVmeta_models_of), *temp.shape))
                    predictions_data[idx_meta] = temp
                else:
                    temp = self.CVmeta_models_of[idx_meta].predict_proba(new_data, method='average')
                    predictions_data[idx_meta] = temp
            
            if average:
                predictions_data = predictions_data.mean(axis=0)
            
            if return_features:
                new_features_data = new_features_data.reshape((new_features_data.shape[0], len(self.CVmodels), -1 ) )
                new_features_data = new_features_data.transpose(1,0,2)
                return predictions_data, new_features_data # Meta / Simple Learniner
            else:
                return predictions_data, None
        else:
            new_data = np.column_stack((data, new_features_data))
            for idx_meta in range(len(self.CVmeta_models)):
                if idx_meta == 0:
                    temp = self.CVmeta_models[idx_meta].predict_proba(new_data, method='average')
                    predictions_data = np.zeros((len(self.CVmeta_models), *temp.shape))
                    predictions_data[idx_meta] = temp
                else:
                    temp = self.CVmeta_models[idx_meta].predict_proba(new_data, method='average')
                    predictions_data[idx_meta] = temp
            
            if average:
                predictions_data = predictions_data.mean(axis=0)
            
            if return_features:
                new_features_data = new_features_data.reshape((new_features_data.shape[0], len(self.CVmodels), -1 ) )
                new_features_data = new_features_data.transpose(1,0,2)
                return predictions_data, new_features_data # Meta / Simple Learniner
            else:
                return predictions_data, None
    
    
class StackingRegressorCV():
    def __init__(self, models=[], meta_models=[], custom_metric_model=None, custom_metric_meta=None,n_split=10):
        if models is None or meta_models is None:
            raise('No models or meta models initialiaze !')
        self.models = models
        self.CVmodels = []
        self.meta_models = meta_models
        self.CVmeta_models = []
        self.CVmeta_models_of = [] #train only on new features
        self.n_split=n_split
        self.only_features=False
        
        self.custom_metric_model = custom_metric_model
        if self.custom_metric_model is None:
            self.custom_metric_model = [None]*len(self.models)
            
        self.custom_metric_meta = custom_metric_meta
        if self.custom_metric_meta is None:
            self.custom_metric_meta = [None]*len(self.meta_models)
            
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
    
    def fit(self, X, y, X_test=None, y_test=None, eval_metric=None, early_stopping_rounds=50, num_iterations=10000, verbose=False, only_features=False, save='features.csv', names=None, verbose_eval_cv = False):
        """"
            Fit method using cross validation. 
            only features : train the model only on new features else create two meta learner : one which learn 
            on new features, the second on data+new features.
        """
        
        ### train models simple Learner
        self.only_features = only_features
        for idx in range(len(self.models)):
            cvmodel = CrossValRegressor(self.models[idx], n_split=self.n_split)
            if idx == 0:
                new_features_train, new_features_test = cvmodel.fit(X=X, y=y, X_test=X_test, y_test=y_test, eval_metric=eval_metric, 
                                                                    early_stopping_rounds=early_stopping_rounds, verbose=verbose,
                                                                    verbose_eval=verbose_eval_cv, custom_eval_metric=self.custom_metric_model[idx])
                # = cvmodel.predict_proba(X, method='average')
            else:
                temp, temp2 = cvmodel.fit(X=X, y=y, X_test=X_test, y_test=y_test, eval_metric=eval_metric, early_stopping_rounds=early_stopping_rounds, 
                            verbose=verbose, verbose_eval=verbose_eval_cv , custom_eval_metric=self.custom_metric_model[idx])
                 #= cvmodel.predict_proba(X, method='average')
                new_features_train = np.column_stack((new_features_train, temp))
                if new_features_test is not None:
                    new_features_test = np.column_stack((new_features_test, temp2))
            self.CVmodels.append(cvmodel)
            
        if eval_metric is not None:
            print("Validation K-fold averaging : ", eval_metric(y, np.mean(new_features_train.reshape( (new_features_train.shape[0], len(self.CVmodels))), axis=1)))
        
        if X_test is not None and y_test is not None:
            pred_test = new_features_test.reshape((new_features_test.shape[0], len(self.CVmodels)))
            pred_test = pred_test.transpose(1,0)
            print("Evaluation Test simple learner averaging :", eval_metric(y_test, np.mean(pred_test, axis=0)))
            #_, preds_test = self.predict_proba(X_test, False, True)
            #print(np.not_equal(pred_test, preds_test))
            #print(eval_metric(y_test, np.mean(preds_test, axis=0)))
        
        ### train meta models ###
        if names is not None:
            if len(names) != new_features_train.shape[1]:
                print('number of colums names and number of features differents, create artificial names')
            
                names = ['features_'+str(i+1) for i in range(new_features_train.shape[1]) ]
        else:
            names = ['features_'+str(i+1) for i in range(new_features_train.shape[1]) ]
            
        #new_features_train = np.round(new_features_train, 4)
        #if new_features_test is not None:
         #   new_features_test = np.round(new_features_test, 4)
            
        ftrain = pd.DataFrame(new_features_train, columns=names)
        ftrain.to_csv('train_'+save, index=False)
        ftest = pd.DataFrame(new_features_test, columns=names)
        ftest.to_csv('test_'+save, index=False)
        
        ## train meta  only on new features
        print('### Meta Learner on new features ###')
        preds_meta_val_of = np.zeros((len(self.meta_models), X.shape[0]))
        if X_test is not None:
            preds_meta_test_of = np.zeros((len(self.meta_models), X_test.shape[0]))
            
        begin = time.time()
        for idx in range(len(self.meta_models)):
            cvmetaof = CrossValRegressor(copy.deepcopy(self.meta_models[idx]), n_split=self.n_split)
            preds_val, preds_test = cvmetaof.fit(new_features_train, y, X_test=new_features_test, y_test=y_test,
                       eval_metric=eval_metric, early_stopping_rounds=early_stopping_rounds, verbose=verbose,
                       verbose_eval=verbose_eval_cv , custom_eval_metric=self.custom_metric_meta[idx])
            preds_meta_val_of[idx,:] = preds_val
            
            if X_test is not None:
                preds_meta_test_of[idx,:] = preds_test
            
            self.CVmeta_models_of.append(cvmetaof)
            
            
        print('End training Meta only feature:', time.time()-begin)
        if eval_metric is not None:
            print("Validation K-fold averaging Meta Only new features: ", eval_metric(y, np.mean(preds_meta_val_of, axis=0)))
        if X_test is not None and y_test is not None:
            #preds_meta_test  = preds_test.reshape((preds_test.shape[0], len(self.CVmeta_models), -1))
            #preds_meta_test = preds_meta_test.transpose(1,0,2) # (model, rowdata, probability)
            print("Evaluation Meta learner averaging only new feature:", eval_metric(y_test, np.mean(preds_meta_test_of, axis=0) ))
        
        ## train meature  on  data + new features
        if only_features == False:
            print('### Meta Learner on data + new features ###')
            new_features_train = np.column_stack((X, new_features_train))
            if new_features_test is not None:
                new_features_test = np.column_stack((X_test, new_features_test))
        
            preds_meta_val = np.zeros(preds_meta_val_of.shape)
            if X_test is not None:
                preds_meta_test = np.zeros(preds_meta_test_of.shape)
                
            print('shape new data with meta features', new_features_train.shape)        
            begin= time.time()
            for idx in range(len(self.meta_models)):
                cvmeta = CrossValRegressor(self.meta_models[idx], n_split=self.n_split)
                preds_val, preds_test = cvmeta.fit(new_features_train, y, X_test=new_features_test, y_test=y_test,
                           eval_metric=eval_metric, early_stopping_rounds=early_stopping_rounds, verbose=verbose,
                           verbose_eval=verbose_eval_cv , custom_eval_metric=self.custom_metric_meta[idx])
                self.CVmeta_models.append(cvmeta)
                preds_meta_val[idx,:] = preds_val
                
                if X_test is not None:
                    preds_meta_test[idx,:] = preds_test
                    
                    
            print('End training Meta all data :', time.time()-begin)
            
            if eval_metric is not None:
                print("Validation K-fold averaging Meta all data: ", eval_metric(y, np.mean( preds_meta_val, axis=0)))
            if X_test is not None and y_test is not None:
                #preds_meta_test  = preds_test.reshape((preds_test.shape[0], len(self.CVmeta_models), -1))
                #preds_meta_test = preds_meta_test.transpose(1,0,2) # (model, rowdata, probability)
                print("Evaluation Meta learner averaging all data:", eval_metric(y_test, np.mean(preds_meta_test, axis=0) ))
                #preds_proba_meta_test, _ = self.predict_proba(X_test, average=False, return_features=False)
                #print(np.array_equiv(preds_meta_test, preds_proba_meta_test))
                #print(preds_proba_meta_test.shape, preds_meta_test.shape)
                #print(np.not_equal(preds_meta_test, preds_proba_meta_test).sum())
    
    def save(self, name='stackRegressor.pkl'):
        self.models = None
        self.meta_models = None
        with open(name, 'wb') as f:
            pickle.dump(self, f)
        
   
    def predict(self, data, average=True, return_features=False, only_features=False):
        """
            predict regression of the stacking. 1) predict new features with simple models
            2) predict from the features +data the results by default. return by default an average of the the meta learner
            
            average : average the meta learner results if there are multiple meta learners
            return_features : return features created by simples models (not average)
            only_features : return prediction of the meta learner only on the new features
        """
        
        if only_features == False and self.only_features == True:
            raise('Model train only with new features, put only_features parameters as True')
        predictions_data = -1
        # predict simple model
        for idx_models in range(len(self.CVmodels)):
            if idx_models == 0:
                temp = self.CVmodels[idx_models].predict(data, method='average')
                new_features_data = temp
            else:
                temp = self.CVmodels[idx_models].predict(data, method='average')
                new_features_data = np.column_stack((new_features_data, temp))

        
        # predict with meta learners
        if only_features:
            # prediction only on new features
            new_data = new_features_data
            for idx_meta in range(len(self.CVmeta_models_of)):
                if idx_meta == 0:
                    temp = self.CVmeta_models_of[idx_meta].predict(new_data, method='average')
                    predictions_data = np.zeros((len(self.CVmeta_models_of), *temp.shape))
                    predictions_data[idx_meta, :] = temp
                else:
                    temp = self.CVmeta_models_of[idx_meta].predict(new_data, method='average')
                    predictions_data[idx_meta, :] = temp
            
            if average:
                predictions_data = predictions_data.mean(axis=0)
            
            if return_features:
                new_features_data = new_features_data.reshape((new_features_data.shape[0], len(self.CVmodels) ) )
                new_features_data = new_features_data.transpose(1,0) # format to (models, features row, features columnsS)
                return predictions_data, new_features_data  # Meta / Simple Learniner
            else:
                return predictions_data, None
        else:
            new_data = np.column_stack((data, new_features_data))
            for idx_meta in range(len(self.CVmeta_models)):
                if idx_meta == 0:
                    temp = self.CVmeta_models[idx_meta].predict(new_data, method='average')
                    predictions_data = np.zeros((len(self.CVmeta_models), *temp.shape))
                    predictions_data[idx_meta, :] = temp
                else:
                    temp = self.CVmeta_models[idx_meta].predict(new_data, method='average')
                    predictions_data[idx_meta, :] = temp
            
            if average:
                predictions_data = predictions_data.mean(axis=0)
            
            if return_features:
                new_features_data = new_features_data.reshape((new_features_data.shape[0], len(self.CVmodels) ) )
                new_features_data = new_features_data.transpose(1,0)
                return predictions_data, new_features_data # Meta / Simple Learniner
            else:
                return predictions_data, None
    
        