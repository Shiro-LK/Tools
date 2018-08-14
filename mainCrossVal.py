# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 16:50:11 2018

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

def acc_binary(y_true, y_preds):
    '''
        preds is probabily between 0 and 1. Considere >0.5, class 1 else class 0.
    '''
    
    preds = np.round(y_preds)
    return ('acc_binary', accuracy_score(y_true, preds))
def acc(y_true, y_preds):
    '''
        preds is probabily with n dimension so we need to take the argmax corresponding 
        to the class predicted
    '''
    
    preds = np.argmax(y_preds, axis=1)
    return ('acc', accuracy_score(y_true, preds))


# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y =iris.target
#np.place(y, y==2, 1)
print(y.shape)
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)




## lgb
lgb_params_classif = {'metric':'multi_logloss', 'nthread':4, 'n_estimators':10000, 'learning_rate':0.02, 'num_leaves' :2,
            'colsample_bytree':0.9497036, 'subsample':0.8715623, 'max_depth':8, 'reg_alpha':0.041545473,
            'reg_lambda':0.0735294,  'min_child_weight':2,
            'silent':-1,'verbose':-1, 'objective':'multiclass', 'seed':3, 'num_class':len(np.unique(y))}

lgb_params_classif2 = {'metric':'auc', 'nthread':4, 'n_estimators':10000, 'learning_rate':0.02, 'num_leaves' :2,
            'colsample_bytree':0.9497036, 'subsample':0.8715623, 'max_depth':8, 'reg_alpha':0.041545473,
            'reg_lambda':0.0735294,  'min_child_weight':2,
            'silent':-1,'verbose':-1, 'objective':'binary', 'seed':3}


model_single=LGBMClassifier(**lgb_params_classif2)
model_single.fit(x_train, y_train, eval_set=[(x_val, y_val)],verbose=False, early_stopping_rounds=10)
preds_val = model_single.predict_proba(x_val)
print('Acc single model :', acc(y_val, preds_val))


model = CrossValClassifier(LGBMClassifier(**lgb_params_classif2), n_split=10)
model.fit(x_train, y_train, x_val, y_val, eval_metric=acc)
model.save_models('test_lgbclassifcv.pkl')
del model

with open('test_lgbclassifcv.pkl', 'rb') as f:
    model = pickle.load(f)
preds = model.predict_proba(x_val)
print("Evaluation CV : ", acc(y_val, preds))








### XGBOOST

xgb_params2 = {'objective':'multi:softprob', 'max_depth':2, 'random_state':10, 'n_estimators':1500, 'learning_rate':0.1, 'silent':False,
                  'booster':'gbtree', 'metric': 'mlogloss', 'num_class':len(np.unique(y)) }#"binary:logistic"
xgb_params = {'objective':'binary:logistic', 'max_depth':2, 'random_state':10, 'n_estimators':1500, 'learning_rate':0.1, 'silent':False,
                  'booster':'gbtree', 'metric': 'logloss',}
model = XGBClassifier(**xgb_params2)
model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False, early_stopping_rounds=50)
preds_val = model.predict_proba(x_val, ntree_limit=model.best_iteration)
print("EVALUATION FINAL : ", acc(y_val, preds_val))


#model = xgb.train(xgb_params2, dtrain=dtrain,num_boost_round=1000, evals=[(dval, 'eval')],
                                #                      early_stopping_rounds=20, verbose_eval=False )
### Classification
model = CrossValClassifier(XGBClassifier(**xgb_params2), n_split=10)
model.fit(x_train, y_train, x_val, y_val, eval_metric=acc)

model.save_models('test_classifcv.pkl')
del model

with open('test_classifcv.pkl', 'rb') as f:
    model = pickle.load(f)
preds = model.predict_proba(x_val)

print(y_val.shape, preds.shape)
print('Evaluation CV : ', acc(y_val, preds))

#
## adaboost classifier
print("### ADABOOST ###")
      
model_single = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), algorithm="SAMME", random_state=10, n_estimators=10)
model_single.fit(x_train, y_train)
preds_val = model_single.predict_proba(x_val)
print('Single Model :', acc(y_val, preds_val))


model = CrossValClassifier(
         AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), algorithm="SAMME", random_state=10, n_estimators=10)
                                        ,  n_split=10)
model.fit(x_train, y_train, x_val, y_val, eval_metric=acc)
model.save_models('test_adaboostclassifcv.pkl')
del model

with open('test_adaboostclassifcv.pkl', 'rb') as f:
    model = pickle.load(f)
preds = model.predict_proba(x_val)
print('Evaluation adaboost CV :', acc(y_val, preds))

## rf classifier
print('#### Random Forest ####')
      
single_model = RandomForestClassifier(bootstrap=True, max_depth=20, random_state=10, max_features=0.6, 
                                      min_samples_leaf=4, min_samples_split=2, n_estimators=50, n_jobs=-1)
single_model.fit(x_train, y_train)
preds_val = single_model.predict_proba(x_val)
print('Evaluation single model : ', acc(y_val, preds_val))

model = CrossValClassifier(
        RandomForestClassifier(bootstrap=True, max_depth=20, random_state=10, max_features=0.6, 
                                            min_samples_leaf=4, min_samples_split=2, n_estimators=50, n_jobs=-1)
                                        , n_split=10)
model.fit(x_train, y_train, x_val, y_val, eval_metric=acc)
model.save_models('test_rfclassifcv.pkl')
del model

with open('test_rfclassifcv.pkl', 'rb') as f:
    model = pickle.load(f)
preds = model.predict_proba(x_val)
print('Evaluation RF CV :', acc(y_val, preds))

## SVM classifier
print("### svm ###")
single_model = SVC(C=1.0, kernel='linear', probability=True)
single_model.fit(x_train, y_train)
preds_val = single_model.predict_proba(x_val)
print('Evaluation Single model :', acc(y_val, preds))

model = CrossValClassifier(
         SVC(C=1.0, kernel='linear', probability=True), n_split=10)
model.fit(x_train, y_train, x_val, y_val, eval_metric=acc)
model.save_models('test_svmcv.pkl')
del model

with open('test_svmcv.pkl', 'rb') as f:
    model = pickle.load(f)
preds = model.predict_proba(x_val)
print('Evaluation SVM CV :', acc(y_val, preds))

### CatBoostClassifier
catboost_params ={
    'loss_function': 'MultiClass',
    'learning_rate': 0.1,
    'iterations': 5000,
    'depth': 8,
    #'class_weights': [1, 2],
    'bootstrap_type': 'Bernoulli',
    'random_seed': 3,
    'verbose': False,
    #'eval_metric':'AUC',
    'classes_count': len(np.unique(y))
}


model = CrossValClassifier(
         CatBoostClassifier(**catboost_params), n_split=10)
model.fit(x_train, y_train, x_val, y_val, eval_metric=acc)
model.save_models('test_catboostcv.pkl')
del model

with open('test_catboostcv.pkl', 'rb') as f:
    model = pickle.load(f)
preds = model.predict_proba(x_val)
print('Evaluation CatBoost CV :', acc(y_val, preds))

single_model = CatBoostClassifier(**catboost_params)
single_model.fit(x_train, y_train, eval_set=[(x_val, y_val)], use_best_model=True, early_stopping_rounds=50, verbose=False)
preds_val = single_model.predict_proba(x_val)
print('Evaluation CatBoost Single Model :', acc(y_val, preds_val))