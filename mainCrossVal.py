# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 16:50:11 2018

@author: shiro
"""
from sklearn import datasets
from CrossValidation import RMSE, CrossVal
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
dtrain =  xgb.DMatrix(x_train, label=y_train) # datas.slice(train_idx)
dval = xgb.DMatrix(x_val, label=y_val)

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
print(y_train, y_val)


xgb_params2 = {'objective':'multi:softprob', 'max_depth':2, 'random_state':10, 'n_estimators':1500, 'learning_rate':0.1, 'silent':False,
                  'booster':'gbtree', 'metric': 'mlogloss', 'num_class':len(np.unique(y)) }#"binary:logistic"

#model = xgb.train(xgb_params2, dtrain=dtrain,num_boost_round=1000, evals=[(dval, 'eval')],
                                #                      early_stopping_rounds=20, verbose_eval=False )
### Classification
model = CrossVal(xgb, 'xgb', n_split=5)
model.fit(x_train, y_train, x_val, y_val, classification=True, binary=False, eval_metric=acc, params=xgb_params2)

model.save_models('test_classifcv.pkl')
del model

with open('test_classifcv.pkl', 'rb') as f:
    model = pickle.load(f)
preds = model.predict(x_val)


preds = model.predict(x_val)

print(y_val.shape, preds.shape)
print(acc(y_val, preds))

#
## adaboost classifier
print("### ADABOOST ###")
model = CrossVal(
         AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), algorithm="SAMME", random_state=10, n_estimators=10)
                                        , 'adaboost', n_split=10)
model.fit(x_train, y_train, x_val, y_val, classification=True, eval_metric=acc)
model.save_models('test_adaboostclassifcv.pkl')
del model

with open('test_adaboostclassifcv.pkl', 'rb') as f:
    model = pickle.load(f)
preds = model.predict_proba(x_val)
print(acc(y_val, preds))

## rf classifier
print('#### Random Forest ####')
model = CrossVal(
        RandomForestClassifier(bootstrap=True, max_depth=20, random_state=10, max_features=0.6, 
                                            min_samples_leaf=4, min_samples_split=2, n_estimators=50, n_jobs=-1)
                                        , 'rf', n_split=10)
model.fit(x_train, y_train, x_val, y_val, classification=True, eval_metric=acc)
model.save_models('test_rfclassifcv.pkl')
del model

with open('test_rfclassifcv.pkl', 'rb') as f:
    model = pickle.load(f)
preds = model.predict_proba(x_val)
print(acc(y_val, preds))

## SVM classifier
print("### svm ###")
model = CrossVal(
         SVC(C=1.0, kernel='linear', probability=True), 'svm', n_split=10)
model.fit(x_train, y_train, x_val, y_val, classification=True, eval_metric=acc)
model.save_models('test_svmcv.pkl')
del model

with open('test_svmcv.pkl', 'rb') as f:
    model = pickle.load(f)
preds = model.predict_proba(x_val)
print(acc(y_val, preds))

## lgb
lgb_params_classif = {'metric':'multi_logloss', 'nthread':4, 'n_estimators':10000, 'learning_rate':0.02, 'num_leaves' :2,
            'colsample_bytree':0.9497036, 'subsample':0.8715623, 'max_depth':8, 'reg_alpha':0.041545473,
            'reg_lambda':0.0735294,  'min_child_weight':2,
            'silent':-1,'verbose':-1, 'objective':'multiclass', 'seed':3, 'num_class':len(np.unique(y))}

model = CrossVal(lgb, 'lgb', n_split=10)
model.fit(x_train, y_train, x_val, y_val, classification=True, eval_metric=acc, params=lgb_params_classif)
model.save_models('test_lgbclassifcv.pkl')
del model

with open('test_lgbclassifcv.pkl', 'rb') as f:
    model = pickle.load(f)
preds = model.predict(x_val)
print(acc(y_val, preds))
def regression():
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
    
    
    ## XGB regression 
    xgb_params = { 'max_depth':20, 'random_state':10, 'n_estimators':1500, 'learning_rate':0.1, 'silent':False,
                      'booster':'gbtree', 'min_child_weight':57, 'gamma':1.45, 'alpha':0.0,
                       'subsample':0.67, 'colsample_bytree':0.054, 'colsample_bylevel':0.5, 'metric': 'rmse'}
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_post, y_train_csv, test_size=0.1, random_state=10)
    model = CrossVal(xgb, 'xgb', n_split=10)
    model.fit(X_train.values, np.log1p(Y_train.values), X_val.values, eval_metric=RMSE, params=xgb_params)
    model.save_models('test_regressioncv.pkl')
    del model
    
    with open('test_regressioncv.pkl', 'rb') as f:
        model = pickle.load(f)
    preds = model.predict(X_val.values)
    print(RMSE(np.log1p(Y_val.values), preds))
    
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
    
    model = CrossVal(lgb, 'lgb', n_split=10)
    model.fit(X_train.values, np.log1p(Y_train.values), X_val.values, eval_metric=RMSE, params=lgb_params)
    model.save_models('test_regressionlgbcv.pkl')
    del model
    
    with open('test_regressionlgbcv.pkl', 'rb') as f:
        model = pickle.load(f)
    preds = model.predict(X_val.values)
    
    print(RMSE(np.log1p(Y_val.values), preds))
    
    ## rf REGRESSION
    
    model = CrossVal(
            RandomForestRegressor(bootstrap=True, max_depth=10, random_state=10, max_features=0.4, 
                                                min_samples_leaf=4, min_samples_split=2, n_estimators=50, n_jobs=-1)
                                            , 'rf', n_split=10)
    model.fit(np.nan_to_num(X_train.values), np.log1p(Y_train.values), np.nan_to_num(X_val.values), eval_metric=RMSE)
    model.save_models('test_regressionrfcv.pkl')
    del model
    
    with open('test_regressionrfcv.pkl', 'rb') as f:
        model = pickle.load(f)
    preds = model.predict(np.nan_to_num(X_val.values))
    
    print(RMSE(np.log1p(Y_val.values), preds))
    
    ## adaboost regression
    
    model = CrossVal(
             AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), random_state=10, n_estimators=10)
                                            , 'adaboost', n_split=10)
    model.fit(np.nan_to_num(X_train.values), np.log1p(Y_train.values), np.nan_to_num(X_val.values), eval_metric=RMSE)
    model.save_models('test_adaboostcv.pkl')
    del model
    
    with open('test_adaboostcv.pkl', 'rb') as f:
        model = pickle.load(f)
    preds = model.predict(np.nan_to_num(X_val.values))
    
    print(RMSE(np.log1p(Y_val.values), preds))