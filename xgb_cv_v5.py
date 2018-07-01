
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import sklearn.metrics as metrics
import warnings
warnings.filterwarnings("ignore")
tr_x = pd.read_csv('./input/tr_x.v5.csv')
tr_y = pd.read_csv('./input/tr_y.v5.csv')
va_x = pd.read_csv('./input/va_x.v5.csv')
va_y = pd.read_csv('./input/va_y.v5.csv')
te_x = pd.read_csv('./input/te_x.v5.csv')
te_uid = te_x.user_id

# In[3]:
tr_x = tr_x.drop(['user_id'], axis=1)
te_x = te_x.drop(['user_id'], axis=1)
va_x = va_x.drop(['user_id'], axis=1)

train_x = pd.concat([tr_x, va_x], axis=0)
train_y = pd.concat([tr_y, va_y], axis=0)
# In[4]:

## scikit-learn XGB.classfier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

model_default = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True,
                             objective='binary:logistic', booster='gbtree', n_jobs=-1,
                              gamma=0, min_child_weight=1, max_delta_step=0, 
                              subsample=1, colsample_bytree=1, colsample_bylevel=1, 
                              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5,
                              random_state=1)
model_default.fit(tr_x, tr_y)
# feature importance
fimp = model_default.feature_importances_
sort_feature = tr_x.columns[np.argsort(-fimp)]

# In[5]:
'''
# Festure Selection
from sklearn.metrics import precision_recall_fscore_support
for i in range(1, len(sort_feature)):
    model_default.fit(tr_x[sort_feature[:i]], tr_y)
    results = model_default.predict(va_x[sort_feature[:i]])
    print(i, precision_recall_fscore_support(va_y, results,pos_label=1, average='binary'))
'''
# ### n_estimators = []
'''
cv_params = {'n_estimators': [100,200,300,400,500]}
other_params = {'n_estimators': 100, 
                'learning_rate': 0.1, 
                'max_depth': 3, 'min_child_weight': 1, 'max_delta_step':0,
                'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel':1,
                'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,'objective':'binary:logistic', 'silent':True,
                'random_state':1, 'n_jobs':-1, 'booster':'gbtree', 'scale_pos_weight':1, 'base_score':0.5} 
model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(tr_x, tr_y)
evalute_result = optimized_GBM.grid_scores_
print('results:{0}'.format(evalute_result))
print('best_params_：{0}'.format(optimized_GBM.best_params_))
print('best_score_:{0}'.format(optimized_GBM.best_score_))

'''
# ### n_estimators = []
'''
cv_params = {'n_estimators': [60,80,100,120,140]}
other_params = {'n_estimators': 100, 
                'learning_rate': 0.1, 
                'max_depth': 3, 'min_child_weight': 1, 'max_delta_step':0,
                'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel':1,
                'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,'objective':'binary:logistic', 'silent':True,
                'random_state':1, 'n_jobs':-1, 'booster':'gbtree', 'scale_pos_weight':1, 'base_score':0.5} 
model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(tr_x, tr_y)
evalute_result = optimized_GBM.grid_scores_
print('results:{0}'.format(evalute_result))
print('best_params_：{0}'.format(optimized_GBM.best_params_))
print('best_score_:{0}'.format(optimized_GBM.best_score_))
'''
# ### n_estimators = []
'''
cv_params = {'n_estimators': [70,80,90]}
other_params = {'n_estimators': 80, 
                'learning_rate': 0.1, 
                'max_depth': 3, 'min_child_weight': 1, 'max_delta_step':0,
                'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel':1,
                'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,'objective':'binary:logistic', 'silent':True,
                'random_state':1, 'n_jobs':-1, 'booster':'gbtree', 'scale_pos_weight':1, 'base_score':0.5} 
model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(tr_x, tr_y)
evalute_result = optimized_GBM.grid_scores_
print('results:{0}'.format(evalute_result))
print('best_params_：{0}'.format(optimized_GBM.best_params_))
print('best_score_:{0}'.format(optimized_GBM.best_score_))
'''
# ### n_estimators = []
'''
cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10]}
other_params = {'n_estimators': 90, 
                'learning_rate': 0.1, 
                'max_depth': 3, 'min_child_weight': 1, 'max_delta_step':0,
                'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel':1,
                'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,'objective':'binary:logistic', 'silent':True,
                'random_state':1, 'n_jobs':-1, 'booster':'gbtree', 'scale_pos_weight':1, 'base_score':0.5} 
model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(tr_x, tr_y)
evalute_result = optimized_GBM.grid_scores_
print('results:{0}'.format(evalute_result))
print('best_params_：{0}'.format(optimized_GBM.best_params_))
print('best_score_:{0}'.format(optimized_GBM.best_score_))
'''
# ### n_estimators = []
'''
cv_params = {'min_child_weight': [1,3, 5, 7, 9]}
other_params = {'n_estimators': 90, 
                'learning_rate': 0.1, 
                'max_depth': 3, 'min_child_weight': 1, 'max_delta_step':0,
                'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel':1,
                'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,'objective':'binary:logistic', 'silent':True,
                'random_state':1, 'n_jobs':-1, 'booster':'gbtree', 'scale_pos_weight':1, 'base_score':0.5} 
model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(tr_x, tr_y)
evalute_result = optimized_GBM.grid_scores_
print('results:{0}'.format(evalute_result))
print('best_params_：{0}'.format(optimized_GBM.best_params_))
print('best_score_:{0}'.format(optimized_GBM.best_score_))
'''
# ### n_estimators = []
'''
cv_params = {'gamma': [0,0.1, 0.2, 0.3, 0.4, 0.5]}
other_params = {'n_estimators': 90,
                'learning_rate': 0.1, 
                'max_depth': 3, 'min_child_weight': 1, 'max_delta_step':0,
                'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel':1,
                'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,'objective':'binary:logistic', 'silent':True,
                'random_state':1, 'n_jobs':-1, 'booster':'gbtree', 'scale_pos_weight':1, 'base_score':0.5} 
model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(tr_x, tr_y)
evalute_result = optimized_GBM.grid_scores_
print('results:{0}'.format(evalute_result))
print('best_params_：{0}'.format(optimized_GBM.best_params_))
print('best_score_:{0}'.format(optimized_GBM.best_score_))
'''
# ### n_estimators = []
'''
cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9,1], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9,1]}
other_params = {'n_estimators': 90, 
                'learning_rate': 0.1, 
                'max_depth': 3, 'min_child_weight': 1, 'max_delta_step':0,
                'subsample': 0.1, 'colsample_bytree': 1, 'colsample_bylevel':1,
                'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,'objective':'binary:logistic', 'silent':True,
                'random_state':1, 'n_jobs':-1, 'booster':'gbtree', 'scale_pos_weight':1, 'base_score':0.5} 
model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(tr_x, tr_y)
evalute_result = optimized_GBM.grid_scores_
print('results:{0}'.format(evalute_result))
print('best_params_：{0}'.format(optimized_GBM.best_params_))
print('best_score_:{0}'.format(optimized_GBM.best_score_))

'''

'''
cv_params = {'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]}
other_params = {'n_estimators': 90, 
                'learning_rate': 0.1, 
                'max_depth': 3, 'min_child_weight': 1, 'max_delta_step':0,
                'subsample': 1, 'colsample_bytree': 0.7, 'colsample_bylevel':1,
                'gamma': 0.1, 'reg_alpha': 0, 'reg_lambda': 1,'objective':'binary:logistic', 'silent':True,
                'random_state':1, 'n_jobs':-1, 'booster':'gbtree', 'scale_pos_weight':1, 'base_score':0.5} 
model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(tr_x, tr_y)
evalute_result = optimized_GBM.grid_scores_
print('results:{0}'.format(evalute_result))
print('best_params_：{0}'.format(optimized_GBM.best_params_))
print('best_score_:{0}'.format(optimized_GBM.best_score_))
'''

'''
cv_params = {'learning_rate': [0.01,0.02, 0.05, 0.07, 0.1]}
other_params = {'n_estimators': 90, 
                'learning_rate': 0.1, 
                'max_depth': 3, 'min_child_weight': 1, 'max_delta_step':0,
                'subsample': 1, 'colsample_bytree': 0.7, 'colsample_bylevel':1,
                'gamma': 0.1, 'reg_alpha': 1, 'reg_lambda': 3,'objective':'binary:logistic', 'silent':True,
                'random_state':1, 'n_jobs':-1, 'booster':'gbtree', 'scale_pos_weight':1, 'base_score':0.5} 
model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(tr_x, tr_y)
evalute_result = optimized_GBM.grid_scores_
print('results:{0}'.format(evalute_result))
print('best_params_：{0}'.format(optimized_GBM.best_params_))
print('best_score_:{0}'.format(optimized_GBM.best_score_))
'''

#'''
model_pred = XGBClassifier(n_estimators= 90, 
                learning_rate= 0.1, 
                max_depth= 3, min_child_weight= 1, max_delta_step=0,
                subsample= 1, colsample_bytree= 0.7, colsample_bylevel=1,
                gamma= 0.1, reg_alpha= 1, reg_lambda= 3,objective='binary:logistic', silent=True,
                random_state=1, n_jobs=-1, booster='gbtree', scale_pos_weight=1, base_score=0.5)


# In[24]:

best_feature_num = 51
'''
from sklearn.metrics import precision_recall_fscore_support
for feature_num in [51]:
    print('FEATURE: ', feature_num, '*****')
    model_pred.fit(tr_x[sort_feature[:feature_num]], tr_y)
    xgb_preds = model_pred.predict_proba(va_x[sort_feature[:feature_num]])[:, 1]
    threshold = np.array(range(30, 60, 2))/100.0
    for trd in threshold:
        results = np.zeros(len(va_y))
        results[xgb_preds>trd]=1
        print(trd, precision_recall_fscore_support(va_y, results,pos_label=1, average='binary'))
'''

'''
model_pred.fit(train_x[sort_feature[:feature_num]], train_y)
xgb_preds = model_pred.predict_proba(te_x[sort_feature[:feature_num]])
results =  pd.concat([te_uid,te_x.register_day], axis=1)
results['pred'] = xgb_preds[:,1]
trd = 0.4
actuser = results[results.pred>trd].user_id.unique()
np.savetxt('xgb.v5.'+'feature'+ str(feature_num) +'.cv.trd'+str(trd)+'.txt', actuser, fmt='%d')
'''


model_pred.fit(train_x[sort_feature[:best_feature_num]], train_y)
preds_tr = model_pred.predict_proba(train_x[sort_feature[:best_feature_num]])[:,1]
preds_te = model_pred.predict_proba(te_x[sort_feature[:best_feature_num]])[:, 1]
np.savetxt('./output/xgb_b_v5_train_prob.txt', preds_tr, fmt='%6f')
np.savetxt('./output/xgb_b_v5_test_prob.txt', preds_te, fmt='%6f')
