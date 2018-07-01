
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
#print(os.listdir("./input"))

# Any results you write to the current directory are saved as output.

import sklearn.metrics as metrics
import warnings
warnings.filterwarnings("ignore")
train_x = pd.read_csv('./input/train_x.csv')
train_y = pd.read_csv('./input/train_y.csv')
test_x = pd.read_csv('./input/test_x.csv')
te_uid = test_x.user_id

train_x = train_x.drop(['user_id'], axis=1)
test_x = test_x.drop(['user_id'], axis=1)


## scikit-learn XGB.classfier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# CV n_estimators
'''
cv_params = {'n_estimators': [100,200,300,400,500]}
other_params = {'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3, 'min_child_weight': 1, 'max_delta_step':0,
                'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel':1,
                'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,'objective':'binary:logistic', 'silent':True,
                'random_state':2, 'n_jobs':-1, 'booster':'gbtree', 'scale_pos_weight':1, 'base_score':0.5} 
model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('results:{0}'.format(evalute_result))
print('best_params_：{0}'.format(optimized_GBM.best_params_))
print('best_score_:{0}'.format(optimized_GBM.best_score_))

'''

# CV n_estimators
'''
cv_params = {'n_estimators': [500, 600, 300, 800, 300]}
other_params = {'n_estimators': 500, 
                'learning_rate': 0.1, 
                'max_depth': 3, 'min_child_weight': 1, 'max_delta_step':0,
                'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel':1,
                'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,'objective':'binary:logistic', 'silent':True,
                'random_state':1, 'n_jobs':-1, 'booster':'gbtree', 'scale_pos_weight':1, 'base_score':0.5} 
model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('results:{0}'.format(evalute_result))
print('best_params_：{0}'.format(optimized_GBM.best_params_))
print('best_score_:{0}'.format(optimized_GBM.best_score_))
#'''

# CV n_estimators
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
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('results:{0}'.format(evalute_result))
print('best_params_：{0}'.format(optimized_GBM.best_params_))
print('best_score_:{0}'.format(optimized_GBM.best_score_))
'''

# CV max_depth
'''
cv_params = {'max_depth': [1, 3, 5, 7, 9]}
other_params = {'n_estimators': 80, 
                'learning_rate': 0.1, 
                'max_depth': 5, 'min_child_weight': 1, 'max_delta_step':0,
                'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel':1,
                'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,'objective':'binary:logistic', 'silent':True,
                'random_state':1, 'n_jobs':-1, 'booster':'gbtree', 'scale_pos_weight':1, 'base_score':0.5} 
model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('results:{0}'.format(evalute_result))
print('best_params_：{0}'.format(optimized_GBM.best_params_))
print('best_score_:{0}'.format(optimized_GBM.best_score_))

'''

# CV min_child_weight
'''
cv_params = {'min_child_weight': [7, 9, 11, 13]}
other_params = {'n_estimators': 80, 
                'learning_rate': 0.1,
                'max_depth': 5, 'min_child_weight': 1, 'max_delta_step':0,
                'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel':1,
                'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,'objective':'binary:logistic', 'silent':True,
                'random_state':1, 'n_jobs':-1, 'booster':'gbtree', 'scale_pos_weight':1, 'base_score':0.5} 
model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('results:{0}'.format(evalute_result))
print('best_params_：{0}'.format(optimized_GBM.best_params_))
print('best_score_:{0}'.format(optimized_GBM.best_score_))
'''

# CV gamma
'''
cv_params = {'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
other_params = {'n_estimators': 80, 
                'learning_rate': 0.1, 
                'max_depth': 5, 'min_child_weight': 9, 'max_delta_step':0,
                'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel':1,
                'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,'objective':'binary:logistic', 'silent':True,
                'random_state':1, 'n_jobs':-1, 'booster':'gbtree', 'scale_pos_weight':1, 'base_score':0.5} 
model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('results:{0}'.format(evalute_result))
print('best_params_：{0}'.format(optimized_GBM.best_params_))
print('best_score_:{0}'.format(optimized_GBM.best_score_))

'''

# CV subsample and colsample_bytree
'''
cv_params = {'subsample': [0.1, 0.3, 0.5, 0.7, 0.9], 
             'colsample_bytree': [0.1, 0.3, 0.5, 0.7, 0.9]}
other_params = {'n_estimators': 80,
                'learning_rate': 0.1, 
                'max_depth': 5, 'min_child_weight': 9, 'max_delta_step':0,
                'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel':1,
                'gamma': 0.4, 'reg_alpha': 0, 'reg_lambda': 1,'objective':'binary:logistic', 'silent':True,
                'random_state':1, 'n_jobs':-1, 'booster':'gbtree', 'scale_pos_weight':1, 'base_score':0.5} 
model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('results:{0}'.format(evalute_result))
print('best_params_：{0}'.format(optimized_GBM.best_params_))
print('best_score_:{0}'.format(optimized_GBM.best_score_))
'''

# CV reg_alpha and reg_lambda
'''
cv_params = {'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]}
other_params = {'n_estimators': 80, 
                'learning_rate': 0.1, 
                'max_depth': 5, 'min_child_weight': 9, 'max_delta_step':0,
                'subsample': 0.7, 'colsample_bytree': 0.1, 'colsample_bylevel':1,
                'gamma': 0.4, 'reg_alpha': 0, 'reg_lambda': 1,'objective':'binary:logistic', 'silent':True,
                'random_state':1, 'n_jobs':-1, 'booster':'gbtree', 'scale_pos_weight':1, 'base_score':0.5} 
model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('results:{0}'.format(evalute_result))
print('best_params_：{0}'.format(optimized_GBM.best_params_))
print('best_score_:{0}'.format(optimized_GBM.best_score_))

'''

# CV learning_rate
'''
cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
other_params = {'n_estimators': 80, 
                'learning_rate': 0.1, 
                'max_depth': 5, 'min_child_weight': 9, 'max_delta_step':0,
                'subsample': 0.7, 'colsample_bytree': 0.1, 'colsample_bylevel':1,
                'gamma': 0.4, 'reg_alpha': 3, 'reg_lambda': 2,'objective':'binary:logistic', 'silent':True,
                'random_state':1, 'n_jobs':-1, 'booster':'gbtree', 'scale_pos_weight':1, 'base_score':0.5} 
model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('results:{0}'.format(evalute_result))
print('best_params_：{0}'.format(optimized_GBM.best_params_))
print('best_score_:{0}'.format(optimized_GBM.best_score_))

'''
#'''
model_pred = XGBClassifier(n_estimators= 80, 
                learning_rate= 0.1, 
                max_depth= 5, min_child_weight= 9, max_delta_step=0,
                subsample= 0.7, colsample_bytree= 0.1, colsample_bylevel=1,
                gamma= 0.4, reg_alpha= 3, reg_lambda= 2,objective='binary:logistic', silent=True,
                random_state=1, n_jobs=-1, booster='gbtree', scale_pos_weight=1, base_score=0.5)

model_pred.fit(train_x, train_y)
fimp = model_pred.feature_importances_
sort_feature = train_x.columns[np.argsort(-fimp)] 
'''
### Feature Importance and Selection
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=1111, shuffle=True)
for feature_num in range(2, 150, 1):
    score = 0.0
    for train_index, test_index in kf.split(train_x):
        train_x_tr, train_x_va = train_x.iloc[train_index], train_x.iloc[test_index]
        train_y_tr, train_y_va = train_y.iloc[train_index], train_y.iloc[test_index]
        model_pred.fit(train_x_tr[sort_feature[:feature_num]], train_y_tr)
        results = model_pred.predict(train_x_va[sort_feature[:feature_num]])
        score += precision_recall_fscore_support(train_y_va, results, pos_label=1, average='binary')[2]
    score /= 5
    print(feature_num, score)
'''
best_feature_num = 32
'''
### Threshold Selection (best: 0.41)

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
model_pred.fit(train_x[sort_feature[:best_feature_num]], train_y)
xgb_preds = model_pred.predict_proba(train_x[sort_feature[:best_feature_num]])[:,1]
for trd in np.asarray(range(35,50,1))/100.0:
    prediction = np.zeros(xgb_preds.shape[0])
    prediction[xgb_preds>trd] = 1
    print(trd, precision_recall_fscore_support(train_y, prediction, pos_label=1, average='binary'))
'''

best_threshold=0.42

### Prediction

#feature_num = train_x.shape[1] ## No Feature Selection
'''
model_pred.fit(train_x[sort_feature[:best_feature_num]], train_y)
xgb_preds = model_pred.predict_proba(test_x[sort_feature[:best_feature_num]])
results =  pd.DataFrame(te_uid)
results['pred'] = xgb_preds[:,1]
actuser = results[results.pred>best_threshold].user_id.unique()
np.savetxt('./output/xgb_b_'+'cvfeature'+ str(best_feature_num) +'_cv_trd'+str(best_threshold)+'.txt', actuser, fmt='%d')
'''

model_pred.fit(train_x[sort_feature[:best_feature_num]], train_y)
preds_tr = model_pred.predict_proba(train_x[sort_feature[:best_feature_num]])[:,1]
preds_te = model_pred.predict_proba(test_x[sort_feature[:best_feature_num]])[:, 1]

np.savetxt('./output/xgb_b_train_prob.txt', preds_tr, fmt='%6f')
np.savetxt('./output/xgb_b_test_prob.txt', preds_te, fmt='%6f')
