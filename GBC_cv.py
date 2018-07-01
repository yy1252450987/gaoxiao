
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
train_x.fillna(0, inplace=True)
test_x.fillna(0, inplace=True)
## scikit-learn Gradient Boosting classfier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# CV n_estimators
'''
cv_params = {'n_estimators': [50,100,150,200,250]}
other_params = {'n_estimators': 50, 'learning_rate': 0.1, 'subsample':0.8,
                'min_samples_split':400,'min_samples_leaf':50, 
                'max_depth': 8, 'max_features':15,
                'random_state': 1}
model = GradientBoostingClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('results:{0}'.format(evalute_result))
print('best_params_：{0}'.format(optimized_GBM.best_params_))
print('best_score_:{0}'.format(optimized_GBM.best_score_))

'''

# CV n_estimators
'''
cv_params = {'n_estimators': [160,180,200,220,240]}
other_params = {'n_estimators': 200, 'learning_rate': 0.1, 'subsample':0.8,
                'min_samples_split':400,'min_samples_leaf':50, 
                'max_depth': 8, 'max_features':15,
                'random_state': 1}
model = GradientBoostingClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('results:{0}'.format(evalute_result))
print('best_params_：{0}'.format(optimized_GBM.best_params_))
print('best_score_:{0}'.format(optimized_GBM.best_score_))
'''

# max_depth and min_samples_split:
'''
cv_params = {'max_depth': [6,8,10,12,14], 'min_samples_split':[200,400,600,800,1000]}
other_params = {'n_estimators': 200, 'learning_rate': 0.1, 'subsample':0.8,
                'min_samples_split':400,'min_samples_leaf':50, 
                'max_depth': 8, 'max_features':15,
                'random_state': 1}
model = GradientBoostingClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('results:{0}'.format(evalute_result))
print('best_params_：{0}'.format(optimized_GBM.best_params_))
print('best_score_:{0}'.format(optimized_GBM.best_score_))
'''

# CV min_samples_split and min_samples_leaf
'''
cv_params = {'min_samples_split': [600,800,1000,1200,1400], 'min_samples_leaf':[30,40,50,60,70]}
other_params = {'n_estimators': 200, 'learning_rate': 0.1, 'subsample':0.8,
                'min_samples_split':400,'min_samples_leaf':50, 
                'max_depth': 10, 'max_features':15,
                'random_state': 1}
model = GradientBoostingClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('results:{0}'.format(evalute_result))
print('best_params_：{0}'.format(optimized_GBM.best_params_))
print('best_score_:{0}'.format(optimized_GBM.best_score_))
'''

# CV subsample
'''
cv_params = {'subsample': [0.7,0.75,0.8,0.85,0.9]}
other_params = {'n_estimators': 200, 'learning_rate': 0.1, 'subsample':0.8,
                'min_samples_split':1000,'min_samples_leaf':50, 
                'max_depth': 10, 'max_features':15,
                'random_state': 1}
model = GradientBoostingClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('results:{0}'.format(evalute_result))
print('best_params_：{0}'.format(optimized_GBM.best_params_))
print('best_score_:{0}'.format(optimized_GBM.best_score_))
'''

# CV learning_rate and n_estimators
'''
cv_params = {'learning_rate': [0.1,0.05], 'n_estimators':[200,400]}
other_params = {'n_estimators': 200, 'learning_rate': 0.1, 'subsample':0.85,
                'min_samples_split':1000,'min_samples_leaf':50, 
                'max_depth': 10, 'max_features':15,
                'random_state': 1}
model = GradientBoostingClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('results:{0}'.format(evalute_result))
print('best_params_：{0}'.format(optimized_GBM.best_params_))
print('best_score_:{0}'.format(optimized_GBM.best_score_))
'''


model_pred = GradientBoostingClassifier(n_estimators=200,learning_rate= 0.1, subsample=0.85,min_samples_split=1000,min_samples_leaf=50, max_depth= 10, max_features=15,random_state=1)

model_pred.fit(train_x, train_y)
fimp = model_pred.feature_importances_
sort_feature = train_x.columns[np.argsort(-fimp)] 
#print(sort_feature)
'''
### Feature Importance and Selection
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=1, shuffle=True)
for feature_num in range(20, 150, 10):
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
best_feature_num = 80
'''
### Threshold Selection (best: 0.41)

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
model_pred.fit(train_x[sort_feature[:best_feature_num]], train_y)
xgb_preds = model_pred.predict_proba(train_x[sort_feature[:best_feature_num]])[:,1]
for trd in np.asarray(range(40,45,1))/100.0:
    prediction = np.zeros(xgb_preds.shape[0])
    prediction[xgb_preds>trd] = 1
    print(trd, precision_recall_fscore_support(train_y, prediction, pos_label=1, average='binary'))
'''

best_threshold=0.41

### Prediction

#feature_num = train_x.shape[1] ## No Feature Selection
'''
model_pred.fit(train_x[sort_feature[:best_feature_num]], train_y)
xgb_preds = model_pred.predict_proba(test_x[sort_feature[:best_feature_num]])
results =  pd.DataFrame(te_uid)
results['pred'] = xgb_preds[:,1]
actuser = results[results.pred>best_threshold].user_id.unique()
np.savetxt('./output/gbc_b_'+'cvfeature'+ str(best_feature_num) +'_cv_trd'+str(best_threshold)+'.txt', actuser, fmt='%d')

'''

model_pred.fit(train_x[sort_feature[:best_feature_num]], train_y)
preds_tr = model_pred.predict_proba(train_x[sort_feature[:best_feature_num]])[:,1]
preds_te = model_pred.predict_proba(test_x[sort_feature[:best_feature_num]])[:, 1]
np.savetxt('./output/gbc_b_train_prob.txt', preds_tr, fmt='%6f')
np.savetxt('./output/gbc_b_test_prob.txt', preds_te, fmt='%6f')
