
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


# In[2]:

# Load dataset
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
# Normalization
train_x.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
test_x.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
train_x = np.asarray(train_x)
test_x = np.asarray(test_x)
train_y = np.asarray(train_y)
#cv_params = {'learning_rate_init': [0.0001, 0.001, 0.01, 0.1]}
#other_params = {'solver':'adam','hidden_layer_sizes':(128,32,8,2), 'activation':'relu', 'batch_size':32, 'learning_rate':'constant', 'learning_rate_init':0.1}
#model = MLPClassifier(**other_params)

print('Model Selection....')
#param_grids = [{'solver':['adam'],'hidden_layer_sizes':[(100,50,20,10,2),(100,50,20,2),(100,50,2)], 'activation':['relu','tanh'], 'batch_size':[32,64,128,256], 'learning_rate':['constant'], 'learning_rate_init':[0.1, 0.01, 0.001, 0.0001]},
#               {'solver':['sgd'], 'hidden_layer_sizes':[(100,50,20,10,2),(100,50,20,2),(100,50,2)], 'activation':['relu','tanh'], 'batch_size':[32,64,128,256], 'learning_rate':['constant','adaptive'], 'learning_rate_init':[0.1, 0.01, 0.001, 0.0001]}]

### MLPC parameter
#hidden_layer_sizes=(10,),(10,5),(10,5,2)
#activation='relu' 'tanh'
#solver='adam' 'sgd'
# alpha=0.0001
# batch_size=32,64,128,
# learning_rate= ‘constant’, 'invscaling', 'adaptive(sgd)'
# power_t = '0.3, 0.5, 0.7(invscaling)'
# learning_rate_init= 0.1, 0.01, 1e-3, 1e-4
# max_iter = 10000
# momentum = 0.9(sgd)
# beta_1 = 0.9(adam)
# beta_2 = 0.999(adam)
# epsilon = 1e-8(adam)

# training model

#clf = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
#clf.fit(train_x, train_y)
#means = clf.cv_results_['mean_test_score']
#stds = clf.cv_results_['std_test_score']
#for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))solver='sgd'
clf = MLPClassifier(hidden_layer_sizes=(128,64,32,16,8), activation='tanh', solver='sgd', batch_size=128,alpha=0.001, learning_rate='adaptive', learning_rate_init=0.1,shuffle=True, max_iter=100,verbose=True, random_state=100)
#lf = MLPClassifier(hidden_layer_sizes=(128,32,8,2), activation='relu', solver='adam', batch_size=64,alpha=0.001, learning_rate='constant', learning_rate_init=0.01,shuffle=True, max_iter=100, early_stopping=True, validation_fraction=0.3, verbose=True)
clf.fit(train_x, train_y)
prediction = clf.predict_proba(test_x)
results =  pd.DataFrame(te_uid)
results['pred'] = prediction[:, 1]

for trd in np.array(range(30,50,1))/100.0:
    
    actuser = results[results.pred>trd].user_id.unique()
    print(trd,len(actuser))
    #np.savetxt('./output/mlpc_b_'+'cvfeature'+ str(239) +'_cv_trd'+str(trd)+'.txt', actuser, fmt='%d')
