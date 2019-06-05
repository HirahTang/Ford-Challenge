#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:58:19 2019

@author: TH
"""

!kaggle competitions download -c stayalert

#%%

# =============================================================================
# Library Importing
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from sklearn.svm import LinearSVC # Linear Support Vector Classigication
from sklearn.svm import NuSVC

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
#%%

# =============================================================================
# Data Loading
# =============================================================================

train = pd.read_csv('fordTrain.csv')
test = pd.read_csv('fordTest.csv')
sol = pd.read_csv('Solution.csv')
exp = pd.read_csv('example_submission.csv')

#%%

# =============================================================================
# Data inspection.
# =============================================================================

train.info()
test.info()
train.head()

sum(train['IsAlert'] == 0)

#%%
# =============================================================================
# First Model
# =============================================================================

# =============================================================================
# Data preprocessing
# =============================================================================

new_df = pd.DataFrame()  # Subset

noalert = train.index[train['IsAlert'] == 0]
turnnoalert = train.iloc[noalert-1][train['IsAlert'] == 1] # The change moment data


#new_df 

for i in turnnoalert.index: # Return 500ms before and after the moment
    new_df = new_df.append(train[i-5:i+6])
    
new_df = new_df.drop_duplicates() # Drop Dupicate instances

new_df
#%%

# =============================================================================
# Feature Engineering
# =============================================================================

#pca = PCA() 
pcadata = new_df.drop(columns = ['TrialID', 'ObsNum', 'IsAlert']) # Drop unneccessary columns

# Standarization

X_scaled = preprocessing.scale(pcadata) 
pcadata.info()

# Principle Component Analysis

pca=PCA() 
pca.fit(X_scaled) 
X_pca=pca.transform(X_scaled) 
#let's check the shape of X_pca array
print ("shape of X_pca", X_pca.shape)
#%%
# Scree Plot
y = pca.explained_variance_ratio_

x = np.array([i for i in range(1, len(y)+1)])

plt.plot(x, y, 'r-x')

#%%

# Filter principle components
sum(pca.explained_variance_ratio_[:14])
X_pca = X_pca[:,:14] # Slice the first 14 features
X_pca.shape 

X_pca # Training features
new_df.IsAlert # Target features

#%%

# =============================================================================
# Modeling 
# =============================================================================

# Logistic Regression Cross Validation

clf_CV = LogisticRegressionCV(cv=10, random_state=0, solver = 'liblinear') # Model Setting
clf_CV.fit(X_pca, new_df.IsAlert) # Model Fitting

# Evalution

clf_CV.score(X_pca, new_df.IsAlert) 

# Confustion matrix evaluation

clf_cv_pred = clf_CV.predict(X_pca)
len(clf_cv_pred == 1)
sum(clf_cv_pred == 0)

confusion_matrix(new_df.IsAlert, clf_cv_pred)
sum(new_df.IsAlert == 1)

roc_auc_score(new_df.IsAlert, clf_cv_pred)

#%%

# =============================================================================
# Try Another Models
# =============================================================================
# Cross Validated Naive Bayes Model
skf = StratifiedKFold(n_splits=10)
params = {}
nb = GaussianNB()
gs = GridSearchCV(nb, cv=skf, param_grid=params, return_train_score=True)

x_train, x_test, y_train, y_test = train_test_split(X_pca, new_df.IsAlert, random_state = 42)

gs.fit(x_train, y_train)
gs.cv_results_
gs.score(X_pca, new_df.IsAlert)
gs_predict = gs.predict(X_pca)

confusion_matrix(new_df.IsAlert, gs_predict)

roc_auc_score(new_df.IsAlert, gs_predict)
#nb.fit(x_train, y_train)
#nb.score(x_test, y_test)



# Cross Validated  Random Forest
grid_forest = RandomForestClassifier(random_state = 42)
hyperparams = [{'n_estimators':[5, 10 ,50]}]
grid_search = GridSearchCV(grid_forest, hyperparams, cv = 10, scoring = 'neg_mean_squared_error')

grid_search.fit(x_train, y_train)

selected_model = grid_search.best_estimator_
print ('Grid winner', selected_model.score(x_test, y_test))
selected_model.score(X_pca, new_df.IsAlert)
randomfpred = grid_search.predict(X_pca)
confusion_matrix(new_df.IsAlert, randomfpred)
roc_auc_score(new_df.IsAlert, randomfpred)

import pickle
from joblib import dump, load

dump(grid_search, 'randomforest.joblib')
rf_2 = load('randomforest.joblib') 
rf_2.best_estimator_.score(X_pca, new_df.IsAlert)
randomfpred_2 = rf_2.predict(X_pca)

confusion_matrix(new_df.IsAlert, randomfpred_2)
roc_auc_score(new_df.IsAlert, randomfpred_2)
#%%

# =============================================================================
# Support Vector Machine & Neural Network
# =============================================================================

lisvc = LinearSVC(random_state=42, tol=1e-5)
svcgridsearch = GridSearchCV(lisvc, cv = skf, param_grid = params, return_train_score = True)
svcgridsearch.fit(x_train, y_train) # Failed to convergence

svcgridsearch.cv_results_
svcgridsearch.score(X_pca, new_df.IsAlert)
lisvc_predict = svcgridsearch.predict(X_pca)

confusion_matrix(new_df.IsAlert, lisvc_predict)

roc_auc_score(new_df.IsAlert, lisvc_predict)

#%%

nusvc = NuSVC(gamma = 'scale')
nusvcgridsearch = GridSearchCV(nusvc, cv = skf, param_grid = params, return_train_score = True)
nusvcgridsearch.fit(x_train, y_train) 

nusvcgridsearch.cv_results_
nusvcgridsearch.score(X_pca, new_df.IsAlert)
nusvc_predict = nusvcgridsearch.predict(X_pca)

confusion_matrix(new_df.IsAlert, nusvc_predict)

roc_auc_score(new_df.IsAlert, nusvc_predict)

#%%
# Run from here 
# Epsilon - Support Vecto Regression
from sklearn.svm import SVR
svr = SVR(gamma='scale', C=1.0, epsilon=0.2)
svrgridsearch = GridSearchCV(svr, cv = skf, param_grid = params, return_train_score = True)
svrgridsearch.fit(x_train, y_train) 

svrgridsearch.cv_results_
svrgridsearch.score(X_pca, new_df.IsAlert)
svr_predict = svrgridsearch.predict(X_pca)

confusion_matrix(new_df.IsAlert, svr_predict)

roc_auc_score(new_df.IsAlert, svr_predict)

#%%

# C - Support Vector Regression
from sklearn.svm import SVC 

svc = SVC(gamma='auto')
svcgridsearch = GridSearchCV(svc, cv = skf, param_grid = params, return_train_score = True)
svcgridsearch.fit(x_train, y_train) 

svcgridsearch.cv_results_
svcgridsearch.score(X_pca, new_df.IsAlert)
svc_predict = svcgridsearch.predict(X_pca)

confusion_matrix(new_df.IsAlert, svc_predict)

roc_auc_score(new_df.IsAlert, svc_predict)

#%%

# Neural Network

from sklearn.neural_network import MLPClassifier

nn1 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
nngridsearch = GridSearchCV(nn1, cv = skf, param_grid = params, return_train_score = True)
nngridsearch.fit(x_train, y_train) 

nngridsearch.cv_results_
nngridsearch.score(X_pca, new_df.IsAlert)
nn_predict = nngridsearch.predict(X_pca)

confusion_matrix(new_df.IsAlert, nn_predict)

roc_auc_score(new_df.IsAlert, nn_predict)
#%%
#
## =============================================================================
## Exploratory Analysis
## =============================================================================
#
#train.head()
#max(train.TrialID)
#trialrange = [i for i in range(0, max(train.TrialID)+1)]
#
#alert_mean = []
#for i in trialrange:
#    alert_mean.append(train[train['TrialID'] == i]['IsAlert'].mean())
#    
#    
#alert_mean    
#df_new = train[train.TrialID == 1]
#df_new.to_csv(r'testdataset.csv')
#num_bins = 10
#n, bins, patches = plt.hist(alert_mean, num_bins, facecolor='blue', alpha=0.5)
#plt.show()    

#import math
#len(alert_mean)
#
#n = 0
#for i in alert_mean:
##    print (i)
#    if math.isnan(i):
##        num = alert_mean.index(i)
##        del alert_mean[num]
#        print (i)
#        print (alert_mean.index(i))
##        n += 1
#print (n)
#train[train['TrialID'] == 472]
##for i in train['TrialID']:
##    print (i)
#print (len(alert_mean))
#%%
# =============================================================================
# Feature Engineering
# =============================================================================
'''
train_2 = train[train['TrialID'] <= 1]
train_new = pd.DataFrame()
train_new
for i in range(0, max(train_2.TrialID)+1): # Within each trial
    temp_data = train_2[train_2['TrialID'] == i]
    for j in list(train_2)[3:]: # For all the attribute of each trial
#        print (train_2[train_2['TrialID'] == i][j])
        temp_data['m{}'.format(j)] = train_2[train_2['TrialID'] == i][j].rolling(window = 5).mean() #The Rolling mean
        temp_data['sd{}'.format(j)] = train_2[train_2['TrialID'] == i][j].rolling(window = 5).std()
    train_new = train_new.append(temp_data)
    
train_new.iloc[1200:1230]
#%%
train.head()

exp_train = train[train['TrialID'] == 0]
#exp_train = exp_train[0:5]
exp_train.head()
exp_train.iloc[0, 3:33]
exp_train.iloc[:,-1]
n = 0
for i in range(5):
    exp_train['m{}'.format(i)] = exp_train['TrialID']
    
for i in range(0, max(train2.TrialID)+1):
    for j in list(train)
max(train.TrialID)
for i in list(exp_train)[3:]:
    sum_ = 0
    for j in range(0, len(exp_train)):
        print (exp_train[i][j])
#    sum_ += exp_train[i]
#    exp_train['m{}'.format(i)] = exp_train[i].mean()

#    print (exp_train[i].mean())
#    n += 1
#    print (n)
exp_train.P1
exp_train['P1'].rolling(window = 5).mean()
exp_train.P1.rolling(window = 5).std()
#%%

# =============================================================================
# test section 
# =============================================================================
test_sum = [i for i in range(10)]
test_sum
#[i for i in range(1, len(test_sum)+1)]
sum_ = 0
rolling_mean = []
for i in range(1, len(test_sum)+1):
    sum_ += test_sum[i-1]
    rolling_mean.append(sum_ / i)
rolling_mean
#haha1 = '-' * 20 + 'predict' + '-' * 20 
#haha2 = '-' * 15 + 'No' + '-' * 15 + 'Yes' + '-' * 10
#haha3 = '-' * 10 + 'No' + '{}'.format()
#print (haha1,'\n', haha2, '\n', haha3)
'''
#
#try_ = pd.DataFrame(columns = ['1','2', '3'])
#try_['1'] = [1,3,3,5,6]
#try_
#try_.fillna(0)
#try_
