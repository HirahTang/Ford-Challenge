#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 18:36:53 2019

@author: TH
"""

# Library Importation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load

test = pd.read_csv('fordTest.csv')
sol = pd.read_csv('Solution.csv')
exp = pd.read_csv('example_submission.csv')

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

# Choose the three models with the best performance
rf_pca = load('randomforest.joblib') 
rf_mstd = load('mstdforest.joblib')
nn_mstd = load('mstdnn.joblib')
#%%
from sklearn.decomposition import PCA
from sklearn import preprocessing
len(test)

pcatest = test.drop(columns = ['TrialID', 'ObsNum', 'IsAlert']) # Drop unneccessary columns

# Standarization

test_scaled = preprocessing.scale(pcatest) 

# Principle Component Analysis

pca=PCA() 
pca.fit(pcatest) 
test_pca=pca.transform(pcatest) 
#let's check the shape of X_pca array

#%%
# Scree Plot
y = pca.explained_variance_ratio_

x = np.array([i for i in range(1, len(y)+1)])

plt.plot(x, y, 'r-x')
test_pca = test_pca[:,:14] # Slice the first 14 features
test_pca.shape 
sum(pca.explained_variance_ratio_[:14])
#%%

sum(sol['Prediction'] == 0)

test_pca

sol['Prediction']
len(test_pca)
# Evaluate the first model

rf_pca.best_estimator_.score(test_pca, sol['Prediction'])

randomfpred = rf_pca.predict(test_pca)
confusion_matrix(sol['Prediction'], randomfpred)
roc_auc_score(sol['Prediction'], randomfpred)

#%%
# Feature Engineering
test_new = pd.DataFrame() # New df
#train_new
for i in range(0, max(test.TrialID)+1): # Within each trial
    temp_data = test[test['TrialID'] == i] # Create a temporary df for each trial
    for j in list(test)[3:]: # For all the attributes of each trial
#        print (train_2[train_2['TrialID'] == i][j])
        temp_data['m{}'.format(j)] = test[test['TrialID'] == i][j].rolling(window = 5).mean() # Create the Rolling mean
        temp_data['sd{}'.format(j)] = test[test['TrialID'] == i][j].rolling(window = 5).std() # Create the Rolling Std
    test_new = test_new.append(temp_data)
    
#%%
test_new = test_new.fillna(0) # Missing Value
test_new = test_new.drop(columns = ['TrialID', 'ObsNum', 'IsAlert']) # Drop unneccessary columns

#test_new
#%%
from sklearn.metrics import f1_score

rf_mstd.score(test_new, sol['Prediction'])

randomfpred2 = rf_mstd.predict(test_new)

confusion_matrix(sol['Prediction'], randomfpred2)
roc_auc_score(sol['Prediction'], randomfpred2)
f1_score(sol['Prediction'], randomfpred2)
#%%

nn_mstd.score(test_new, sol['Prediction'])

nnpred = nn_mstd.predict(test_new)

confusion_matrix(sol['Prediction'], nnpred)
roc_auc_score(sol['Prediction'], nnpred)
f1_score(sol['Prediction'], nnpred)
#%%

#rf_mstd.decision_function(test_new)

#ROC curve plotting

from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for each class
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
#for i in range(sol['Prediction']):
#    fpr[i], tpr[i], _ = roc_curve(sol['Prediction'], randomfpred2)
#    roc_auc[i] = auc(fpr[i], tpr[i])
plot_1 = roc_curve(sol['Prediction'], randomfpred2)
plot_2 = roc_curve(sol['Prediction'], nnpred)
plot_3 = roc_curve(sol['Prediction'], randomfpred)
# Compute micro-average ROC curve and ROC area
#fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#%%
plt.figure()
lw = 2
plt.plot(plot_1[0], plot_1[1], color='darkorange',
         lw=lw, label='Random Forest ROC curve (area = {0:0.2f})'
               ''.format(auc(plot_1[0], plot_1[1])))
plt.plot(plot_2[0], plot_2[1], color='red',
         lw=lw, label='Neural Network ROC curve (area = {0:0.2f})'
               ''.format(auc(plot_2[0], plot_2[1])))
#plt.plot(plot_3[0], plot_3[1], color='green',
#         lw=lw, label='Neural Network ROC curve (area = {0:0.2f})'
#               ''.format(auc(plot_3[0], plot_3[1])))
#,label='ROC curve (area = %0.2f)' % auc(sol['Prediction'], nnpred)
#,label='ROC curve (area = %0.2f)' % auc(sol['Prediction'], randomfpred2)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()