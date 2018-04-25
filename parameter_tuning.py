# -*- coding: utf-8 -*-
"""
@author: Shenghua Xiang
"""

import numpy as np
from sklearn import linear_model
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#this script contains functions to tune parameters



#use gridsearchcv to tune parameters
def SVM_tune(X_train,y_train):
    C = [0.1,0.01,1]
    gamma_range = [0.001,0.01,0.1,1]
    #alpha = np.logspace(,3,num=10)
    parameters = {'kernel':['linear','rbf'], 'C':C,'gamma':gamma_range}
    svc = SVC()
    clf = GridSearchCV(svc, parameters,cv = 10)
    clf.fit(X_train, y_train)
    print clf.best_params_
    return clf.best_params_

def random_forest_tune(X_train,y_train):
    param_ran = {'n_estimators':(10,100,500,1000),'max_depth' : (None,4,5,6)}
    ran_forest = RandomForestClassifier()
    ran_gscv = GridSearchCV(ran_forest,param_ran,cv=10)
    ran_gscv.fit(X_train,y_train)
    print ran_gscv.best_params_
    return ran_gscv.best_params_

#use holdoutset to tune a single parameter C and plot image of C-Accuracy on holdout set
def logistic_regression_plot(X_train, y_train):
    #split data into training set and holdout set
    X_train2, X_holdout, y_train2, y_holdout = train_test_split(X_train, y_train, test_size=0.3, random_state= 137, shuffle = True)
    C = np.logspace(-3,3,num=30)
    auc_train = []
    auc_holdout = []
    for val in C:
        model = LogisticRegression(C = val)
        model.fit(X_train2,y_train2)
        y_pre_train = model.predict(X_train2)
        y_pre_holdout = model.predict(X_holdout)
        auc_train.append(accuracy_score(y_train2,y_pre_train))
        auc_holdout.append(accuracy_score(y_holdout,y_pre_holdout))
        
    C2 = C[np.argmax(auc_holdout)]
    train_scores_mean = np.mean(auc_train)
    train_scores_std = np.std(auc_train)
    test_scores_mean = np.mean(auc_holdout)
    test_scores_std = np.std(auc_holdout)
    
    plt.title("Accuracy with Logistic Regression")
    plt.xlabel("$C$")
    plt.ylabel("accuracy")
    plt.ylim(0.0, 1)
    lw = 2
    plt.semilogx(C, auc_train, label="Training",
             color="darkorange", lw=lw)
    plt.fill_between(C, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
    plt.semilogx(C, auc_holdout, label="Validation",
             color="navy", lw=lw)
    plt.fill_between(C, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1,
                 color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
    return C2
    

