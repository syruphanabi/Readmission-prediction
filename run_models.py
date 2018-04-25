# -*- coding: utf-8 -*-
"""
@author: Shenghua Xiang
"""

import pandas as pd
import numpy as np
from scipy import interp
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from utils import show_metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
from itertools import cycle
from utils import draw_roc
from sklearn.ensemble import RandomForestClassifier


def logistic_regression(X,y,X_train,y_train,X_test,y_test,params):
    reg = LogisticRegression(C = params)
    reg.fit (X_train,y_train)
    y_pre = reg.predict(X_test)
    metrics = show_metrics('Logistic Regression',y_test,y_pre)
    draw_roc(X,y,X_train,y_train,X_test,y_test,reg)
    
    return metrics

def svm(X,y,X_train,y_train,X_test,y_test,params):
    reg = SVC(params)
    reg.fit (X_train,y_train)
    y_pre = reg.predict(X_test)
    metrics = show_metrics('SVM',y_test,y_pre)
    draw_roc(X,y,X_train,y_train,X_test,y_test,reg)
    
    return metrics

def random_forest(X,y,X_train,y_train,X_test,y_test,params):
    reg = RandomForestClassifier(n_estimators = params['n_estimators'],max_depth = params['max_depth'])
    reg.fit(X_train,y_train)
    y_pre = reg.predict(X_test)
    metrics = show_metrics('Random Forest',y_test,y_pre)
    draw_roc(X,y,X_train,y_train,X_test,y_test,reg)   
    return metrics

