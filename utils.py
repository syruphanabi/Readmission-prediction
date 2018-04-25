# -*- coding: utf-8 -*-
"""
@author: Shenghua Xiang
"""
import numpy as np
from sklearn.metrics import *
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
from scipy import interp

#this script contains utility funtions


#functions from homework1
def classification_metrics(Y_pred, Y_true):
	#NOTE: It is important to provide the output in the same order
    acc = accuracy_score(Y_pred,Y_true)
    auc_ = roc_auc_score(Y_pred,Y_true)
    precision = precision_score(Y_pred,Y_true)
    recall = recall_score(Y_pred,Y_true)
    f1score = f1_score(Y_pred,Y_true)
    return acc,auc_,precision,recall,f1score

def show_metrics(classifierName,Y_pred,Y_true):
    print "______________________________________________"
    print "Classifier: "+classifierName
    acc, auc_, precision, recall, f1score = classification_metrics(Y_pred,Y_true)
    print "Accuracy: "+str(acc)
    print "AUC: "+str(auc_)
    print "Precision: "+str(precision)
    print "Recall: "+str(recall)
    print "F1-score: "+str(f1score)
    print "______________________________________________"
    print ""
    return acc,auc_,precision,recall,f1score
    

#create three svmlight files that stores train set, test set and the whole set separately
def save_svmlight(X,y,save_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 137, shuffle = True)
    #save train and test separately into svmlight format
    
    wholeset = open('features/'+save_name+'.whole', 'wb')
    trainset = open('features/'+save_name+'.train', 'wb')
    testset = open('features/'+save_name+'.test','wb')
    dump_svmlight_file(X_train,y_train,trainset)
    dump_svmlight_file(X_test,y_test,testset)
    dump_svmlight_file(X,y,wholeset)    
    trainset.close()
    testset.close()
    wholeset.close()


#this function draws ROC curve for averaged 10 repeated CV and test.
def draw_roc(X,y,X_train,y_train,X_test,y_test,reg):
    #plot for average 10-fold cv
    base_fpr = np.linspace(0, 1, 100)  
    tprs = []
    aucs = []
    cv = StratifiedKFold(n_splits=10)
    for train, test in cv.split(X, y):
        y_score = reg.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], y_score[:, 1])
        tprs.append(interp(base_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        
    tprs = np.array(tprs)
    #take the average of the 10 repeated trials
    mean_tprs = tprs.mean(axis = 0)
      
    plt.plot([0, 1], [0, 1], linestyle='--',label='Luck')
    
    #plot 10-fold cross-validation ROC
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    plt.plot(base_fpr, mean_tprs, color='b',linestyle = '--',
         label=r'10-fold CV (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
    
    #plot test ROC
    y_pre = reg.fit(X_train, y_train).predict_proba(X_test)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pre[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr,tpr,linewidth = 1.2,color = 'r',label = 'testing (AUC = %0.2f )' % (roc_auc), alpha = 0.8)
  
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()
    