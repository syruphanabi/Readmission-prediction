# -*- coding: utf-8 -*-
"""
@author: Shenghua Xiang
"""


from sklearn.datasets import load_svmlight_file
import parameter_tuning
import run_models
import ffnet_model

#select which features to use

#which_features = 'Combined'
#which_features = 'Uni'
which_features = 'word2vec2000'
n = 2000

data_train = load_svmlight_file('features/'+which_features+'.train',n_features= n)
data_test = load_svmlight_file('features/'+which_features+'.test',n_features = n)
data = load_svmlight_file('features/'+which_features+'.whole',n_features= n)

X_train = data_train[0]
y_train = data_train[1]

X_test = data_test[0]
y_test = data_test[1]

X = data[0]
y = data[1]

#parameter tuning using logistic regression, SVM, and Random Forest, and return the best params
svm_params = parameter_tuning.SVM_tune(X_train,y_train)
rf_params = parameter_tuning.random_forest_tune(X_train,y_train)
C = parameter_tuning.logistic_regression_plot(X_train, y_train)

#train the model with the best params, display metrics and plot the ROC curve 
svm = run_models.svm(X,y,X_train,y_train,X_test,y_test,svm_params)
random_forest = run_models.random_forest(X,y,X_train,y_train,X_test,y_test,rf_params)
lr = run_models.logistic_regression(X,y,X_train,y_train,X_test,y_test,C)

#train feed forward nerual network with given params, and plot the ROC curve
params = {'input_size':n, 'hidden_unit':250}
ffnet_model.FfNet_draw_roc(X.toarray(),y,X_train.toarray(),y_train,X_test.toarray(),y_test,params)








