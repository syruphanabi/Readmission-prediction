# -*- coding: utf-8 -*-
"""
@author: Shenghua Xiang
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from utils import save_svmlight

#since 137 is my favorite prime number, and it's related to fine structure constant
RANDOM_STATE = 137

#read readmission records and non-readmission records 
one_set = pd.read_csv("data/one_set_processed.csv",header=None)
zero_set = pd.read_csv("data/zero_set_processed.csv",header=None)

#concatenate the texts and the labels
whole_text = pd.concat([one_set.iloc[:,1],zero_set.iloc[:,1]],ignore_index=True)
whole_label = pd.concat([one_set.iloc[:,0],zero_set.iloc[:,0]],ignore_index=True)

#using TF-IDF Vectorizer the transform texts into matrix. X is a 9693*46300 matrix
vectorizer = TfidfVectorizer(max_df=0.5,min_df=2, stop_words='english')
X = vectorizer.fit_transform(whole_text)

#use univariate feature selection to select 4000 features that are best related to the label
X_reduced = SelectKBest(chi2, k=4000).fit_transform(X, whole_label)

X_reduced2 = SelectKBest(chi2,k=2000).fit_transform(X,whole_label)
#save the univariate selected features
save_svmlight(X_reduced,whole_label,'Uni')
save_svmlight(X_reduced2, whole_label,'Uni2000')




#