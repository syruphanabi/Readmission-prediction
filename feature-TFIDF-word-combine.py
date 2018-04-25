# -*- coding: utf-8 -*-
"""
@author: Shenghua Xiang
"""

from sklearn.datasets import load_svmlight_file
import scipy.sparse as sp
import numpy as np
import utils

#this script combines features from TF-IDF features and word2vec

word2vec = load_svmlight_file('features/word2vec2000.text',n_features=2000)
word2vec_mat = word2vec[0].todense()
word2vec_label = word2vec[1]

uni = load_svmlight_file('features/Uni2000.whole',n_features =2000 )
uni_mat = uni[0].todense()


c = np.concatenate((uni_mat,word2vec_mat), axis=1)
utils.save_svmlight(c,word2vec_label,'Combined')



