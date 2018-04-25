#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 13:40:30 2018

@author: syrup
"""

import numpy as np
import pandas as pd
from textblob import TextBlob
import gensim.models
from utils import save_svmlight


p_set = pd.read_csv("data/one_set_processed.csv", header = None)
n_set = pd.read_csv("data/zero_set_processed.csv", header = None)

text = pd.concat([p_set.iloc[:,1],n_set.iloc[:,1]], ignore_index = True)
label = pd.concat([p_set.iloc[:,0],n_set.iloc[:,0]], ignore_index = True)  


processed_text = []

for index,x in text.iteritems():
    blob = TextBlob(x)
    words = [unicode(x) for x in blob if not any(c.isdigit() for c in x)]
    processed_text.append(words)
    
model = gensim.models.Word2Vec(processed_text, size=2000, window=7, min_count=1, workers=4)
model.train(processed_text,total_examples=len(processed_text),epochs=10)


def word_vec_sum(text):
    word_vecs = [model.wv[word] for word in text]
    wvsum = list(sum(word_vecs))
    temp = [x*x for x in wvsum]
    temp = wvsum / np.sqrt(sum(temp))
    #print sum([x*x for x in temp])
    return list(temp)

X = [word_vec_sum(x) for x in processed_text]

save_svmlight(X,label,'word2vec2000')

