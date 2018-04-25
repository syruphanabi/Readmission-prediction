#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 18:06:37 2018

@author: syrup


Selecting Discharge summary note only
"""

import pandas as pd



# pick up 'Discharge summary'
discsum = pd.DataFrame()

chunksize = 10**4
for chunk in pd.read_csv("../data/note/NOTEEVENTS.csv", chunksize = chunksize):
    temp = chunk.loc[chunk["CATEGORY"] == 'Discharge summary']
    discsum = pd.concat([discsum,temp])


# add label to note data
discsum = discsum.ix[:,['SUBJECT_ID','HADM_ID','CHARTDATE','TEXT']]

readmission = pd.read_csv("../data/readmission.csv")
readmission = readmission.ix[:,['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','V20']]

positive = readmission.loc[readmission['V20'] == 1]
negative = readmission.loc[readmission['V20'] == 0]

p_set = pd.merge(positive, discsum, how = 'inner', on = ['SUBJECT_ID','HADM_ID'])
n_set = pd.merge(negative, discsum, how = 'inner', on = ['SUBJECT_ID','HADM_ID'])


# sava as csv
p_set.to_csv("../data/p_set.csv")
n_set.to_csv("../data/n_set.csv")
