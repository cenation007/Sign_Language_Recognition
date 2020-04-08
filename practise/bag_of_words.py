#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:46:01 2020

@author: ayush
"""
from sklearn.cluster import MiniBatchKMeans
from image import process
import matplotlib.pyplot as plt
import os
import random
import numpy as np
path= 'train'
img_descs=[]
label =1
y=[]

def train_test_val_split_idxs(total_rows, percent_test, percent_val):
    if percent_test + percent_val >= 1.0:
        raise ValueError('percent_test and percent_val must sum to less than 1.0')

    row_range = range(total_rows)

    no_test_rows = int(total_rows*(percent_test))
    test_idxs = random.sample(row_range,no_test_rows)
    row_range = [idx for idx in row_range if idx not in test_idxs]

    no_val_rows = int(total_rows*(percent_val))
    val_idxs = random.sample(row_range,no_val_rows)
    training_idxs = [idx for idx in row_range if idx not in val_idxs]

    return training_idxs, test_idxs, val_idxs

def cluster_features(img_descs,training_idxs,cluster_model):
    n_clusters = cluster_model.n_clusters
    training_descs = [img_descs[i] for i in training_idxs]
    all_train_descs = [desc for desc_list in training_descs for desc in desc_list]
    all_train_descs = np.array(all_train_descs)
    cluster_model.fit(all_train_descs)
    img_clustered_words = [cluster_model.predict(raw) for raw in img_descs]
    #img_clustered_words = cluster_model.cluster_centers_
    #print(img_clustered_words)
    img_hist = np.array([np.bincount(clustered_words,minlength=n_clusters) for clustered_words in img_clustered_words])
    X=img_hist
    return X, cluster_model
    
def perform_split(X,y,training_ind,test_ind,val_ind):
    return X[training_ind],X[test_ind],X[val_ind],y[training_ind],y[test_ind],y[val_ind]

for (dirpath,dirnames,filenames) in os.walk(path):
    for dirname in dirnames:
        print(dirname)
        for(direcpath,direcnames,files) in os.walk(path+"/"+dirname):
            for file in files:
                actual_path=path+"/"+dirname+"/"+file
                _,des=process(actual_path)
                img_descs.append(des)
                y.append(label)
        label=label+1

training_idxs, test_idxs, val_idxs = train_test_val_split_idxs(len(y),0.3,0.2)

X,cluster_model = cluster_features(img_descs, training_idxs, MiniBatchKMeans(n_clusters=150))

X_train,X_test,X_val,y_train,y_test,y_val = perform_split(X,y,training_idxs,test_idxs,val_idxs)