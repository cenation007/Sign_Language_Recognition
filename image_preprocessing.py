#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 00:47:37 2020

@author: ayush
"""

import numpy as np
import cv2
import os 
import csv
path = "train"
a=[]



for i in range(9216):
    a.append("pixel"+str(i))

with open('train70.csv','w') as csvfile:
    writer = csv.DictWriter(csvfile,fieldnames=a)
    writer.writeheader()
with open('train30.csv','w') as csvfile:
    writer = csv.DictWriter(csvfile,fieldnames=a)
    writer.writeheader()
label = 0

with open('train70.csv','a') as csvfile:
    spamwriter = csv.writer(csvfile)
    with open('train30.csv','a') as file:
        writer = csv.writer(file)
        for (dirpath,dirnames,filenames) in os.walk(path):
            for dirname in dirnames:
                print(dirname)
                for(dirp,dirn,files) in os.walk(path+"/"+dirname):
                    amount = 0.7*len(files)
                    i=0
                    for file in files:
                        file_path = path+'/'+dirname+'/'+file
                        img=cv2.imread(file_path);
                        img=cv2.resize(img,(96,96))
                        flattened = img.flatten()
                        line=[label]+np.array(flattened).tolist()
                        if i<amount:
                            spamwriter.writerow(line)
                        else:
                            writer.writerow(line)
                        i+=1
                label+=1
                        
                    
        
