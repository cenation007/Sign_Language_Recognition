#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 23:34:41 2020

@author: ayush
"""

import numpy as np
import cv2
from image import process1
from bag_of_words import cluster_model,svc 

cap = cv2.VideoCapture(0)
label = ['jp','a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
while(True):
    ret, frame = cap.read()
    cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)
    frame=frame[100:300,100:300]
    kp,desc = process1(frame)
    reduced_clusters = cluster_model.predict(desc)
    testing = np.array([np.bincount(reduced_clusters,minlength=150)])
    predicted = svc.predict(testing)
    i=int(predicted[0])
    cv2.putText(frame,label[i],(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()