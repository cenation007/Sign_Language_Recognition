#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:10:24 2020

@author: ayush
"""

import cv2 as cv
import numpy as np

def process(path):
    img = cv.imread(path)
    img = cv.resize(img,(50,50))
    img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    min_y = np.array([0,40,30],np.uint8)
    max_y = np.array([43,255,254],np.uint8)
    mask = cv.inRange(img,min_y,max_y)
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    mask = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel)
    lower = np.array([170,80,30],np.uint8)
    upper = np.array([180,255,250],np.uint8)
    mask2 = cv.inRange(img,lower,upper)
    
    final_mask = cv.addWeighted(mask,0.5,mask2,0.5,0.0)
    
    frame_skin = cv.bitwise_and(img,img,mask = final_mask)
    img = cv.addWeighted(img,1.5,frame_skin,-0.5,0)
    frame_skin = cv.bitwise_and(img,img,mask = final_mask)
    frame = cv.cvtColor(frame_skin,cv.COLOR_HSV2BGR)
    frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    _,contours,_ = cv.findContours(frame,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    largest_contour = 0
    index = 1
    while index < len(contours):
        if(cv.contourArea(contours[index])>cv.contourArea(contours[largest_contour])):
            largest_contour = index
        index += 1
    #cv.drawContours(frame,contours,largest_contour,(255,255,255),-1)
    contour_dimensions = cv.boundingRect(contours[largest_contour])
    contour_perimeter_x, contour_perimeter_y, contour_perimeter_width, contour_perimeter_height = contour_dimensions
    #new_image = cv.rectangle(frame,(contour_perimeter_x,contour_perimeter_y),
                            #(contour_perimeter_x+contour_perimeter_width,
                            #contour_perimeter_y+contour_perimeter_height),
                            #(255,255,0),8)
    square_side = max(contour_perimeter_x, contour_perimeter_height) - 1
    height_half = (contour_perimeter_y + contour_perimeter_y +
                   contour_perimeter_height) / 2
    width_half = (contour_perimeter_x + contour_perimeter_x +
                  contour_perimeter_width) / 2
    height_min, height_max = height_half - \
        square_side / 2, height_half + square_side / 2
    width_min, width_max = width_half - square_side / 2, width_half + square_side / 2

    #if (height_min >= 0 and height_min < height_max and width_min >= 0 and width_min < width_max):
    #    frame = frame[int(height_min):int(height_max), int(width_min):int(width_max)]
        
    img2 = cv.Canny(frame,200,200)
    #img2 = cv.resize(img2,(256,256))
    #img2 = cv.resize(img2,(96,96))
    orb = cv.KAZE_create()
    kp, des = orb.detectAndCompute(img2,None)
    #img2 = cv.drawKeypoints(img2,kp,None,color=(0,255,0), flags=0)
    #img2 = cv.resize(img2,(256,256))
    return kp,des




def process1(img):
    img = cv.resize(img,(50,50))
    img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    min_y = np.array([0,40,30],np.uint8)
    max_y = np.array([43,255,254],np.uint8)
    mask = cv.inRange(img,min_y,max_y)
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    mask = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel)
    lower = np.array([170,80,30],np.uint8)
    upper = np.array([180,255,250],np.uint8)
    mask2 = cv.inRange(img,lower,upper)
    
    final_mask = cv.addWeighted(mask,0.5,mask2,0.5,0.0)
    
    frame_skin = cv.bitwise_and(img,img,mask = final_mask)
    img = cv.addWeighted(img,1.5,frame_skin,-0.5,0)
    frame_skin = cv.bitwise_and(img,img,mask = final_mask)
    frame = cv.cvtColor(frame_skin,cv.COLOR_HSV2BGR)
    frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    _,contours,_ = cv.findContours(frame,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    largest_contour = 0
    index = 1
    while index < len(contours):
        if(cv.contourArea(contours[index])>cv.contourArea(contours[largest_contour])):
            largest_contour = index
        index += 1
    #cv.drawContours(frame,contours,largest_contour,(255,255,255),-1)
    #contour_dimensions = cv.boundingRect(contours[largest_contour])
    #contour_perimeter_x, contour_perimeter_y, contour_perimeter_width, contour_perimeter_height = contour_dimensions
    #new_image = cv.rectangle(frame,(contour_perimeter_x,contour_perimeter_y),
                            #(contour_perimeter_x+contour_perimeter_width,
                            #contour_perimeter_y+contour_perimeter_height),
                            #(255,255,0),8)
    #square_side = max(contour_perimeter_x, contour_perimeter_height) - 1
    #height_half = (contour_perimeter_y + contour_perimeter_y +
     #              contour_perimeter_height) / 2
    #width_half = (contour_perimeter_x + contour_perimeter_x +
    #              contour_perimeter_width) / 2
    #height_min, height_max = height_half - \
     #   square_side / 2, height_half + square_side / 2
    #width_min, width_max = width_half - square_side / 2, width_half + square_side / 2

    #if (height_min >= 0 and height_min < height_max and width_min >= 0 and width_min < width_max):
    #    frame = frame[int(height_min):int(height_max), int(width_min):int(width_max)]
        
    img2 = cv.Canny(frame,200,200)
    #img2 = cv.resize(img2,(256,256))
    #img2 = cv.resize(img2,(96,96))
    orb = cv.KAZE_create()
    kp, des = orb.detectAndCompute(img2,None)
    img2 = cv.drawKeypoints(img2,kp,None,color=(0,255,0), flags=0)
     
    #img2 = cv.resize(img2,(256,256))
    return kp,des