#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 08:21:19 2018

@author: alkesha
"""

import cv2
import numpy as np

#function to draw sketch
def sketching(image):
    grey_image= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    grey_blur=cv2.GaussianBlur(grey_image,(5,5),0)
    #grey_blur=cv2.bilateralFilter(grey_image,9,75,75)
    canny=cv2.Canny(grey_blur,10,50)
    ret,frame=cv2.threshold(canny,65,255,cv2.THRESH_BINARY_INV)
    return frame 

camera=cv2.VideoCapture(0)
while True:
    ret,frame=camera.read()
    cv2.imshow("sketch live",sketching(frame))
    if cv2.waitKey(1)==ord('q'):
        break
    
camera.release()
camera.destroyAllWindows()