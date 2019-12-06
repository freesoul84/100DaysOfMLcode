#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 22:56:38 2019

@author: alkesha
"""
# import the necessary packages
from imutils.video import VideoStream
import cv2

tracker =cv2.TrackerCSRT_create()
 
bounding_box = None
video = VideoStream(0).start()

while True:
    frame = video.read()
    frame=frame
    if frame is None:
        break    
    (h, w) = frame.shape[:2]
    if bounding_box is not None:
        (t, box) = tracker.update(frame)
 
        if t:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 111, 0), 3)
            
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    if key == ord("c"):
        bounding_box = cv2.selectROI("Frame", frame, fromCenter=False,
                               )
        tracker.init(frame,bounding_box)
        
    elif key == ord("q"):
        break

else:
	video.release()
cv2.destroyAllWindows()
