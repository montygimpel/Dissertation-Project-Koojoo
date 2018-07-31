import numpy as np
import cv2

cap = cv2.VideoCapture('output.mov') #Open video file

fgbg = cv2.createBackgroundSubtractorMOG2() #Create the background substractor

while(cap.isOpened()):
    ret, frame = cap.read() #read a frame
    
    fgmask = fgbg.apply(frame) #Use the substractor

    _, contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # find contours on fgmask

    if len(contours) != 0: 
        cv2.drawContours(frame, contours, -1, (0,255,0), 2) #draw contours on original frame
        contour = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0), 2)
        print(contour[0]) 

    cv2.imshow('Frame',frame)
    #cv2.imshow('Background Substraction',fgmask)
    #print(contour)
    cv2.waitKey(0)

cap.release() #release video file
cv2.destroyAllWindows() #close all openCV windows