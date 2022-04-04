# -*- coding: utf-8 -*-
import cv2
import numpy as np

def on_trackbar(x):
    pass

img = cv2.imread("week7/line_drive/src/ex.jpg")

if img is None:
    print("Image load failed!")
    exit(1)

blur = cv2.GaussianBlur(img, (3, 3), 0)
ycrcb = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)

cv2.imshow("src", img)
cv2.namedWindow("ycrcb")
cv2.createTrackbar("low_Y", "ycrcb", 109, 255, on_trackbar)
cv2.createTrackbar("low_Cr", "ycrcb", 122, 255, on_trackbar)
cv2.createTrackbar("low_Cb", "ycrcb", 115, 255, on_trackbar)

cv2.createTrackbar("high_Y", "ycrcb", 188, 255, on_trackbar)
cv2.createTrackbar("high_Cr", "ycrcb", 142, 255, on_trackbar)
cv2.createTrackbar("high_Cb", "ycrcb", 140, 255, on_trackbar)

while cv2.waitKey(1) != 27:
    low_Y = cv2.getTrackbarPos("low_Y", "ycrcb")
    low_Cr = cv2.getTrackbarPos("low_Cr", "ycrcb")
    low_Cb = cv2.getTrackbarPos("low_Cb", "ycrcb")
    
    high_Y = cv2.getTrackbarPos("high_Y", "ycrcb")
    high_Cr = cv2.getTrackbarPos("high_Cr", "ycrcb")
    high_Cb = cv2.getTrackbarPos("high_Cb", "ycrcb")
    
    low = np.array([low_Y, low_Cr, low_Cb])
    high = np.array([high_Y, high_Cr, high_Cb])
    
    bin = cv2.inRange(ycrcb, low, high)
    cv2.imshow("ycrcb", bin)