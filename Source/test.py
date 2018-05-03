import cv2
import numpy as np

img = cv2.imread('test_images/roi1_0.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


edges = cv2.Canny(gray, 100, 200, apertureSize=3)
cv2.imshow('edges', edges)
cv2.waitKey(0)

minLineLength = 20
maxLineGap = 20
lines = cv2.HoughLinesP(edges, 1, np.pi/90, 25, minLineLength, maxLineGap)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('hough', img)
cv2.waitKey(0)
