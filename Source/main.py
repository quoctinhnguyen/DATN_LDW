# -*- coding: utf-8 -*-

import os
import logging
import logging.handlers
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob
import numpy as np
import pickle
from numpy.linalg import inv
from scipy.signal import argrelextrema
from moviepy.editor import VideoFileClip
from collections import deque
from scipy import ndimage

from detect_lane import *

cap = cv2.VideoCapture("test_videos/test2.wmv")
_, frame = cap.read()
imshape = frame.shape
A = (0.42*imshape[1], 0.53*imshape[0])
B = (0.58*imshape[1], 0.53*imshape[0])
C = (0.82*imshape[1], 0.75*imshape[0])
D = (0.25*imshape[1], 0.75*imshape[0])
vertices = np.array([[B, C, D, A]])

cap.set(cv2.CAP_PROP_FPS, 10)

# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if int(major_ver) < 3:
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    print(
        "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(
        "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
if __name__ == '__main__':
    while(cap.isOpened()):
        ret, frame = cap.read()

        detector = LaneDetector()
        processed_img = detector.process(frame, vertices)

        cv2.imshow("Screen", processed_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
