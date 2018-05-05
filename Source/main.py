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

cap = cv2.VideoCapture("test_videos/test2.flv")
_, frame = cap.read()
imshape = frame.shape
A = (0.42*imshape[1], 0.53*imshape[0])
B = (0.58*imshape[1], 0.53*imshape[0])
C = (0.82*imshape[1], 0.75*imshape[0])
D = (0.25*imshape[1], 0.75*imshape[0])
vertices = np.array([[B, C, D, A]])

while(cap.isOpened()):
    ret, frame = cap.read()

    detector = LaneDetector()
    processed_img = detector.process(frame, vertices)

    cv2.imshow("Screen", processed_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
