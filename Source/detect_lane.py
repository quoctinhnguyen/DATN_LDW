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

from preprocessing_image import *
from hough_transform import *

QUEUE_LENGTH = 50


class LaneDetector:
    def __init__(self):
        self.left_lines = deque(maxlen=QUEUE_LENGTH)
        self.right_lines = deque(maxlen=QUEUE_LENGTH)

    def process(self, image, vertices):
        image_processed = preprocess_image(image)
        regions = region_of_interest(image_processed, vertices)
        lines = hough_lines(regions)
        left_line, right_line = lane_lines(image, lines)
        font = cv2.FONT_HERSHEY_SIMPLEX

        def mean_line(line, lines):
            if line is not None:
                lines.append(line)

            if len(lines) > 0:
                line = np.mean(lines, axis=0, dtype=np.int32)
                # make sure it's tuples not numpy array for cv2.line to work
                line = tuple(map(tuple, line))
            return line

        left_line = mean_line(left_line,  self.left_lines)
        # print("Left")
        # print(left_line)
        right_line = mean_line(right_line, self.right_lines)
        # print("right")
        # print(right_line)
        if left_line is not None and right_line is not None:
            (xl1, yl1), (xl2, yl2) = left_line
            (xr1, yr1), (xr2, yr2) = right_line
            al = yl1 - yl2
            bl = xl2 - xl1
            ar = yr1 - yr2
            br = xr1 - xr2
            dlx, dly = xl2 - xl1, yl1 - yl2
            drx, dry = xr1 - xr2, yr1 - yr2
            anglel = np.arctan2(dly, dlx) * (180/np.pi)
            angler = np.arctan2(dry, drx) * (180/np.pi)
            if anglel > angler:
                cv2.putText(image, 'Left', (10, 500), font, 4,
                            (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, 'Right', (10, 500), font, 4,
                            (255, 255, 255), 2, cv2.LINE_AA)
        elif left_line is None and right_line is not None:
            cv2.putText(image, 'Right', (10, 500), font, 4,
                        (255, 255, 255), 2, cv2.LINE_AA)
        elif right_line is None and left_line is not None:
            cv2.putText(image, 'Left', (10, 500), font, 4,
                        (255, 255, 255), 2, cv2.LINE_AA)
        else:
            print("error")

        return draw_lane_lines(image, (left_line, right_line))
