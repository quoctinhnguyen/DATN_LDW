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

fgbg = cv2.createBackgroundSubtractorMOG2()


def preprocess_image(image):
    # undistort it
    # image = undistort(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = denoise(image)

    image = subtract_background(image)

    image = fgbg.apply(image)

    return image


def denoise(frame):
    frame = cv2.medianBlur(frame, 5)
    frame = cv2.GaussianBlur(frame, (11, 11), 0)

    return frame


def undistort(img):
    # Load pickle
    dist_pickle = pickle.load(open("calibration_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # Undistort sample image
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted


def subtract_background(image):
    lower = np.uint8([160])
    upper = np.uint8([200])

    mask1 = cv2.inRange(image, lower, upper)

    lower = np.uint8([160])
    upper = np.uint8([200])

    mask2 = cv2.inRange(image, lower, upper)

    imaged = cv2.bitwise_or(mask1, mask2)

    imaged = cv2.subtract(image, imaged)

    canny = cv2.Canny(image, 20, 40)

    imaged = cv2.bitwise_and(imaged, canny)

    return imaged


def region_of_interest(img, vertices):
    # defining a blank mask to start with
    mask = np.zeros_like(img, dtype=np.uint8)
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, np.array([vertices], dtype=np.int32), ignore_mask_color)
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
