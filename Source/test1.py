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


def denoise(frame):
    frame = cv2.medianBlur(frame, 5)
    frame = cv2.GaussianBlur(frame, (11, 11), 0)

    return frame


def train_bg_subtractor(inst, cap, num=500):
    '''
        BG substractor need process some amount of frames to start giving result
    '''
    print('Training BG Subtractor...')
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        inst.apply(frame, None, 0.001)
        i += 1
        if i >= num:
            return cap


def draw_image_histogram(image, channels, color='k'):
    hist = cv2.calcHist([image], channels, None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])


def show_grayscale_histogram(image):
    draw_image_histogram(image, [0])
    plt.show()


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


def hough_lines(image):
    """
    `image` should be the output of a Canny transform.

    Returns hough lines (not the image with lines)
    """
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=25, minLineLength=20, maxLineGap=300)


def draw_lines(image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
    # the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
    if make_copy:
        image = np.copy(image)  # don't want to modify the original
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image


def average_slope_intercept(lines):
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)

    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue  # ignore a vertical line
                if y2 == y1:
                    continue  # ignore a vertical line
                slope = (y2-y1)/(x2-x1)
                intercept = y1 - slope*x1
                length = np.sqrt((y2-y1)**2+(x2-x1)**2)
                anglel = np.arctan2(abs((y2-y1)), abs((x2-x1))) * (180/np.pi)
                if anglel < 20:
                    continue

                if slope < 0:  # y is reversed in image
                    left_lines.append((slope, intercept))
                    left_weights.append((length))
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append((length))
    except:
        pass

    # add more weight to longer lines
    left_lane = np.dot(left_weights,  left_lines) / \
        np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / \
        np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane  # (slope, intercept), (slope, intercept)


def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None

    slope, intercept = line

    # make sure everything is integer as cv2.line requires it

    x1 = (y1 - intercept)//slope
    x2 = (y2 - intercept)//slope
    y1 = (y1)
    y2 = (y2)

    if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
        return None

    return ((x1, y1), (x2, y2))


def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)

    y1 = image.shape[0]  # bottom of the image
    y2 = y1*0.55         # slightly lower than the middle

    left_line = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)
    if left_line is not None and right_line is not None:
        (xl1, yl1), (xl2, yl2) = left_line
        (xr1, yr1), (xr2, yr2) = right_line
        xa, ya = A
        xb, yb = B
        if xr2 - xl2 < (xb-xa - 150):
            left_line = None
            right_lane = None

    if left_line is not None:
        (xl1, yl1), (xl2, yl2) = left_line
        if xl2 > 640:
            left_line = None
    if right_line is not None:
        (xr1, yr1), (xr2, yr2) = right_line
        if xr2 < 600:
            right_line = None
        if xr1 < xr2:
            right_line = None

    return left_line, right_line


def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=20):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)


QUEUE_LENGTH = 50


class LaneDetector:
    def __init__(self):
        self.left_lines = deque(maxlen=QUEUE_LENGTH)
        self.right_lines = deque(maxlen=QUEUE_LENGTH)

    def process(self, image):
        edges = process_image(image)
        regions = region_of_interest(edges, vertices)
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


def undistort(img):
    # Load pickle
    dist_pickle = pickle.load(open("calibration_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # Undistort sample image
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted


def process_image(image):
        # undistort it
    image = undistort(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = denoise(image)

    image = subtract_background(image)

    image = fgbg.apply(image)

    image = region_of_interest(image, vertices)
    # image = denoise(image)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edged = cv2.Canny(gray, 50, 100)

    # # find contours in the image and initialize the mask that will be
    # # used to remove the bad contours
    # (images, cnts, _) = cv2.findContours(
    #     edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # mask = np.ones(image.shape[:2], dtype="uint8") * 255
    # mask2 = np.ones(image.shape[:2], dtype="uint8") * 255
    # # loop over the contours
    # for c in cnts:
    #     # if the contour is bad, draw it on the mask
    #     if is_contour_bad(c):
    #         cv2.drawContours(mask, [c], -1, 0, -1)

    # # remove the contours from the image and show the resulting images
    # image = cv2.bitwise_not(edged, edged, mask=mask)
    # image = cv2.bitwise_not(edged, edged, mask=mask2)
    # image = cv2.Canny(image, 20, 50)

    return image


def is_contour_bad(c):
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # the contour is 'bad' if it is not a rectangle
    return not len(approx) == 4


cap = cv2.VideoCapture("test_videos/test1.mp4")
_, frame = cap.read()
imshape = frame.shape
A = (0.42*imshape[1], 0.53*imshape[0])
B = (0.58*imshape[1], 0.53*imshape[0])
C = (0.82*imshape[1], 0.75*imshape[0])
D = (0.25*imshape[1], 0.75*imshape[0])
vertices = np.array([[B, C, D, A]])
fgbg = cv2.createBackgroundSubtractorMOG2()


while(cap.isOpened()):
    ret, frame = cap.read()
    # processed_img = region_of_interest(frame, vertices)
    # processed_img = process_image(frame)
    detector = LaneDetector()
    processed_img = detector.process(frame)

    cv2.imshow("Screen", processed_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# def process_video(video_input, video_output):
#     detector = LaneDetector()

#     clip = VideoFileClip(os.path.join('test_videos', video_input))
#     processed = clip.fl_image(detector.process)
#     processed.write_videofile(os.path.join(
#         'output_videos', video_output), audio=False)


# process_video('test3.mp4', 'test3.mp4')
