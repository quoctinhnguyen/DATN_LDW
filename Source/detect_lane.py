import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import glob
import numpy as np
import pickle
from numpy.linalg import inv
from scipy.signal import argrelextrema
from moviepy.editor import VideoFileClip
from collections import deque
from scipy import ndimage


def undistort(img):
    # Load pickle
    dist_pickle = pickle.load(open("calibration_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # Undistort sample image
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted


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


def perspective_transform(img, mtx, dist, isColor=True):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    imshape = img.shape
    if(isColor):
        # Convert undistorted image to grayscale
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    else:
        gray = undist

    xoffset = 0  # offset for dst points
    yoffset = 0
    img_size = (undist.shape[1], undist.shape[0])
    src = np.float32([A, B, C, D])
    dst = np.float32([[xoffset, yoffset], [img_size[0]-xoffset, yoffset],
                      [img_size[0]-xoffset, img_size[1]-yoffset],
                      [xoffset, img_size[1]-yoffset]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return warped, M


def increase_contrast(image):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=-2, tileGridSize=(8, 8))
    clahe2 = cv2.createCLAHE(clipLimit=10, tileGridSize=(8, 8))
    # convert from BGR to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    s2 = clahe2.apply(s)
    lab = cv2.merge((h, l2, s2))  # merge channels
    return cv2.cvtColor(lab, cv2.COLOR_HLS2BGR)  # convert from LAB to BGR


def apply_smoothing(image, kernel_size=15):
    """
    kernel_size must be postivie and odd
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def filter(image):
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # color mask 1
    # lower = np.uint8([0, 210, 0])
    # upper = np.uint8([30, 255, 150])
    lower = np.uint8([0, 0, 0])
    upper = np.uint8([0, 0, 0])
    mask0 = cv2.inRange(converted, lower, upper)
    # color mask 2
    lower = np.uint8([0, 210, 0])
    upper = np.uint8([50, 255, 255])
    mask1 = cv2.inRange(converted, lower, upper)

    mask = cv2.bitwise_or(mask0, mask1)
    return cv2.bitwise_and(image, image, mask=mask)


# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720  # meters per pixel in y dimension
xm_per_pix = 3.7/700  # meters per pixel in x dimension


def getCurvatureForLanes(processed_img, prev_left_fitx, prev_right_fitx, prev_left_peak, prev_right_peak):
    yvals = []
    leftx = []
    rightx = []
    imageHeight = processed_img.shape[0]
    imageWidth = processed_img.shape[1]
    bufferForDecidingByDistanceFromMid = 10

    left_histogram = np.sum(
        processed_img[(imageHeight//4):, :(imageWidth//2)], axis=0)
    right_histogram = np.sum(
        processed_img[(imageHeight//4):, (imageWidth//2):], axis=0)

    # get local maxima
    starting_left_peak = np.argmax(left_histogram)
    leftx.append(starting_left_peak)

    starting_right_peak = np.argmax(right_histogram)
    rightx.append(starting_right_peak + imageWidth//2)

    curH = imageHeight
    yvals.append(curH)
    increment = 25
    columnWidth = 150
    leftI = 0
    rightI = 0
    while (curH - increment >= imageHeight//4):
        curH = curH - increment
        leftCenter = leftx[leftI]
        leftI += 1
        rightCenter = rightx[rightI]
        rightI += 1

        # calculate left and right index of each column
        leftColumnL = max((leftCenter - columnWidth//2), 0)
        rightColumnL = min((leftCenter + columnWidth//2), imageWidth)

        leftColumnR = max((rightCenter - columnWidth//2), 0)
        rightColumnR = min((rightCenter + columnWidth//2), imageWidth)

        # imageHeight/2 - (imageHeight - curH)
        leftHistogram = np.sum(
            processed_img[curH - increment:curH, leftColumnL:rightColumnL], axis=0)
        rightHistogram = np.sum(
            processed_img[curH - increment:curH, leftColumnR:rightColumnR], axis=0)

        left_peak = np.argmax(leftHistogram)
        right_peak = np.argmax(rightHistogram)
        if(left_peak):
            leftx.append(left_peak+leftColumnL)
        else:
            leftx.append(leftx[leftI-1])

        if(right_peak):
            rightx.append(right_peak+leftColumnR)
        else:
            rightx.append(rightx[rightI-1])
        yvals.append(curH)

    yvals = np.array(yvals)
    rightx = np.array(rightx)
    leftx = np.array(leftx)

    # Determine the fit in real space
    left_fit_cr = np.polyfit(yvals*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(yvals*ym_per_pix, rightx*xm_per_pix, 2)

    # Define y-value where we want radius of curvature
    y_eval = np.max(yvals)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) \
        / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) \
        / np.absolute(2*right_fit_cr[0])

    # Fit a second order polynomial to each lane line
    left_fit = np.polyfit(yvals, leftx, 2)
    left_fitx = left_fit[0]*yvals**2 + left_fit[1]*yvals + left_fit[2]
    right_fit = np.polyfit(yvals, rightx, 2)
    right_fitx = right_fit[0]*yvals**2 + right_fit[1]*yvals + right_fit[2]

    return left_curverad, right_curverad, left_fitx, right_fitx, yvals, starting_right_peak, starting_left_peak


def drawLane(warped, M, undist, left_fitx, right_fitx, yvals):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = inv(M)
    newwarp = cv2.warpPerspective(
        color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)


def hough_lines(image):
    """
    `image` should be the output of a Canny transform.

    Returns hough lines (not the image with lines)
    """
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)


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
                slope = (y2-y1)/(x2-x1)
                intercept = y1 - slope*x1
                length = np.sqrt((y2-y1)**2+(x2-x1)**2)
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

    return ((x1, y1), (x2, y2))


def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)

    y1 = image.shape[0]  # bottom of the image
    y2 = y1*0.55         # slightly lower than the middle

    left_line = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)

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

        def mean_line(line, lines):
            if line is not None:
                lines.append(line)

            if len(lines) > 0:
                line = np.mean(lines, axis=0, dtype=np.int32)
                # make sure it's tuples not numpy array for cv2.line to work
                line = tuple(map(tuple, line))
            return line

        left_line = mean_line(left_line,  self.left_lines)
        print("Left")
        print(left_line)
        right_line = mean_line(right_line, self.right_lines)
        print("right")
        print(right_line)
        return draw_lane_lines(image, (left_line, right_line))


def process_image(image):
        # undistort it
    image = undistort(image)

    increased = increase_contrast(image)

    gray = cv2.cvtColor(increased, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    smooth2 = cv2.GaussianBlur(gray2, (7, 7), 0)

    lower = np.uint8([1])
    upper = np.uint8([1])

    ret, thresh2 = cv2.threshold(smooth, 240, 255, cv2.THRESH_BINARY)

    mask = ndimage.binary_opening(
        thresh2, structure=np.ones((3, 3))).astype(int)
    # canny = cv2.Canny(mask, 20, 50)
    # canny = region_of_interest(canny, vertices)
    mask = cv2.inRange(mask, lower, upper)
    print(mask)
    # image = cv2.bitwise_and(mask, canny)
    # image = cv2.Canny(image, 20, 100)
    # perspective_transform
    # img, M =
    # gray
    # gray = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
    # perspective_transform(img, mtx, dist, isColor=True)

    # increase_contrast
    # contrast = increase_contrast(img)

    # e = region_of_interest(contrast, vertices)

    # retval, contrast = cv2.threshold(contrast, 200, 255, cv2.THRESH_BINARY)

    # contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)

    # smooth

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # filter
    # filted = filter(contrast)
    # filted = cv2.cvtColor(filted, cv2.COLOR_BGR2GRAY)
    # edge
    # canny = cv2.Canny(smooth, 20, 40)

    # merge = cv2.bitwise_and(canny, filted)

    # re = region_of_interest(merge, vertices)

    # perspective_t
    # ransform

    return mask



# Main
cap = cv2.VideoCapture("test_videos/test1.mp4")
# cap2 = cv2.VideoCapture("test_videos/test2.mp4")
# cap3 = cv2.VideoCapture("test_videos/test3.mp4")
_, frame = cap.read()
imshape = frame.shape
A = (0.39*imshape[1], 0.58*imshape[0])
B = (0.64*imshape[1], 0.58*imshape[0])
C = (0.8*imshape[1], 0.75*imshape[0])
D = (0.28*imshape[1], 0.75*imshape[0])

vertices = np.array([[B, C, D, A]])
# Load pickle
dist_pickle = pickle.load(open("calibration_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
# fgbg = cv2.createBackgroundSubtractorMOG2()

while(cap.isOpened()):
    ret, frame = cap.read()
    # ret2, frame2 = cap2.read()
    # ret3, frame3 = cap3.read()

    processed_img = process_image(frame)
    # processed_img2 = process_image(frame2)
    # processed_img3 = process_image(frame3)

    # detector = LaneDetector()
    # processed_img = detector.process(frame)
    cv2.imshow('Video_Test1', processed_img)
    # cv2.imshow('Video_Test2', processed_img2)
    # cv2.imshow('Video_Test3', processed_img3)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
