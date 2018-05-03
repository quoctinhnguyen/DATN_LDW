# -*- coding: utf-8 -*-

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


def plot_images(original, modified, title):
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(original)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(modified, cmap='gray')
    ax2.set_title(title, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def undistort(img):
    # Load pickle
    dist_pickle = pickle.load(open("calibration_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # Undistort sample image
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted


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
    src = np.float32([(0.42*imshape[1], 0.54*imshape[0]), (0.59*imshape[1], 0.54*imshape[0]), (0.79*imshape[1],
                                                                                               0.75*imshape[0]), (0.26*imshape[1], 0.75*imshape[0])])
    dst = np.float32([[xoffset, yoffset], [img_size[0]-xoffset, yoffset],
                      [img_size[0]-xoffset, img_size[1]-yoffset],
                      [xoffset, img_size[1]-yoffset]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return warped, M


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobelx = np.absolute(sobelx)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return sxbinary
# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    magnitude = np.sqrt(sobelx*sobelx + sobely*sobely)
    # 5) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*magnitude/np.max(magnitude))
    # 6) Create a binary mask where mag thresholds are met
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= mag_thresh[0]) &
             (scaled_sobel <= mag_thresh[1])] = 1
    # 7) Return this mask as your binary_output image
    return sxbinary
# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    sbinary = np.zeros_like(direction)
    sbinary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return sbinary
# Combined different thresholding techniques


def combined_thresh(img):
    # Choose a Sobel kernel size
    ksize = 21

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(
        img, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(
        img, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(img, sobel_kernel=7, mag_thresh=(50, 100))
    dir_binary = dir_threshold(img, sobel_kernel=15, thresh=(0.4, 1.3))

    # Combine them
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) | (grady == 1)) & (
        (mag_binary == 1) | (dir_binary == 1))] = 1
    return combined

# Edit this function to create your own pipeline.


def color_thresh(img, s_thresh=(10, 200), l_thresh=(115, 255)):
    img = np.copy(img)
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Threshold x gradient and color
    color_gradient_binary = np.zeros_like(s_channel)
    color_gradient_binary[((s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])) & (
        (l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1]))] = 255
    # a = cv2.bitwise_and(img, img, mask=color_gradient_binary)
    print(s_channel.shape)
    return color_gradient_binary


# Edit this function to create your own pipeline.


def color_gradient_thresh(img, s_thresh=(10, 200), l_thresh=(115, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient and color
    color_gradient_binary = np.zeros_like(s_channel)
    color_gradient_binary[((s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])) & ((l_channel >= l_thresh[0]) & (
        l_channel <= l_thresh[1])) | ((scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1]))] = 1
    return color_gradient_binary


def detect_edges(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)


def process_image(image):
    # undistort it
    img = undistort(image)

    img = region_of_interest(frame, vertices)
    # img2 = region_of_interest(frame2, vertices)
    # img3 = region_of_interest(frame3, vertices)

    smooth = apply_smoothing(img, 11)
    # smooth2 = apply_smoothing(img2, 11)
    # smooth3 = apply_smoothing(img3, 11)

    canny = cv2.Canny(smooth, 20, 50)
    # canny2 = cv2.Canny(smooth2, 20, 50)
    # canny3 = cv2.Canny(smooth3, 20, 50)

    img = increate_contrast(img)
    # img2 = increate_contrast(img2)
    # img3 = increate_contrast(img3)

    img = select_white_yellow(img)
    # img2 = select_white_yellow(img2)
    # img3 = select_white_yellow(img3)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

    img = cv2.bitwise_and(canny, img)
    # img2 = cv2.bitwise_and(canny2, img2)
    # img3 = cv2.bitwise_and(canny3, img3)
    # Apply perspective transform
    img, M = perspective_transform(img, mtx, dist, isColor=False)
    # slice1Copy = np.uint8(img)
    # imake = cv2.bitwise_and(image, image, mask=slice1Copy)
    # e = detect_edges(imake)
    return img, M


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


# Main
cap = cv2.VideoCapture("test_videos/test1.mp4")
cap2 = cv2.VideoCapture("test_videos/test2.mp4")
cap3 = cv2.VideoCapture("test_videos/test3.mp4")
_, frame = cap.read()
imshape = frame.shape
# vertices = np.array([[(0.59*imshape[1], 0.54*imshape[0]), (0.79*imshape[1],
#                                                            0.75*imshape[0]), (0.26*imshape[1], 0.75*imshape[0]), (0.42*imshape[1], 0.54*imshape[0])]])
vertices = np.array([[(imshape[1], 0.5*imshape[0]), (imshape[1],
                                                     imshape[0]), (0, imshape[0]), (0, 0.5*imshape[0])]])
# Load pickle
dist_pickle = pickle.load(open("calibration_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]


def increate_contrast(image):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    # convert from BGR to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2, a, b))  # merge channels
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR


def select_white_yellow(image):
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # white color mask
    lower = np.uint8([10, 100, 0])
    upper = np.uint8([50, 255, 255])
    mask0 = cv2.inRange(converted, lower, upper)
    # white color mask
    lower = np.uint8([100, 100, 0])
    upper = np.uint8([120, 200, 255])
    mask1 = cv2.inRange(converted, lower, upper)
    # # white color mask
    # lower = np.uint8([0, 80, 0])
    # upper = np.uint8([10, 200, 255])
    # mask0 = cv2.inRange(converted, lower, upper)
    # # white color mask
    # lower = np.uint8([15, 0, 0])
    # upper = np.uint8([30, 200, 255])
    # mask1 = cv2.inRange(converted, lower, upper)
    # # yellow color mask
    # lower = np.uint8([30, 80, 0])
    # upper = np.uint8([90, 200, 255])
    # mask2 = cv2.inRange(converted, lower, upper)
    # # yellow color mask
    # lower = np.uint8([90, 100, 0])
    # upper = np.uint8([180, 200, 255])
    # mask3 = cv2.inRange(converted, lower, upper)
    # # combine the mask
    # mask4 = cv2.bitwise_or(mask0, mask1)
    # mask5 = cv2.bitwise_or(mask2, mask4)
    # mask = cv2.bitwise_or(mask3, mask5)
    mask = cv2.bitwise_or(mask0, mask1)
    return cv2.bitwise_and(image, image, mask=mask)


def apply_smoothing(image, kernel_size=15):
    """
    kernel_size must be postivie and odd
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


while(cap.isOpened()):
    ret, frame = cap.read()
    # ret2, frame2 = cap2.read()
    # ret3, frame3 = cap3.read()

    img = region_of_interest(frame, vertices)
    # img2 = region_of_interest(frame2, vertices)
    # img3 = region_of_interest(frame3, vertices)

    smooth = apply_smoothing(img, 21)
    # smooth2 = apply_smoothing(img2, 11)
    # smooth3 = apply_smoothing(img3, 11)

    canny = cv2.Canny(smooth, 10, 25)
    # canny2 = cv2.Canny(smooth2, 20, 50)
    # canny3 = cv2.Canny(smooth3, 20, 50)

    img = increate_contrast(smooth)
    # img2 = increate_contrast(img2)
    # img3 = increate_contrast(img3)

    img = select_white_yellow(img)
    # img2 = select_white_yellow(img2)
    # img3 = select_white_yellow(img3)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

    img = cv2.bitwise_and(canny, img)
    # img2 = cv2.bitwise_and(canny2, img2)
    # img3 = cv2.bitwise_and(canny3, img3)

    # processed_img, M = process_image(frame)
    # left_curverad, right_curverad, left_fitx, right_fitx, yvals, right_peak, left_peak = getCurvatureForLanes(
    #     processed_img, [], [], [], [])

    # # Plot the two lines
    # plt.xlim(0, 1280)
    # plt.ylim(0, 720)
    # plt.plot(left_fitx, yvals, color='green', linewidth=3)
    # plt.plot(right_fitx, yvals, color='green', linewidth=3)
    # plt.gca().invert_yaxis()  # to visualize as we do the images

    # result = drawLane(processed_img, M, frame, left_fitx, right_fitx, yvals)
    # img = region_of_interest(frame, vertices)
    # lane = lane_marking_detection(roi)
    # gray = np.max(lane, axis=-1, keepdims=1)
    # edge = detect_edges(img)
    # smooth = apply_smoothing(edge, 5)
    # edge = detect_edges(result)
    # dist = cv2.distanceTransform(
    #     src=mag_combined, distanceType=cv2.DIST_L2, maskSize=5)

    # canny_edge = cv2.Canny(frame, 20, 100)
    # cv2.imshow('Video_Test1', img)
    # cv2.imshow('Video_Test1', img)
    # img2, M = perspective_transform(img2, mtx, dist, isColor=False)

    cv2.imshow('Video_Test1', img)
    # cv2.imshow('Video_Test2', img2)
    # cv2.imshow('Video_Test3', img3)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
