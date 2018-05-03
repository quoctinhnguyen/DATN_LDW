# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 20:03:49 2018

@author: tuank
"""
import cv2


def capture_video(dir_video, number_image=3):
    cap = cv2.VideoCapture(dir_video)
    i = 0
    while(cap.isOpened() and i < number_image):
        ret, frame = cap.read()
        cv2.imwrite("test_images/frame3_%d.jpg" % i, frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()
