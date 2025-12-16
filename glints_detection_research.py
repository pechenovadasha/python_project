import os
import cv2
# import matplotlib.pyplot as plt
import numpy as np

import time

def get_contour_center(contour):

    if len(contour) == 0:
        return 0, 0  
    

    contour = np.array(contour).reshape(-1, 2)
    

    if len(contour) == 1:
        return contour[0][0], contour[0][1]
    

    M = cv2.moments(contour)
    if M["m00"] == 0:

        center_x = np.mean(contour[:, 0])
        center_y = np.mean(contour[:, 1])
        return center_x, center_y
    else:
       
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY


def find_glint(pupil_contour, image, thr_glint: int = 200):
   
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    pupil_x, pupil_y = get_contour_center(pupil_contour)

    if pupil_x == 0 or pupil_y == 0:
        return None, None, None, None
   
    nearest_x, nearest_y = None, None
    # Контур меньше 5 пикселей не определится
    if pupil_x < 5 or pupil_y < 5 or \
       pupil_x > gray.shape[1]-5 or pupil_y > gray.shape[0]-5:
        return  pupil_x, pupil_y, nearest_x, nearest_y


    _, th = cv2.threshold(gray, thr_glint, 255, cv2.THRESH_BINARY)


    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pupil_x, pupil_y, nearest_x, nearest_y


    min_dist = float("inf")
    #  Находим ближайший отблеск
    for cnt in contours:
        cx, cy = get_contour_center(cnt)
        dist = np.hypot(cx - pupil_x, cy - pupil_y)
        if dist < min_dist:
            min_dist = dist
            nearest_x, nearest_y = cx, cy

   
    return pupil_x, pupil_y, nearest_x, nearest_y
