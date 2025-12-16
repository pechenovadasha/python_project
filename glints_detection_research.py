import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
import time

@njit(nogil=True, cache=True, fastmath=True)
def get_contour_center(contour):
    n = len(contour)
    if n == 0:
        return 0.0, 0.0
    
    if n <= 10:
        sum_x = 0.0
        sum_y = 0.0
        for i in range(n):
            sum_x += contour[i][0][0]
            sum_y += contour[i][0][1]
        return sum_x / n, sum_y / n
    
    m00 = 0.0
    m10 = 0.0
    m01 = 0.0
    
    for i in range(n):
        x = float(contour[i][0][0])
        y = float(contour[i][0][1])
        m00 += 1.0
        m10 += x
        m01 += y
    
    if abs(m00) < 1e-6:
        return 0.0, 0.0
    
    return m10 / m00, m01 / m00

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
