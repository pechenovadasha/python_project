import cv2
import numpy as np
from glints_detection_research import find_glint  
# import matplotlib.pyplot as plt
import time
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_WORKERS = 1

def find_two_largest_contours_indices(contours, max_area=1000):
    
    if not contours:
        return None, None
    

    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= max_area:
            filtered_contours.append(contour)
    

    if len(filtered_contours) < 2:
        return None, None
    

    areas = [cv2.contourArea(c) for c in filtered_contours]
    i0, i1 = np.argsort(areas)[-2:]
    
    return i0, i1

def k_means_method(gray):
    # отсееваем черный фон
    pixels = gray[gray > 33].reshape(-1, 1).astype(np.float32)

    if len(pixels) < 2:
        return None

    K = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # centers — это значения яркости для каждого кластера
    centers = centers.flatten()
    sorted_idx = np.argsort(centers)

    # Предположим: серый = 0-й, белый = 1-й
    gray_center = centers[sorted_idx[0]]

    # Определим диапазон вокруг серого центра
    tolerance = 40  
    lower = max(0, gray_center - tolerance)
    upper = min(255, gray_center + tolerance)

    lower = np.array([lower], dtype=np.uint8)
    upper = np.array([upper], dtype=np.uint8)

    # Строим маску по этому диапазону
    mask = cv2.inRange(gray, lower, upper)

    # Морфология для очистки
    th = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    return th



def find_contours(gray, show_result=False, method='kmeans'): #The most danger place
    

    if method == 'kmeans':
        th = k_means_method(gray)
    else:
        _, th = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if show_result == True:
        cv2.imshow("Diff", gray)
        cv2.imshow("Threshold", th)
        cv2.waitKey(0)
    cnts, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if show_result == True:
        gray_plot = gray.copy()
        gray_plot = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(gray_plot, cnts, -1, (0, 0, 255), 1)
        cv2.imshow("findContours", gray_plot)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return cnts


class PupilTracker:


    def __init__(self):


        self.pupilL = (None, None)
        self.pupilR = (None, None)
        self.glintL = (None, None)
        self.glintR = (None, None)

        self.capasity = 2

        self.roi_left_points = np.zeros((self.capasity, 2), dtype=np.float32)  # [x, y]
        self.roi_right_points = np.zeros((self.capasity, 2), dtype=np.float32)
        self.roi_points_idx = 0  
        
        self.roi_coords = []

        self.roi_coords = [None, None]
        self.use_roi = False
        self.recalculation_roi = 100
        

        self.mean_roi_left = 0
        self.mean_roi_right = 0

        self.amount_process = 0

    
    def roi_process(eye_img, x1, y1, is_left):
        cnts = find_contours(eye_img, False, 'kmeans')
        
        if not cnts:
            return (None, None), (None, None), is_left
        
        # Берем самый большой контур, принимаем его в качестве зрачка 
        largest = max(cnts, key=cv2.contourArea)

        # Находим отблески и принимаем близжайший за необходимый. 
        # Данная функция возвращает координаты центра зрачка и отблеска

        glint_and_pupil_center = find_glint(largest, eye_img)
        p_x_roi, p_y_roi, g_x_roi, g_y_roi = glint_and_pupil_center

        # Необходимо данные координаты привести к искомым размерам и запомнить их
        pupil = (None, None)
        glint = (None, None)
        if p_x_roi is not None and p_y_roi is not None:
            pupil = (p_x_roi + x1, p_y_roi + y1)
        else:
            print("Error: left pupil center is none") 
        if g_x_roi is not None and  g_y_roi is not None:    
            glint = (g_x_roi + x1, g_y_roi + y1)
        else:
            print("Error: left glint is none") 
        return pupil, glint, is_left


    def process(self, diff_orig, bright_img, dir_name):

        # параметр для отладки этапа нахождения контуров. Если true то будет включен режим отрисовки 
        show = False

        # инициализация параметров для РОИ
        self.amount_process += 1
        if self.amount_process == self.recalculation_roi: # перезапись Roi каждые n кадров 
            self.amount_process = 0
            
            self.roi_left_points.fill(0)
            self.roi_right_points.fill(0)
            self.roi_points_idx = 0
            self.use_roi = False

        
        search_gray = diff_orig
        if search_gray.ndim == 3:
            search_gray = cv2.cvtColor(search_gray, cv2.COLOR_BGR2GRAY)
        search_gray = cv2.normalize(search_gray, None, 0, 255, cv2.NORM_MINMAX)


        # Если набралось статистики для РОИ, то используем его
        if self.use_roi == True:
            # Получаем координаты области левого и правого глаза (прямоугольник)
            l_x1, l_y1, l_x2, l_y2 = self.roi_coords[0]
            r_x1, r_y1, r_x2, r_y2 = self.roi_coords[1]

            # Обработка глаз происходит в двух потоках для каждого глаза
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Запускаем обработку левого и правого глаза параллельно
                future_left = executor.submit(
                    self.roi_process, 
                    search_gray[l_y1:l_y2, l_x1:l_x2], 
                    l_x1, l_y1, 
                    True
                )
                future_right = executor.submit(
                    self.roi_process,
                    search_gray[r_y1:r_y2, r_x1:r_x2], 
                    r_x1, r_y1, 
                    False
                )

                left_result = future_left.result()
                right_result = future_right.result()

                pupil_left, glint_left, _ = left_result
                pupil_right, glint_right, _ = right_result
                
                self.pupilL, self.glintL = pupil_left, glint_left
                self.pupilR, self.glintR = pupil_right, glint_right
                
        else:
            # Алгоритм без использованиея РОИ. Для него вы используем метод трешхолда
            cnts = find_contours(search_gray,show, 'adaptive')
            if cnts == None:
                print("Error: countours don't find")
                return (None, None), (None, None), (None, None), (None, None)

            cnts_L = cnts_R = cnts

            idxL, idxR = find_two_largest_contours_indices(cnts)

            if idxL is None or idxR is None:
                print("Error: countours don't find")
                return (None, None), (None, None), (None, None), (None, None)


            if (cnts_L[idxL][0][0][0] < cnts_R[idxR][0][0][0]):
                pupils = [cnts_L[idxL], cnts_R[idxR]] 
            else:
                pupils = [cnts_R[idxR], cnts_L[idxL]] 

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_left = executor.submit(find_glint, pupils[0], bright_img)
                future_right = executor.submit(find_glint, pupils[1], bright_img)
                
                find_left = future_left.result()
                find_right = future_right.result()

            self.pupilL = find_left[:2] if find_left is not None else (0, 0)
            self.pupilR = find_right[:2] if find_right is not None else (0, 0)
            self.glintL = find_left[2:] if find_left is not None else (0, 0)
            self.glintR = find_right[2:] if find_right is not None else (0, 0)        

        color = bright_img.copy()

        if self.use_roi == False:
            # Получаем количество заполненных точек
            filled_count = min(self.roi_points_idx, self.capasity)
            
            if filled_count < self.capasity:
                # Быстрое вычисление средних без nan_to_num
                mean_left = np.mean(self.roi_left_points[:filled_count], axis=0)
                mean_right = np.mean(self.roi_right_points[:filled_count], axis=0)
                
                # Проверяем валидность точек
                add_left = (mean_left[0] == 0 and mean_left[1] == 0) or \
                          (self.pupilL[0] is not None and self.pupilL[1] is not None and
                           (1.5 * mean_left[0] > self.pupilL[0] or 
                            1.5 * mean_left[1] > self.pupilL[1]))
                
                add_right = (mean_right[0] == 0 and mean_right[1] == 0) or \
                           (self.pupilR[0] is not None and self.pupilR[1] is not None and
                            (1.5 * mean_right[0] > self.pupilR[0] or 
                             1.5 * mean_right[1] > self.pupilR[0]))
                
                # Добавляем точки в массивы
                if add_left and self.pupilL[0] is not None and self.pupilL[1] is not None:
                    self.roi_left_points[self.roi_points_idx % self.capasity] = [self.pupilL[0], self.pupilL[1]]
                
                if add_right and self.pupilR[0] is not None and self.pupilR[1] is not None:
                    self.roi_right_points[self.roi_points_idx % self.capasity] = [self.pupilR[0], self.pupilR[1]]
                
                if add_left or add_right:
                    self.roi_points_idx += 1
            else:
                # Активируем ROI
                self.use_roi = True

                width = 500
                height = 400

                # Вычисляем средние из заполненных данных
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Определяем функцию для вычисления одного ROI внутри контекста
                    def calc_roi(points):
                        if len(points) == 0:
                            return None
                        x, y = np.mean(points, axis=0)
                        x1 = int(x - width//2)  
                        y1 = int(y - height//2) 
                        x2 = int(x +  width//2)
                        y2 = int(y + height//2)
                        return (x1, y1, x2, y2)
                    

                    future_left = executor.submit(calc_roi, self.roi_left_points[:valid_count])
                    future_right = executor.submit(calc_roi, self.roi_right_points[:valid_count])
                    
                    self.roi_coords = [future_left.result(), future_right.result()]

        # Отрисовка центров
        if 0:
            if (self.pupilL is not None):
                cv2.circle(color, (int(self.pupilL[0]), int(self.pupilL[1])),  2, (0,0,255), -1) 
            if (self.pupilR is not None):    
                cv2.circle(color,  (int(self.pupilR[0]), int(self.pupilR[1])),  2, (0,0,255), -1)
            if (self.glintL is not None):
                cv2.circle(color, (int(self.glintL[0]), int(self.glintL[1])),  2, (0,255,255), -1) 
            if (self.glintR is not None):    
                cv2.circle(color,  (int(self.glintR[0]), int(self.glintR[1])),  2, (0,255,255), -1)
            cv2.imshow("Result", color)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return self.pupilL, self.pupilR, self.glintL, self.glintR

_TRACKER = PupilTracker()

def process_frame(diff, dark_img, bright_img, dir_name):
    return _TRACKER.process(diff, bright_img, dir_name)