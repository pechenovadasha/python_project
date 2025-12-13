import cv2
import numpy as np

def refine_contour(approx_contour, bright_img, expand_px=10, glare_threshold=None, show=False):
    
    if len(bright_img.shape) == 3:
        gray = cv2.cvtColor(bright_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = bright_img.copy()


    x, y, w, h = cv2.boundingRect(approx_contour)
    x1 = max(x - expand_px, 0)
    y1 = max(y - expand_px, 0)
    x2 = min(x + w + expand_px, gray.shape[1])
    y2 = min(y + h + expand_px, gray.shape[0])
    roi = gray[y1:y2, x1:x2]

    # normalized_image = cv2.normalize(roi, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    if show == True:
        cv2.imshow("roi", roi)
        # cv2.waitKey(0)

    mean_intensity = np.mean(roi)

    _, thresh_blik = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)
    result_image = roi.copy()

    if show == True:
        cv2.imshow("thresh_blik", thresh_blik)
        # cv2.waitKey(0)


    # Расширяем маску на 2 пикселя
    kernel = np.ones((3, 3), np.uint8)
    expanded_mask = cv2.dilate(thresh_blik, kernel, iterations=3)

    result_image[expanded_mask == 255] = 0 
    normalized_image = cv2.normalize(result_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # print(thresh_adap)
    _, thresh = cv2.threshold(normalized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    # print(thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    
    if show == True:
        cv2.imshow("thresh", thresh)
        cv2.imshow("Without blink", normalized_image)
        # cv2.waitKey(0)

    # blink_mask = np.zeros_like(normalized_image)
    # cv2.drawContours(blink_mask, contours_blik, -1, 255, -1)
    

    contours_global = []
    for cnt in contours:
        cnt_global = cnt + np.array([[[x1, y1]]], dtype=np.int32)
        cnt_global = cnt_global.astype(np.int32)
        contours_global.append(cnt_global)

    if contours_global:
        largest_contour = max(contours_global, key=cv2.contourArea)
    else:
        largest_contour = None

    img_copy = bright_img.copy()
    if len(img_copy.shape) == 2:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)

    # cv2.drawContours(img_copy, [largest_contour], -1, (0,0,255), 1)

    center = None
    if largest_contour is not None and len(largest_contour) >= 5:  # Для эллипса нужно минимум 5 точек
        ellipse = cv2.fitEllipse(largest_contour)
        (center, axes, angle) = ellipse
        center_x, center_y = center
        # cv2.ellipse(img_copy, ellipse, (0, 255, 255), 1)

        # cv2.circle(img_copy, (int(center_x), int(center_y)),  2, (0,165,255), -1)


    if show == True:
        cv2.imshow("Reference Image", img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return center

def reference_center_detection_method(pupils, bright_img):
    
    # img_copy = bright_img.copy()

    # # если изображение чёрно-белое — переводим в цветное, чтобы рисовать цветными контурами
    # if len(img_copy.shape) == 2:
    #     img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)

    # перебираем контуры
    # for i, cnt in enumerate(pupils):
    #     color = (0, 255, 0) if i == 0 else (0, 255, 0)  # первый зелёный, второй красный
    #     cv2.drawContours(img_copy, [cnt], -1, color, 1)
    
    centers = [refine_contour(cnt, bright_img) for cnt in pupils]
    # for i, cnt in enumerate(refined_pupils):
    #     color = (0, 255, 255) if i == 0 else (0, 255, 255)  # первый зелёный, второй красный
    #     cv2.drawContours(img_copy, [cnt], -1, color, 1)
    
    # cv2.imshow("Contours", img_copy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return centers