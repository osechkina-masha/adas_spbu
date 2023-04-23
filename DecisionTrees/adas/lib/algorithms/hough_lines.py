import cv2 as cv
import numpy as np


def hough_lines(img,
                k=5,
                thr1=150,
                thr2=200,
                roi_width_upper=0.3,
                roi_width_lower=.0,
                min_line_length=40,
                max_line_gap=5,
                hough_thr1=2,
                hough_thr2=np.pi / 180,
                vote_thr=100):
    if k // 2 == 0:
        k += 1

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (k, k), 0)
    img = cv.Canny(img, thr1, thr2)

    height = img.shape[0]
    width = img.shape[1]
    polygons = np.array([[
        (roi_width_lower * width, height),
        ((1 - roi_width_lower) * width, height),
        (int((0.5 + roi_width_upper / 2) * width), int(0.5 * height)),
        (int((0.5 - roi_width_upper / 2) * width), int(0.5 * height))
    ]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv.fillPoly(mask, polygons, 255)
    img = cv.bitwise_and(img, mask)

    lines = cv.HoughLinesP(img, hough_thr1, hough_thr2, vote_thr, np.array([]),
                           minLineLength=min_line_length, maxLineGap=max_line_gap)
    line_image = np.zeros_like(img)
    if lines is None:
        return line_image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 1)

    return line_image