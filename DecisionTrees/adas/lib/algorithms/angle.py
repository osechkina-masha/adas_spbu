from typing import List
import numpy as np
import numpy.ma as ma
import cv2 as cv


def get_cluster(angle_v, angles: List[float]):
    cluster = 0
    for i, angle in enumerate(angles):
        if angle_v > angle:
            cluster = i + 1
        else:
            break
    return cluster


def scale_img(arr):
    mask = arr == 0
    arr = ma.masked_array(arr, mask)
    new_arr = ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')
    return np.array(new_arr)


def get_angle_lines(img,
                    horizon: float,
                    step: int,
                    min_size: int,
                    angles: List[float],
                    grad_thr: float):
    start_height = int(horizon) + 1
    segmented = np.zeros_like(img)

    grad_x = (img[start_height + step:-step:, 2 * step::] - img[start_height + step:-step, :-2 * step:]) / (2 * step)
    grad_y = (img[start_height + 2 * step::, step:-step:] - img[start_height:-2 * step:, step:-step:]) / (2 * step)

    grad_thr_mask = np.sqrt(grad_x ** 2 + grad_y ** 2) > grad_thr
    grad_angles = np.arctan2(grad_y, grad_x)

    for j in range(grad_x.shape[0]):
        for i in range(grad_x.shape[1]):
            if not grad_thr_mask[j][i]:
                continue
            segmented[start_height + step + j][step + i] = get_cluster(grad_angles[j][i], angles)
    
    segmented = scale_img(segmented)
    contours, _ = cv.findContours(segmented, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    final_image = np.zeros_like(img)

    for c in contours:
        if len(c) < min_size:
            continue
        for i in range(1, len(c)):
            point, point_prev = c[i][0], c[i - 1][0]
            cv.line(final_image, point_prev, point, (255, 255, 255), 1)
    return final_image
