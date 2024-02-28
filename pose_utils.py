"""
@author: Yanzuo Lu
@author: oliveryanzuolu@gmail.com
"""

import json
import logging
import cv2

import numpy as np

logger = logging.getLogger()

BONES = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
         [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
         [0,15], [15,17]]

JOINT_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

BONE_COLORS = [[153, 0, 0], [153, 51, 0], [153, 102, 0], [153, 153, 0], [102, 153, 0], [51, 153, 0], [0, 153, 0], [0, 153, 51],
               [0, 153, 102], [0, 153, 153], [0, 102, 153], [0, 51, 153], [0, 0, 153], [51, 0, 153], [102, 0, 153],
               [153, 0, 153], [153, 0, 102]]

def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)


def cords_to_map(cords, img_size, old_size=(128, 64), affine_matrix=None, sigma=6):
    old_size = img_size if old_size is None else old_size
    cords = cords.astype(float)
    result = np.zeros(img_size + cords.shape[0:1], dtype='float32')
    for i, point in enumerate(cords):
        if point[0] == -1 or point[1] == -1:
            continue
        point[0] = point[0]/old_size[0] * img_size[0]
        point[1] = point[1]/old_size[1] * img_size[1]
        if affine_matrix is not None:
            point_ =np.dot(affine_matrix, np.matrix([point[1], point[0], 1]).reshape(3,1))
            point_0 = int(point_[1])
            point_1 = int(point_[0])
        else:
            point_0 = int(point[0])
            point_1 = int(point[1])
        xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
        result[..., i] = np.exp(-((yy - point_0) ** 2 + (xx - point_1) ** 2) / (2 * sigma ** 2))
    return result


def draw_pose_from_cords(array, img_size, old_size=(128, 64), radius=2, draw_bones=True):
    colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)
    scale_y = img_size[0] / old_size[0]
    scale_x = img_size[1] / old_size[1]

    if draw_bones:
        for i, (f, t) in enumerate(BONES):
            from_missing = array[f][0] == -1 or array[f][1] == -1
            to_missing = array[t][0] == -1 or array[t][1] == -1
            if from_missing or to_missing:
                continue
            cv2.line(colors, (int(array[f][1] * scale_x), int(array[f][0] * scale_y)),
                     (int(array[t][1] * scale_x), int(array[t][0] * scale_y)), BONE_COLORS[i], radius, cv2.LINE_AA)

    for i, joint in enumerate(array):
        if array[i][0] == -1 or array[i][1] == -1:
            continue
        cv2.circle(colors, (int(joint[1] * scale_x), int(joint[0] * scale_y)), radius + 1, JOINT_COLORS[i], -1, cv2.LINE_AA)

    return colors