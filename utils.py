import numpy as np
import cv2
from skimage import measure

def max_conn_mask(P, origin_height, origin_width):
    h, w = P.shape[0], P.shape[1]
    highlight = np.zeros(P.shape)
    for i in range(h):
        for j in range(w):
            if P[i][j] > 0:
                highlight[i][j] = 1

    # 寻找最大的全联通分量
    labels = measure.label(highlight, neighbors=4, background=0)
    props = measure.regionprops(labels)
    max_index = 0
    for i in range(len(props)):
        if props[i].area > props[max_index].area:
            max_index = i
    max_prop = props[max_index]
    highlights_conn = np.zeros(highlight.shape)
    for each in max_prop.coords:
        highlights_conn[each[0]][each[1]] = 1

    # 最近邻插值：
    highlight_big = cv2.resize(highlights_conn,
                               (origin_width, origin_height),
                               interpolation=cv2.INTER_NEAREST)

    highlight_big = np.array(highlight_big, dtype=np.uint16).reshape(1, origin_height, origin_width)
    # highlight_3 = np.concatenate((np.zeros((2, origin_height, origin_width), dtype=np.uint16), highlight_big * 255), axis=0)
    return highlight_big


def get_bboxes(bin_img):
    img = np.squeeze(bin_img.copy().astype(np.uint8), axis=(0,))

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for c in contours:
        # find bounding box coordinates
        # 现计算出一个简单的边界框
        # c = np.squeeze(c, axis=(1,))
        rect = cv2.boundingRect(c)
        bboxes.append(rect)

    return bboxes