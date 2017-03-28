import tensorflow as tf
import numpy as np
import cv2
from nms import *

FLAGS = tf.app.flags.FLAGS

def draw_boxes(img, bboxes, classes, idx_to_txt):
    h, w, _ = img.shape
    for i, box in enumerate(bboxes):
        scale_img = [h, w, h, w]
        box = [int(a*b) for a,b in zip(box, scale_img)]
        draw_box(img, classes[i], idx_to_txt, box)

def draw_box(img, cls, idx_to_txt, box):
    hsv = np.array([[[int(cls/float(len(idx_to_txt))*255), 255, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0, :]
    bgr = [int(i) for i in bgr]
    text = idx_to_txt[cls]
    cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), bgr, 2)
    cv2.putText(img, text, (box[1], box[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, bgr)

def parse_names(filename):
    f = open(filename, 'r')
    dic = {}
    for idx, line in enumerate(f):
        dic[idx] = line.strip()
    return dic

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (416, 416))
    img = np.expand_dims(img, 0)
    img = img / 255.0
    return img
