import tensorflow as tf
import numpy as np
import cv2
import argparse
import os
import time
import threading

from model import model_spec
from ops import process_logits
from utils import *
from production import *
from benchmarks import *

def eval_one_image(sess, img, image_ph, bboxes, probabilities):
    orig = img
    img_in = preprocess_image(img)

    nn_start = time.time()
    bboxes_val, probabilities_val = sess.run([bboxes, probabilities], feed_dict={image_ph:img_in})
    nn_end = time.time()

    nms_start = time.time()
    bboxes_out, classes = nms(bboxes_val, probabilities_val, FLAGS.iou_thresh)
    print classes
    nms_end = time.time()

    dboxes_start = time.time()
    draw_boxes(orig, bboxes_out, classes, FLAGS.idx_to_txt)
    dboxes_end = time.time()

    image_path_out = FLAGS.image_path.split('.')
    image_path_out[0] += '_out'
    image_path_out = '.'.join(image_path_out)
    cv2.imwrite(image_path_out, orig)
    print 'Read from: ', FLAGS.image_path
    print 'Wrote to: ', image_path_out
    print 'Neural Net Time: ', nn_end - nn_start
    print 'NMS Time: ', nms_end - nms_start
    print 'Draw Boxes Time: ', dboxes_end - dboxes_start

def eval_one_image_x_times(sess, img_in, x_times, image_ph, bboxes, probabilities):
    img_in = preprocess_image(img_in)
    total = 0
    for i in xrange(x_times):
        start = time.time()
        bboxes_val, probabilities_val = sess.run([bboxes, probabilities], feed_dict={image_ph:img_in})
        end = time.time()
        total += (end - start)
    print 'Read from: ', FLAGS.image_path
    print 'FPS: ', 1 / (total / float(x_times))

def eval_one_image_x_times_threaded(sess, img_in, x_times, image_ph, bboxes, probabilities):
    threads = []
    num_threads = 4
    img_in = preprocess_image(img_in)
    def threaded_eval(thread_idx):
        for i in xrange(x_times/num_threads):
            bboxes_val, probabilities_val = sess.run([bboxes, probabilities], feed_dict={image_ph:img_in})
    start = time.time()
    for i in xrange(num_threads):
        t = threading.Thread(target=threaded_eval, args=(i,))
        t.daemon = True
        t.start()
        threads.append(t)
    [t.join() for t in threads]
    end = time.time()
    print 'Read from: ', FLAGS.image_path
    print 'FPS: ', 1 / ((end-start) / float(x_times))
