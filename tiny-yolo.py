import tensorflow as tf
import numpy as np
import cv2
import argparse
import os
import time

from model import model_spec
from ops import process_logits
from utils import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_classes', 80, 'num classes')
tf.app.flags.DEFINE_integer('num_bboxes', 5, 'num bboxes')
tf.app.flags.DEFINE_float('prob_thresh', 0.25, 'prob threshold for scale_prob * cls_prob')
tf.app.flags.DEFINE_float('iou_thresh', 0.8, 'iou threshold for NMS')
tf.app.flags.DEFINE_bool('load_binary', False, 'load binary or not')
tf.app.flags.DEFINE_bool('save', False, 'save weights or not')
tf.app.flags.DEFINE_string('names', 'coco.names', 'class indexes to name')
tf.app.flags.DEFINE_string('read_weights_dir', '', 'weights directory to read from')
tf.app.flags.DEFINE_string('image_path', '', 'image path')
FLAGS.idx_to_txt = parse_names(FLAGS.names)

def eval_one_image(sess, img):
    orig = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (416, 416))
    img = np.expand_dims(img, 0)
    img = img / 255.0
    image_ph = tf.get_default_graph().get_tensor_by_name('input:0')
    bboxes = tf.get_default_graph().get_tensor_by_name('bboxes:0')
    probabilities = tf.get_default_graph().get_tensor_by_name('probabilities:0')

    nn_start = time.time()
    bboxes_val, probabilities_val = sess.run([bboxes, probabilities], feed_dict={image_ph:img})
    nn_end = time.time()

    nms_start = time.time()
    bboxes_out, classes = nms(bboxes_val, probabilities_val, FLAGS.iou_thresh)
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

def eval_one_image_x_times(sess, img_in, x_times):
    '''
    All single image processing. No batching involved.

    ##################
    ### TensorFlow ###
    ##################
    GTX 1060 6GB
    1: 1.99424117824 FPS
    10: 16.3980344115 FPS
    100: 61.4526265944 FPS
    1000: 82.4555698622 FPS
    10000: 87.7515871075 FPS

    CPU
    1: 8.84767865468 FPS
    100: 9.94553707402 FPS

    ##################
    #### Darknet #####
    ##################
    NOTE These tests might be trivialized. Simply ran a for loop X times in
    line 485 in detector.c.

    GTX 1060 6GB
    1: 138 FPS
    10: 143 FPS
    100: 150 FPS
    1000: 153 FPS

    CPU
    1: 0.472595376 FPS
    10: 0.047612055 FPS
    100: 0.471698113 FPS

    '''
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
    img_in = cv2.resize(img_in, (416, 416))
    img_in = np.expand_dims(img_in, 0)
    img_in = img_in / 255.0
    image_ph = tf.get_default_graph().get_tensor_by_name('input:0')
    bboxes = tf.get_default_graph().get_tensor_by_name('bboxes:0')
    probabilities = tf.get_default_graph().get_tensor_by_name('probabilities:0')
    total = 0
    for i in xrange(x_times):
        start = time.time()
        bboxes_val, probabilities_val = sess.run([bboxes, probabilities], feed_dict={image_ph:img_in})
        end = time.time()
        total += (end - start)
    print 'Read from: ', FLAGS.image_path
    print 'FPS: ', 1 / (total / float(x_times))

def main(argv):
    with tf.Graph().as_default() as g:
        image = tf.placeholder(tf.float32, shape=[1, 416, 416, 3], name='input')
        model = tf.make_template('model', model_spec)
        logits = model(image)
        bboxes, probabilities = process_logits(logits)

    with tf.Session(graph=g) as sess:
        saver = tf.train.Saver()
        if FLAGS.load_binary:
            sess.run(tf.global_variables_initializer())
            load_from_binary(sess)
        else:
            saver.restore(sess, os.path.join(os.getcwd(), 'tf-weights', 'tiny-yolo-model.ckpt'))

        if argv[1] == 'eval_one_image':
            img_in = cv2.imread(FLAGS.image_path).astype(np.float32)
            eval_one_image(sess, img_in)

        elif argv[1] == 'eval_one_image_x_times':
            x_times = int(argv[2])
            img_in = cv2.imread(FLAGS.image_path).astype(np.float32)
            eval_one_image_x_times(sess, img_in, x_times)

        if FLAGS.save:
            saver.save(sess, os.path.join(os.getcwd(), 'tf-weights', 'tiny-yolo-model.ckpt'))

if __name__ == '__main__':
    tf.app.run()



