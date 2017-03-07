import tensorflow as tf
import numpy as np
import cv2
import argparse
import os
import time
import threading

# from model import model_spec
from model import yolo9000
from ops import process_logits
from utils import *
from production import *
from benchmarks import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_classes', 9418, 'num classes')
tf.app.flags.DEFINE_integer('num_bboxes', 3, 'num bboxes')
# tf.app.flags.DEFINE_integer('num_classes', 80, 'num classes')
# tf.app.flags.DEFINE_integer('num_bboxes', 5, 'num bboxes')
tf.app.flags.DEFINE_float('prob_thresh', 0.15, 'prob threshold for scale_prob * cls_prob')
tf.app.flags.DEFINE_float('iou_thresh', 0.8, 'iou threshold for NMS')
tf.app.flags.DEFINE_bool('load_binary', False, 'load binary or not')
tf.app.flags.DEFINE_bool('save', False, 'save weights or not')
tf.app.flags.DEFINE_bool('load_pb', False, 'load pb or not')
tf.app.flags.DEFINE_string('names', '9k.names', 'class indexes to name')
tf.app.flags.DEFINE_string('read_weights_path', '', 'weights directory to read from')
tf.app.flags.DEFINE_string('image_path', '', 'image path')
tf.app.flags.DEFINE_string('pb_path', '', 'pb path')
FLAGS.idx_to_txt = parse_names(FLAGS.names)


def main(argv):
    if FLAGS.load_pb:
        with tf.gfile.GFile(FLAGS.pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def)
            inp = graph.get_tensor_by_name('import/input:0')
            bboxes = graph.get_tensor_by_name('import/bboxes:0')
            probabilities = graph.get_tensor_by_name('import/probabilities:0')
        sess = tf.Session(graph=graph)
    else:
        with tf.Graph().as_default() as graph:
            inp = tf.placeholder(tf.float32, shape=[1, 416, 416, 3], name='input')
            model = tf.make_template('model', yolo9000)
            logits = model(inp)
            bboxes, probabilities = process_logits(logits)
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver()
        sess = tf.Session(graph=graph)
        if FLAGS.load_binary:
            sess.run(init_op)
            with graph.as_default():
                load_from_binary(sess)
        else:
            saver.restore(sess, os.path.join(os.getcwd(), 'tf-yolo9000-weights', 'yolo9000-model.ckpt'))
        if FLAGS.save:
            saver.save(sess, os.path.join(os.getcwd(), 'tf-yolo9000-weights', 'yolo9000-model.ckpt'))

    if argv[1] == 'eval_one_image':
        img_in = cv2.imread(FLAGS.image_path).astype(np.float32)
        eval_one_image(sess, img_in, inp, bboxes, probabilities)

    elif argv[1] == 'eval_one_image_x_times':
        x_times = int(argv[2])
        img_in = cv2.imread(FLAGS.image_path).astype(np.float32)
        eval_one_image_x_times(sess, img_in, x_times, inp, bboxes, probabilities)

    elif argv[1] == 'eval_one_image_x_times_threaded':
        x_times = int(argv[2])
        img_in = cv2.imread(FLAGS.image_path).astype(np.float32)
        eval_one_image_x_times_threaded(sess, img_in, x_times, inp, bboxes, probabilities)

    elif argv[1] == 'get_pb':
        get_pb()

if __name__ == '__main__':
    tf.app.run()



