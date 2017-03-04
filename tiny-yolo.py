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

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_classes', 80, 'num classes')
tf.app.flags.DEFINE_integer('num_bboxes', 5, 'num bboxes')
tf.app.flags.DEFINE_float('prob_thresh', 0.25, 'prob threshold for scale_prob * cls_prob')
tf.app.flags.DEFINE_float('iou_thresh', 0.8, 'iou threshold for NMS')
tf.app.flags.DEFINE_bool('load_binary', False, 'load binary or not')
tf.app.flags.DEFINE_bool('save', False, 'save weights or not')
tf.app.flags.DEFINE_bool('load_pb', False, 'load pb or not')
tf.app.flags.DEFINE_string('names', 'coco.names', 'class indexes to name')
tf.app.flags.DEFINE_string('read_weights_path', '', 'weights directory to read from')
tf.app.flags.DEFINE_string('image_path', '', 'image path')
tf.app.flags.DEFINE_string('pb_path', '', 'pb path')
FLAGS.idx_to_txt = parse_names(FLAGS.names)

def eval_one_image(sess, img, image_ph, bboxes, probabilities):
    orig = img
    img_in = preprocess_image(img)

    nn_start = time.time()
    bboxes_val, probabilities_val = sess.run([bboxes, probabilities], feed_dict={image_ph:img_in})
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
            model = tf.make_template('model', model_spec)
            logits = model(inp)
            bboxes, probabilities = process_logits(logits)
            saver = tf.train.Saver()
        sess = tf.Session(graph=graph)
        if FLAGS.load_binary:
            sess.run(tf.global_variables_initializer())
            load_from_binary(sess)
        else:
            saver.restore(sess, os.path.join(os.getcwd(), 'tf-weights', 'tiny-yolo-model.ckpt'))
        if FLAGS.save:
            saver.save(sess, os.path.join(os.getcwd(), 'tf-weights', 'tiny-yolo-model.ckpt'))

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



