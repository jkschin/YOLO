import tensorflow as tf
import numpy as np
import cv2
import argparse
import os
import time

import tiny_yolo
import nms
import ops
import utils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_classes', 80, 'num classes')
tf.app.flags.DEFINE_integer('num_bboxes', 5, 'num bboxes')
tf.app.flags.DEFINE_float('prob_thresh', 0.20, 'prob threshold for scale_prob * cls_prob')
tf.app.flags.DEFINE_float('iou_thresh', 0.8, 'iou threshold for NMS')

tf.app.flags.DEFINE_bool('load_pb', False, 'load pb or not')
tf.app.flags.DEFINE_string('pb_path', '', 'pb path')
tf.app.flags.DEFINE_string('weights_path', os.path.join(os.getcwd(),
    'tf-weights', 'tiny-yolo-model.ckpt'), 'weights directory to read from')

tf.app.flags.DEFINE_string('names', os.path.join(os.getcwd(), 'names',
    'coco.names'), 'class indexes to name')
tf.app.flags.DEFINE_string('image_path', os.path.join(os.getcwd(), 'images',
    'sample.jpg'), 'image path')
FLAGS.idx_to_txt = utils.parse_names(FLAGS.names)

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
            model = tf.make_template('model', tiny_yolo.model_spec)
            logits = model(inp)
            bboxes, probabilities = ops.process_logits(logits)
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver()
        sess = tf.Session(graph=graph)
        saver.restore(sess, FLAGS.weights_path)

    if argv[1] == 'image':
        img_in = cv2.imread(FLAGS.image_path).astype(np.float32)
        img_pr = utils.preprocess_image(img_in)
        bboxes_val, probabilities_val = sess.run([bboxes, probabilities],
            feed_dict={inp: img_pr})
        bboxes_out, classes = nms.nms(bboxes_val, probabilities_val,
            FLAGS.iou_thresh)
        utils.draw_boxes(img_in, bboxes_out, classes, FLAGS.idx_to_txt)
        image_path_out = FLAGS.image_path.split('.')
        image_path_out[0] += '_out'
        image_path_out = '.'.join(image_path_out)
        cv2.imwrite(image_path_out, img_in)
        print 'Read from: ', FLAGS.image_path
        print 'Wrote to: ', image_path_out

    elif argv[1] == 'get_pb':
        get_pb()

if __name__ == '__main__':
    tf.app.run()



