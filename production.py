import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tiny_yolo import model_spec
from ops import process_logits
import time

FLAGS = tf.app.flags.FLAGS

def get_pb():
    with tf.Graph().as_default() as logits_graph:
        inp = tf.placeholder(tf.float32, shape=[1, 416, 416, 3], name='input')
        model = tf.make_template('model', model_spec)
        logits = model(inp)
        bboxes, probabilities = process_logits(logits)

    with tf.Session(graph=logits_graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.read_weights_path)
        output_graph_def = graph_util.convert_variables_to_constants(sess,
                logits_graph.as_graph_def(),
                [bboxes.name[:-2],
                probabilities.name[:-2]])
        with gfile.GFile('tiny-yolo.pb', 'wb') as f:
            f.write(output_graph_def.SerializeToString())





