import tensorflow as tf
from ops import *
from tensorflow.contrib.framework.python.ops import arg_scope

FLAGS = tf.app.flags.FLAGS


def model_spec(image):
    layers=[image]
    counters = {}
    with arg_scope([conv_and_bias], batch_norm=True, counters=counters):
        layers.append(conv_and_bias(0, layers[-1], 3, 16, 1, 'SAME', leaky_relu))
        layers.append(tf.nn.max_pool(layers[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'VALID'))
        layers.append(conv_and_bias(2, layers[-1], 3, 32, 1, 'SAME', leaky_relu))
        layers.append(tf.nn.max_pool(layers[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'VALID'))
        layers.append(conv_and_bias(4, layers[-1], 3, 64, 1, 'SAME', leaky_relu))
        layers.append(tf.nn.max_pool(layers[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'VALID'))
        layers.append(conv_and_bias(6, layers[-1], 3, 128, 1, 'SAME', leaky_relu))
        layers.append(tf.nn.max_pool(layers[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'VALID'))
        layers.append(conv_and_bias(8, layers[-1], 3, 256, 1, 'SAME', leaky_relu))
        layers.append(tf.nn.max_pool(layers[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'VALID'))
        layers.append(conv_and_bias(10, layers[-1], 3, 512, 1, 'SAME', leaky_relu))
        layers.append(tf.nn.max_pool(layers[-1], [1, 2, 2, 1], [1, 1, 1, 1], 'SAME'))
        layers.append(conv_and_bias(12, layers[-1], 3, 1024, 1, 'SAME', leaky_relu))
        layers.append(conv_and_bias(13, layers[-1], 3, 1024, 1, 'SAME', leaky_relu))
        layers.append(conv_and_bias(14, layers[-1], 1, 425, 1, 'SAME', None, batch_norm=False))
    logits = layers[-1]
    merge = []
    for i in xrange(FLAGS.num_bboxes):
        base = i*(5 + FLAGS.num_classes)
        merge.append(logits[:, :, :, base:base+4])
        merge.append(tf.expand_dims(tf.sigmoid(logits[:, :, :, base+4]), -1))
        merge.append(tf.nn.softmax(logits[:, :, :, base+5:base+(FLAGS.num_classes+5)]))
    logits = tf.concat(merge, 3)
    return logits

