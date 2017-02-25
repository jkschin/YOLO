import tensorflow as tf
import numpy as np

def load_4d(sess, shape, conv_weights_vars, pretrained_weights):
    conv_weights = np.empty(shape, dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for l in range(shape[3]):
                    conv_weights[i, j, k, l] = pretrained_weights[(l * shape[0] * shape[1] * shape[2]) + (k * shape[0] * shape[1]) + (i * shape[0]) + j]
    return sess.run(tf.assign(conv_weights_vars, conv_weights))

def load_1d(sess, shape, weights_vars, pretrained_weights):
    weights = np.empty(shape, dtype=np.float32)
    for i in range(shape[0]):
        weights[i] = pretrained_weights[i]
    return sess.run(tf.assign(weights_vars, weights))

def load_from_binary(sess):
    f = open('/home/jkschin/code/github-others/darknet/tiny-yolo.weights', 'rb')
    # NOTE This read is important. It reads out the junk.
    for i in xrange(4):
        np.frombuffer(f.read(4), np.int32)
    idx = 0
    pretrained_weights = np.fromfile(f, np.float32)
    var_list = ['bias', 'scale', 'mean', 'variance', 'conv']
    with tf.variable_scope('model', reuse=True):
        for i in xrange(8):
            for var_name in var_list:
                if var_name == 'conv':
                    var_name = var_name + '_' + str(i)
                    var = tf.get_variable(var_name)
                    shape = map(lambda x: int(x), var.get_shape())
                    end_idx = reduce(lambda x, y: x * y, shape)
                    var = load_4d(sess, shape, var, pretrained_weights[idx:idx+end_idx])
                    print "Loading %s from %d to %d" %(var_name, idx, idx+end_idx)
                    idx += end_idx
                else:
                    var_name = var_name + '_' + str(i)
                    var = tf.get_variable(var_name)
                    shape = map(lambda x: int(x), var.get_shape())
                    end_idx = reduce(lambda x, y: x * y, shape)
                    var = load_1d(sess, shape, var, pretrained_weights[idx:idx+end_idx])
                    print "Loading %s from %d to %d" %(var_name, idx, idx+end_idx)
                    idx += end_idx

        for i in xrange(8, 9):
            for var_name in var_list:
                if var_name == 'bias':
                    var_name = var_name + '_' + str(i)
                    var = tf.get_variable(var_name)
                    shape = map(lambda x: int(x), var.get_shape())
                    end_idx = reduce(lambda x, y: x * y, shape)
                    var = load_1d(sess, shape, var, pretrained_weights[idx:idx+end_idx])
                    print "Loading %s from %d to %d" %(var_name, idx, idx+end_idx)
                    idx += end_idx
                elif var_name == 'conv':
                    var_name = var_name + '_' + str(i)
                    var = tf.get_variable(var_name)
                    shape = map(lambda x: int(x), var.get_shape())
                    end_idx = reduce(lambda x, y: x * y, shape)
                    var = load_4d(sess, shape, var, pretrained_weights[idx:idx+end_idx])
                    print "Loading %s from %d to %d" %(var_name, idx, idx+end_idx)
                    idx += end_idx
    assert idx == 16175385

def darknet_resize(img, w, h):
    shape = img.shape
    resized = np.zeros((h, w, 3))
    part = np.zeros((shape[0], w, 3))
    w_scale = (shape[1] - 1) / float(w - 1)
    h_scale = (shape[0] - 1) / float(h - 1)
    for k in xrange(3):
        for r in xrange(shape[0]):
            for c in xrange(w):
                val = 0.0
                if (c == w-1) or (shape[1] == 1):
                    val = img[r, shape[1]-1, k]
                else:
                    sx = c*w_scale
                    ix = int(sx) #NOTE might have to check this casting
                    dx = sx - ix
                    val = (1 - dx) * img[r, ix, k] + dx * img[r, ix+1, k]
                part[r, c, k] = val
    for k in xrange(3):
        for r in xrange(h):
            sy = r*h_scale
            iy = int(sy) #NOTE might have to check this casting
            dy = sy - iy
            for c in xrange(w):
                val = (1 - dy) * part[iy, c, k]
                resized[r, c, k] = val
            if (r == h-1) or (shape[0] == 1):
                continue
            for c in xrange(w):
                val = dy * part[iy+1, c, k]
                resized[r, c, k] += val
    return resized

def print_layer_weights(sess, feed_dict):
    for node in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        print node.name
        print sess.run(node, feed_dict)
    # var_list = ['bias', 'scale', 'mean', 'variance', 'conv']
    # with tf.variable_scope('model', reuse=True):
    #     for l in xrange(8):
    #         for var_name in var_list:
    #         scale = tf.get_variable('scale_%d' %l)
    #         mean = tf.get_variable('mean_%d' %l)
    #         variance = tf.get_variable('variance_%d' %l)
    #         print scale, mean, variance
    #         print scale.name, mean.name, variance.name
    #         s_val, m_val, var_val = sess.run([scale, mean, variance], feed_dict=feed_dict)
    #         print s_val
    #         print m_val
    #         print var_val

def print_layer_outputs(sess, feed_dict):
    print 'INPUT DEBUG'
    print sess.run(tf.get_collection('inp_debug'), feed_dict=feed_dict)
    print 'CONV DEBUG'
    print sess.run(tf.get_collection('conv_debug'), feed_dict=feed_dict)
    print 'BN DEBUG'
    print sess.run(tf.get_collection('bn_debug'), feed_dict=feed_dict)
    print 'BIAS DEBUG'
    print sess.run(tf.get_collection('bias_debug'), feed_dict=feed_dict)
    # for l in xrange(16):
    #     node = tf.get_collection('layers_%d' %l)
    #     node_val = np.array(sess.run(node, feed_dict=feed_dict)[0])
    #     print node_val.shape
    #     for idx, val in enumerate(node_val.flatten()):
    #         print idx, val

def print_xywh_raw(sess, feed_dict):
    f = open('xywh_raw.txt', 'w')
    output = sess.run(tf.get_collection('xywh_raw'), feed_dict=feed_dict)
    output = np.array(output)
    shape = output.shape
    print shape
    output = np.reshape(output, (reduce(lambda x, y: x * y, shape[:-1]), shape[-1]))
    for i in output:
        f.write(str(i) + "\n")
    f.close()

def print_xywh(sess, feed_dict):
    f = open('xywh.txt', 'w')
    output = sess.run(tf.get_collection('xywh'), feed_dict=feed_dict)
    output = np.array(output)
    shape = output.shape
    print shape
    output = np.reshape(output, (reduce(lambda x, y: x * y, shape[:-1]), shape[-1]))
    for i in output:
        f.write(str(i) + "\n")
    f.close()


