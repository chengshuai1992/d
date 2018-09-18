import tensorflow as tf


import tensorflow as tf
import numpy as np


def alexnet_layer(image):
    train_paramater = {}
    ### Conv1
    ### Output 64, kernel 5, stride 1 out 28*28
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 3, 32], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='biases')
        conv = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], padding='SAME')

        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name=scope)
        train_paramater['conv1'] = [kernel, biases]

    ### Pool1 out 27*27
    pool1 = tf.nn.max_pool(conv1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')

    ### LRN1
    radius = 2;
    alpha = 2e-05;
    beta = 0.75;
    bias = 1.0
    lrn1 = tf.nn.local_response_normalization(pool1,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    ### Conv2
    ### Output 256, pad 2, kernel 5 27*27
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 32, 32], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='biases')
        conv = tf.nn.conv2d(lrn1, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(out, name=scope)
        train_paramater['conv2'] = [kernel, biases]

    ### Pool2  out 13*13
    pool2 = tf.nn.avg_pool(conv2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool2')

    ### LRN2
    radius = 2;
    alpha = 2e-05;
    beta = 0.75;
    bias = 1.0
    lrn2 = tf.nn.local_response_normalization(pool2,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    ### Conv3
    ### Output 384, pad 1, kernel 3 out  13*13
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 32, 64], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
        conv = tf.nn.conv2d(lrn2, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(out, name=scope)
        train_paramater['conv3'] = [kernel, biases]

    ### Pool5  out 3*3*256
    pool5 = tf.nn.avg_pool(conv3,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool5')

    ### FC6  6*6*256
    ### Output 4096
    with tf.name_scope('fc6') as scope:
        shape = int(np.prod(pool5.get_shape()[1:]))

        fc6w = tf.Variable(tf.random_normal([shape, 500], dtype=tf.float32, stddev=1e-2), name='weights')
        fc6b = tf.Variable(tf.constant(0.0, shape=[500], dtype=tf.float32), trainable=True, name='biases')
        pool5_flat = tf.reshape(pool5, [-1, shape])
        fc5 = pool5_flat
        fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
        fc6_drop = tf.nn.dropout(tf.nn.relu(fc6l), 0.5)
        fc6 = tf.nn.relu(fc6l)
        train_paramater['fc6'] = [fc6w, fc6b]

    ### FC8
    ### Output output_dim
    with tf.name_scope('fc8') as scope:
        ### Differ train and val stage by 'fc8' as key

        fc8w = tf.Variable(tf.random_normal([500, 12], dtype=tf.float32, stddev=1e-2), name='weights')
        fc8b = tf.Variable(tf.constant(0.0, shape=[12], dtype=tf.float32), trainable=True, name='biases')
        fc8l = tf.nn.bias_add(tf.matmul(fc6_drop, fc8w), fc8b)
        fc8 = tf.nn.bias_add(tf.matmul(fc6, fc8w), fc8b)

    return fc8l, fc8, train_paramater
