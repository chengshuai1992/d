import tensorflow as tf
import read_record32
import train_model32
import finetune_model32
import os
import read_c10_data
import numpy as np
import dsh_model
BATCH_SIZE = 200
HASHING_BITS = 12
TRAINING_STEPS = 50000 // 200
EPOCH = 600
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.4
DECAY_STEPS =500
IMAGE_SIZE = 227
model_file = "../Data/weight/finetune_weights"
checkpoint_file = "../Data/checkpiont/"
model = {}


def hash_loss(image, label, alpha, m):
    _, D, net_model = dsh_model.alexnet_layer(image)
    w_label = tf.matmul(label, label, False, True)

    r = tf.reshape(tf.reduce_sum(D * D, 1), [-1, 1])
    p2_distance = r - 2 * tf.matmul(D, D, False, True) + tf.transpose(r)
    temp = w_label * p2_distance + (1 - w_label) * tf.maximum(m - p2_distance, 0)

    regularizer = tf.reduce_sum(tf.abs(tf.abs(D) - 1))
    d_loss = tf.reduce_sum(temp) / (BATCH_SIZE * (BATCH_SIZE - 1)) + alpha * regularizer / BATCH_SIZE
    return d_loss, net_model


def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    image = tf.placeholder(tf.float32, shape=[200, 32, 32, 3], name='image')
    label = tf.placeholder(tf.float32, shape=[200, 10], name='label')

    alpha = tf.constant(0.01, dtype=tf.float32, name='tradeoff')
    m = tf.constant(HASHING_BITS * 2, dtype=tf.float32, name='bi_margin')
    global_step = tf.Variable(0, name='global_step', trainable=False)

    saver = tf.train.Saver()

    hloss, model_paramater = hash_loss(image, label, alpha, m)
    '''
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               DECAY_STEPS,
                                               LEARNING_RATE_DECAY,
                                               staircase=True)

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(hloss,
                                                                                               global_step=global_step)

    '''
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE_BASE, beta1=0.5).minimize(hloss, global_step=global_step)

    with tf.Session() as sess:
        '''
        init=tf.global_variables_initializer()
        sess.run(init)
        start=0
        train_x, train_y, test_x, test_y = read_c10_data.read_cifar10_data()

        for epoch in range(start, EPOCH):

            batch_idxs = 50000 // BATCH_SIZE

            for idx in range(start, batch_idxs):
                image_idx = train_x[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
                label_idx = train_y[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]

                _,loss=sess.run([optimizer,hloss], feed_dict={image: image_idx, label: label_idx})
                # writer.add_summary(summary_str, idx + 1)



                if (idx + 1) % 10 == 0:
                    print("[%3d/%3d][%4d/%4d] d_loss: %.8f" % (epoch + 1, EPOCH, idx + 1, batch_idxs, loss))

        '''
        images, labels = read_record32.reader_TFrecord(EPOCH)
        image_batch, label_batch = read_record32.next_batch(images, labels)
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for epoch in range(EPOCH):

            for i in range(TRAINING_STEPS):

                image_b, label_b = sess.run([image_batch, label_batch])
                _, loss, step = sess.run([optimizer, hloss, global_step],
                                         feed_dict={image: image_b, label: label_b})
                if (i + 1) % 50 == 0:
                    print("After %d/%d training Epoch and total %d step , current batch loss is =%.8f" % (
                        epoch + 1, EPOCH, step, loss))

            for layer in model_paramater:
                model[layer] = sess.run(model_paramater[layer])
            print("saving model to %s" % model_file)
            np.save(model_file, np.array(model))
            print("save model successful!")

            if (epoch + 1) % 100 == 0:
                print("saving checkpoint !")
                checkpoint_path = os.path.join(checkpoint_file, 'DSH_model.ckpt')
                saver.save(sess, checkpoint_path, global_step=epoch + 1)
                print('*********   checkpoint model saved    *********')

        coord.request_stop()
        coord.join(threads)

    sess.close()


def test():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'


if __name__ == '__main__':
    train()
