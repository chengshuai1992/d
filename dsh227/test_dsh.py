import os
import numpy as np
import tensorflow as tf
import finetune_model
import read_record

BATCH_SIZE = 200
HASHING_BITS = 12
TRAINING_STEPS = 50000 // 200
EPOCH = 1
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.4
DECAY_STEPS = 500
IMAGE_SIZE = 227
model_file = "../Data/weight/finetune_weights"
checkpoint_file = "../Data/checkpiont/"
shuffle=False

def toBinaryString(binary_like_values):
    numOfImage, bit_length = binary_like_values.shape
    list_string_binary = []
    for i in range(numOfImage):
        str = ''
        for j in range(bit_length):
            str += '0' if binary_like_values[i][j] <= 0 else '1'
        list_string_binary.append(str)
    return list_string_binary


def test():
    image = tf.placeholder(tf.float32, [None, 32, 32, 3], name='image')
    D = finetune_model.alexnet_layer(image)
    res = tf.sign(D)

    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_file)
    saver = tf.train.Saver(tf.all_variables())

    with tf.Session() as sess:
        # train_x, train_y, test_x, test_y = read_cifar10_data()
        images, labels = read_record.reader_TFrecord(EPOCH,shuffle)
        image_batch, label_batch = read_record.next_batch(images, labels)
        file_res = open('result.txt', 'w')

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, os.path.join(checkpoint_file, ckpt_name))
            print('Loading success, global_step is %s' % global_step)
            for i in range(TRAINING_STEPS):

                result = sess.run(D, feed_dict={image: image_batch})
                #print(result)
                b_result = toBinaryString(result)
                index_label = np.argmax(label_batch, axis=1)
                for j in range(BATCH_SIZE):
                    file_res.write(b_result[j] + '\t' + str(index_label[j]) + '\n')

    file_res.close()
    sess.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
