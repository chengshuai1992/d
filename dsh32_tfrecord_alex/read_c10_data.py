import tensorflow as tf
import pickle
import numpy as np



def read_cifar10_data():
    data_dir = "../Data/cifar-10-batches-py/"
    train_name = 'data_batch_'
    test_name = 'test_batch'
    train_X = None
    train_Y = None
    test_X = None
    test_Y = None

    # train data
    for i in range(1, 6):
        file_path = data_dir + train_name + str(i)
        with open(file_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
            if train_X is None:
                train_X = dict['data']
                train_Y = dict['labels']
            else:
                train_X = np.concatenate((train_X, dict['data']), axis=0)
                train_Y = np.concatenate((train_Y, dict['labels']), axis=0)
    # test_data
    file_path = data_dir + test_name
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
        test_X = dict['data']
        test_Y = dict['labels']
    train_X = train_X.reshape((50000, 3, 32, 32)).transpose(0, 2, 3, 1).astype(np.float)
    print (train_X)
    # train_Y = train_Y.reshape((50000)).astype(np.float)
    test_X = test_X.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1).astype(np.float)
    # test_Y.reshape((10000)).astype(np.float)

    train_y_vec = np.zeros((len(train_Y), 10), dtype=np.float)
    test_y_vec = np.zeros((len(test_Y), 10), dtype=np.float)
    for i, label in enumerate(train_Y):
        train_y_vec[i, int(train_Y[i])] = 1.  # y_vec[1,3] means #2 row, #4column
    for i, label in enumerate(test_Y):
        test_y_vec[i, int(test_Y[i])] = 1.  # y_vec[1,3] means #2 row, #4column
    print(len(train_X))
    return train_X / 255., train_y_vec, test_X / 255., test_y_vec

if __name__ == '__main__':
    read_cifar10_data()