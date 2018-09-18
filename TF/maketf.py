import os
import subprocess
import shutil
import cv2
import tensorflow as tf

DIR = "/home/cheng/videosearch/data/video_db/"
file = "/home/cheng/videosearch/data/index/video_list.txt"
frame_file = "/home/cheng/videosearch/data/index/frame_list.txt"
frame = "/home/cheng/videosearch/data/frame/"
record="/home/cheng/videosearch/data/tfrecord"

def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def rename():


    path = os.listdir(DIR)
    num = len(path)
    video_list = open(file, 'w')
    for i in range(len(path)):
        os.rename(os.path.join(DIR, path[i]), os.path.join(DIR, str(i + 1) + ".avi"))
        video_list.write(os.path.join(DIR, str(i + 1) + ".avi" + '\n'))
    video_list.close()
    for i in range(1, num + 1):
        video_num = '%03d' % (i)
        frame_dir = frame + video_num
        shutil.rmtree(frame_dir)
        os.mkdir(frame_dir)
        cmd = "ffmpeg -i " + DIR + str(i) + ".avi -y -f image2 -r 50 -s 64*64 " + frame + video_num + "/%3d.jpg"
        print(cmd)
        status = subprocess.call(cmd, shell=True)

def writer():

    frame_list = open(frame_file, 'w')
    fpath = os.listdir(frame)
    for i in range(len(fpath)):
        spath=os.path.join(frame,fpath[i])
        framelist=os.listdir(spath)
        framelist=sorted(framelist)
        for i in range(len(framelist)):
            s=os.path.join(spath,framelist[i])
            frame_list.write(s+'\n')

def read():

    images=[]
    labels=[]
    list=open(frame_file,'r')
    path_list=list.readlines()
    for i in range(len(path_list)):
        path=path_list[i].split()[0]
        label=int(path.split('/')[-2])-1
        image=cv2.imread(path)
        images.append(image)
        labels.append(label)

    return images,labels

def writer_TFrecord():
    images, labels = read()
    print(len(images))
    print(len(labels))

    if (len(images) == len(labels)):
        print(len(labels))
        print(len(images))
        output_filename = "%s-%.5d-of-%.5d" % ("test", 0, 1)
        out_file = os.path.join(record, output_filename)
        writer = tf.python_io.TFRecordWriter(out_file)
        for j in range(len(images)):
            image_raw = images[j].tobytes()
            label = labels[j]
            data = tf.train.Example(features=tf.train.Features(feature={
                "image": _bytes_feature(image_raw),
                "label": _int64_feature(label)
            }))

            writer.write(data.SerializeToString())

        writer.close()
    print("test data done!")

if __name__ == '__main__':
    #rename()
    #writer()
    #read()
    writer_TFrecord()





