import tensorflow as tf
import os
import pickle
import numpy as np
'''

'''
train_filename_list=['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
test_batch_filename='test_batch'
flag_Dir={'tfrecod_path_train':'./train.tfrecord','tfrecord_path_test':'./test.tfrecord'}
# read the datasets 10000*32*32*3->10000*3072 label:0-9 10000 of each batch
def  unpicke(filename):
    with open(filename,'rb') as f:
        dict=pickle.load(f,encoding="bytes")
        #Image.show(dict[b'data'][1]
        #print(dict)
    return dict


# read the data from a list of filenamepath
def unpicke_queue(filenameList,path):
    if os.path.exists(path):
        print('file exist .......')
        return
    datasets = []
    writer = tf.python_io.TFRecordWriter(path)
    for index in filenameList:
        path=os.path.join('./cifar/',index)
        with open(path,'rb') as fo:
            i_data=pickle.load(fo,encoding="bytes")
            data = i_data[b'data']
            labels = i_data[b'labels']
            #print(labels)
            for inv in range(data.shape[0]):
                #print(data.shape)
                #print(inv)
                data_inv = np.reshape(data[inv],[32,32,3])
                #print(data_inv)
                data_sgd =data_inv.tobytes()
                example = tf.train.Example(features=tf.train.Features(
                    feature={'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(labels[inv])])),
                             'img_row': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_sgd]))}))
                writer.write(example.SerializeToString())
    writer.close()

# make the data into tfrecord
def datasets_to_tfrecord(file_in_path,save_dir):
    if os.path.exists(save_dir):
        print("exist:%s"%(save_dir))
        return
    print("convert to tfrecord....")
    writer=tf.python_io.TFRecordWriter(save_dir)
    dict=unpicke(file_in_path)
    labels=dict[b'labels']
    data=dict[b'data']
    #data=data.astype(bytes)
    for index in range(data.shape[0]):
        print(data.shape)
        print(index)
        data[index]=tf.reshape(data[index],[32,32,3])
        data_sd=data[index].tobytes()
        # 每一行32*32*3
        #da=tf.reshape(data[index],[32,32,3])
        example=tf.train.Example(features=tf.train.Features(feature={'labels':tf.train.Feature(int64_list=tf.train.Int64List(value=[int(labels[index])])),
                                                                     'img_row':tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_sd]))}))
        writer.write(example.SerializeToString())
    writer.close()
def read_tfrecord(filepath):
    if not os.path.exists(filepath):
        print('flie not exist.....')
        return
    print('read data......')
    file_queue=tf.train.string_input_producer([filepath])
    reader=tf.TFRecordReader()
    _,serialize_axample=reader.read(file_queue)
    features=tf.parse_single_example(serialized=serialize_axample,features={'labels':tf.FixedLenFeature([],tf.int64),
                                                                            'img_row':tf.FixedLenFeature([],tf.string)})
    img=tf.decode_raw(features['img_row'],tf.uint8)
    img_re=tf.reshape(img,[32,32,3])
    img_r=tf.cast(img_re,tf.float32)*(1/255)+0.5
    labels=tf.cast(features['labels'],tf.int64)
    #print(img_r)
    #print(labels)
    return img_r,labels



# AlexNet 网络 5层卷积3三层全链接
# define the network struct 简化版AlxNet模型
def inference_alxnet(img,label):
    # img：shape-[batch_size,img_size,img_size,chanel]
    paramenter=[]
    #intdata_train=tf.placeholder([100,32,32,3],name="train_input")
    #y_=tf.placeholder([100,10],name="y_label")
    # conv1 11*11 stride=4
    # return paramenter and fc3_out
    with tf.name_scope('conv1') as scope:
        kerenal=tf.Variable(tf.truncated_normal([2,2,3,64],stddev=0.1,dtype=tf.float32),name='weight1')
        conv=tf.nn.conv2d(img,kerenal,[1,1,1,1],padding='SAME')
        basic=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[64]),trainable=True,name='basic1')
        bais=tf.nn.bias_add(conv,basic)
        conv1=tf.nn.relu(bais,name=scope)
        paramenter+=[kerenal,basic]
    # LRN1
    with tf.name_scope('LRN1') as scope:
        lrn1=tf.nn.lrn(conv1,depth_radius=2,bias=2.0,alpha=1e-4,beta=0.75,name=scope)
    # maxpooling1
    with tf.name_scope('maxpoooling1') as scope:
        pool1=tf.nn.max_pool(lrn1,[1,2,2,1],[1,1,1,1],padding='VALID',name=scope)
    # conv2
    with tf.name_scope('conv2') as scope:
        kerenal=tf.Variable(tf.truncated_normal([2,2,64,192],stddev=0.1,dtype=tf.float32),name='weight2')
        conv=tf.nn.conv2d(pool1,kerenal,[1,1,1,1],padding='SAME')
        basic=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[192]),name='basic2')
        bais=tf.nn.bias_add(conv,basic)
        conv2=tf.nn.relu(bais,name=scope)
        paramenter+=[kerenal,basic]
    # LRN2
    with tf.name_scope('LRN2') as scope:
        lrn2=tf.nn.lrn(conv2,depth_radius=2,bias=2,alpha=1e-4,beta=0.75,name=scope)
    # maxpooling2
    with tf.name_scope('maxpooling2') as scope:
        pool2=tf.nn.max_pool(lrn2,[1,2,2,1],strides=[1,1,1,1],padding='VALID',name=scope)
    # conv3
    with tf.name_scope('conv3') as scope:
        kerenal = tf.Variable(tf.truncated_normal([3, 3, 192,320], stddev=0.1, dtype=tf.float32), name='weight3')
        conv = tf.nn.conv2d(pool2, kerenal, [1, 2, 2, 1], padding='SAME')
        basic = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[320]), name='basic3')
        bais = tf.nn.bias_add(conv, basic)
        conv3 = tf.nn.relu(bais, name=scope)
        paramenter += [kerenal, basic]
    # conv4
    with tf.name_scope('conv4') as scope:
        kerenal=tf.Variable(tf.truncated_normal([3,3,320,256],stddev=0.1,dtype=tf.float32),name='weight4')
        conv=tf.nn.conv2d(conv3,kerenal,[1,2,2,1],padding='SAME')
        basis=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[256]),name='basic4')
        conv4=tf.nn.bias_add(conv,basis,name=scope)
        paramenter+=[kerenal,basis]
    # conv5
    with tf.name_scope('conv5') as scope:
        kerenal=tf.Variable(tf.truncated_normal([3,3,256,256],stddev=0.1,dtype=tf.float32),name='weight5')
        conv=tf.nn.conv2d(conv4,kerenal,strides=[1,2,2,1],padding='SAME')
        basis=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[256]),name='basic5')
        conv5=tf.nn.bias_add(conv,basis,name=scope)
        paramenter+=[kerenal,basis]
    # pooling5
    with tf.name_scope('maxpooling5') as scope:
        # alxnet 网络结构中建议池化核和卷积核尺寸大小应大于步长，重叠卷积和池化使特征更丰富
        pool5=tf.nn.max_pool(conv5,ksize=[1,1,1,1],strides=[1,1,1,1],padding='VALID',name=scope)
        # make pool5 innto a variable of 2d 将pool5 输出的特征图拉伸成二维张量
        shape=pool5.get_shape()
        pool5_float=tf.reshape(pool5,[-1,shape[1].value*shape[2].value*shape[3].value])

    # 全链接层
    with tf.name_scope('fc1') as scope:
        fc1=tf.layers.dense(pool5_float,1024,activation=tf.nn.relu,name=scope)
    #dropout
    drop_out_fc1=tf.nn.dropout(fc1,keep_prob=0.5)
    with tf.name_scope('fc2') as scope:
        fc2=tf.layers.dense(drop_out_fc1,1024,activation=tf.nn.relu,name=scope)
    # drop_out
    drop_out_fc2=tf.nn.dropout(fc2,keep_prob=0.5)
    # out_predictions
    with tf.name_scope('out_prediction') as prediction_current:
        out_prediction=tf.layers.dense(drop_out_fc2,10,activation=tf.nn.softmax,name=prediction_current)
        #out_prediction=out_prediction+1e-4
    return  paramenter,out_prediction

