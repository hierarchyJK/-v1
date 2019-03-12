# -*-coding:utf-8 -*-
"""
@project:untitled3
@author:Kun_J
@file:.py
@ide:untitled3
@time:2019-03-08 17:14:38
@month:三月
"""
from captcha.image import ImageCaptcha
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.gfile as gfile
import tensorflow as tf
import PIL.Image as Image
from keras.utils.vis_utils import plot_model  # 模型可视化
from keras import backend as K
import glob
import pickle
from keras.layers import *
from keras.models import *


# 生成验证码数据集
NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LOWERCASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
             'v', 'w', 'x', 'y', 'z']
UPPERCASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
             'V', 'W', 'X', 'Y', 'Z']
CAPTCHA_CHARSET = NUMBER  # 验证码数据集
CAPTCHA_LEN = 4  # 验证码长度
CAPTCHA_HEIGHT = 60  # 验证码高度
CAPYCHA_WIDTH = 160  # 验证码宽度

TRAIN_DATA_SIZE = 5000  # 验证码数据集的大小
TEST_DATA_SIZE = 1000

TRAIN_DATA_DIR = 'F:\\验证码数据集\\VCR_train_data\\'  # 验证码数据集的目录
TEST_DATA_DIR = 'F:\\验证码数据集\\VCR_test_data\\'

BATCH_SIZE = 100
EPOCHS = 10
OPT = "adam"
LOSS = "binary_crossentropy"

MODEL_DIR = "./Captcha_model/train_demo/"
MODEL_FORMAT = ".h5"
HISTORY_DIR = "./Captcha_history/train_demo/"
HISTORY_FORMAT = ".history"

filename_str = "{} captcha_{}_{}_bs_{}_epochs_{}{}"
# 模型网络结构文件
MODEL_VIS_FILE = "captcha_classfication" + ".png"
# 模型文件
MODEL_FILE = filename_str.format(MODEL_DIR, OPT, LOSS, str(BATCH_SIZE), str(EPOCHS), MODEL_FORMAT)
# 训练记录文件
HISTORY_FILE = filename_str.format(HISTORY_DIR, OPT, LOSS, str(BATCH_SIZE), str(EPOCHS), HISTORY_FORMAT)


def rgb2gray(image):
    """将RGB验证码图像转为灰度图"""
    # 这里一种方法 0.299R+0.587G+0.14B
    return np.dot(image[..., :3], [0.299, 0.587, 0.114])


def text2vec(text, length=CAPTCHA_LEN, charset=CAPTCHA_CHARSET):
    """对验证码中的每个字符进行one-hot编码"""
    text_len = len(text)
    if text_len != length:
        raise ValueError('Error:length od captcha should be {}, but got {}'.format(length, text_len))

    vec = np.zeros(length * len(charset))
    for i in range(length):
        # one-hot编码验证码中的每个数字，每个字符的编码 = 索引 + 偏移量
        vec[charset.index(text[i]) + i * len(charset)] = 1
    return vec


def vet2text(vec):
    """将验证码向量解码成对应的字符"""
    if not isinstance(vec, np.ndarray):
        vec = np.asarray(vec)
    vec = np.reshape(vec, [CAPTCHA_LEN, -1])
    text = ''

    for item in vec:
        text += CAPTCHA_CHARSET[np.argmax(item)]
    return text


def fit_keras_channels(batch, rows=CAPTCHA_HEIGHT, cols=CAPYCHA_WIDTH):
    """适配Keras图像数据格式"""
    if K.image_data_format() == 'channels_first':
        batch = batch.reshape(batch.shape[0], 1, rows, cols)
        input_shape = (1, rows, cols)
    else:  # keras 默认的是channels_last
        batch = batch.reshape(batch.shape[0], rows, cols, 1)
        input_shape = (rows, cols, 1)
    return batch, input_shape


"""读取训练集"""
X_train = []
Y_train = []
for filename in glob.glob(TRAIN_DATA_DIR + '*.png'):
    X_train.append(np.array(Image.open(filename)))
    Y_train.append(filename.lstrip(TRAIN_DATA_DIR).rstrip('.png'))

"""处理训练集图像"""
X_train = np.array(X_train, dtype=np.float32)  # 变成浮点型，保证normalize正常进行
X_train = rgb2gray(X_train)  # 转化为灰度图
X_train = X_train / 255  # normalize
X_train, input_shape = fit_keras_channels(X_train)
print(X_train.shape, type(X_train))  # (3915, 60, 160, 1)
print(input_shape)  # (60, 160, 1)

"""处理训练集标签"""
Y_train = list(Y_train)
for i in range(len(Y_train)):
    Y_train[i] = text2vec(Y_train[i])
Y_train = np.asarray(Y_train)
print(Y_train.shape, type(Y_train))  # (3915, 40)

"""读取测试集、处理测试集图片和标签"""
X_test = []
Y_test = []
for filename in glob.glob(TEST_DATA_DIR + '*.png'):
    X_test.append(np.array(Image.open(filename)))
    Y_test.append(filename.lstrip(TEST_DATA_DIR).rstrip('.png'))

X_test = np.array(X_test, dtype=np.float32)
X_test = rgb2gray(X_test)
X_test = X_test / 255
X_test, _ = fit_keras_channels(X_test)

Y_test = list(Y_test)
for i in range(len(Y_test)):
    Y_test[i] = text2vec(Y_test[i])

Y_test = np.asarray(Y_test)
print(X_test.shape, type(X_test))  # (944, 60, 160, 1)
print(Y_test.shape, type(Y_test))  # (944, 40)

"""创建验证码识别模型"""
# 输入层
inputs = Input(shape=input_shape, name='inputs')
# 第1层卷积
conv1 = Conv2D(filters=32, kernel_size=(3, 3), name="conv1")(inputs)
relu1 = Activation(activation='relu', name='relu1')(conv1)
# 第2层卷积
conv2 = Conv2D(filters=32, kernel_size=(3, 3), name="conv2")(relu1)
relu2 = Activation(activation='relu', name='relu2')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2), padding='same', name='pool2')(relu2)
# 第3层卷积
conv3 = Conv2D(filters=64, kernel_size=(3, 3), name='conv3')(pool2)
relu3 = Activation(activation='relu', name='relu3')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2), padding='same', name='pool3')(relu3)
# 将Pooled feature map 摊平后输入全连接神经网络（参考AlexNet网络）
x = Flatten()(pool3)
x = Dropout(0.5)(x)
x = [Dense(10, activation='softmax', name='fc%d' % (i + 1))(x) for i in range(4)]
# 4个字符向量拼接在一起，与标签向量形式一致，作为模型输出
outs = Concatenate()(x)
# 定义模型的输入和输出
model = Model(inputs=inputs, outputs=outs)
model.compile(optimizer=OPT, loss=LOSS, metrics=['accuracy'])

"""查看模型摘要"""
model.summary()

"""模型可视化"""
plot_model(model, to_file=MODEL_VIS_FILE, show_shapes=True, show_layer_names=True)

"""训练模型"""
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2, validation_data=(X_test, Y_test))

"""保存模型"""
if not tf.gfile.Exists(MODEL_DIR):
    tf.gfile.MakeDirs(MODEL_DIR)
model.save(MODEL_FILE)
print("Saved trained model at：%s" % MODEL_FILE)

"""保存训练过程"""
if tf.gfile.Exists(HISTORY_DIR) == False:
    tf.gfile.MakeDirs(HISTORY_DIR)
with open(HISTORY_FILE, 'wb') as f:
    pickle.dump(history.history, f)
print(HISTORY_FILE)
