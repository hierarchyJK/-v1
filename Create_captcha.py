# -*-coding:utf-8 -*-
"""
@project:untitle
@author:Kun_J
@file:.py
@ide:untitled3
@time:2019-03-06 18:59:02
@month:三月
"""
from captcha.image import ImageCaptcha
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.gfile as gfile
import tensorflow as tf
import PIL.Image as Image

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


def gen_random_text(charset=CAPTCHA_CHARSET, length=CAPTCHA_LEN):
    """ 生成随机字符的方法 """
    text = [random.choice(charset) for _ in range(length)]
    return ''.join(text)


def create_captcha_dataset(size=100, data_dir='F:\\验证码数据集\\data\\', height=60, width=160, image_format='.png'):
    """ 创建并保存验证码数据集的到文件中 """
    if tf.gfile.Exists(data_dir):
        tf.gfile.DeleteRecursively(data_dir)
    tf.gfile.MakeDirs(data_dir)

    captcha = ImageCaptcha(width=width, height=height)  # 创建ImageCaptcha实例
    count = 0
    for _ in range(size):
        count += 1
        # 随机生成验证码字符串
        text = gen_random_text(CAPTCHA_CHARSET, CAPTCHA_LEN)
        captcha.write(text, data_dir + text + image_format)  # 注意相同的验证码图片会overwrite，所有实际上图片会比5000和1000要少
    print('生成了验证图片：', count)
    return None


# 创建并保存训练集
print("生成验证集...")
create_captcha_dataset(TRAIN_DATA_SIZE, TRAIN_DATA_DIR)
# 创建并保存测试集
print("生成测试集...")
create_captcha_dataset(TEST_DATA_SIZE, TEST_DATA_DIR)


def gen_captcha_dataset(size=100, height=60, widtth=160, image_format='.png'):
    # 生成并返回验证码数据集
    captcha = ImageCaptcha(width=widtth, height=height)
    images, texts = [None] * size, [None] * size
    for i in range(size):
        texts[i] = gen_random_text(CAPTCHA_CHARSET, CAPTCHA_LEN)
        # 使用PIL.Image.open()识别新生成的验证码图像,得到一个字符流
        # 然后将图像装换为（W,H,C)的Numpy数组
        images[i] = np.array(Image.open(captcha.generate(texts[i])))
    return images, texts
