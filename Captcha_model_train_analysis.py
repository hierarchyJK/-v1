# -*-coding:utf-8 -*-
"""
@project:untitled3
@author:Kun_J
@file:.py
@ide:untitled3
@time:2019-03-13 10:08:55
@month:三月
"""
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

history_file = './pre-trained/history/optimizer/binary_ce/captcha_adam_binary_crossentropy_bs_100_epochs_100.history'
with open(history_file, 'rb') as f:
    history = pickle.load(f)
print(history.keys())  # dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])

"""训练过程可视化"""
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('Model Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# 加载预训练模型的记录
#HISTORY_DIR = '.\\pre-trained\\history\\optimizer\\binary_ce\\'#不同优化器对于的historry
HISTORY_DIR = '.\\pre-trained\\history\\loss\\'
history = {}
for filename in glob.glob(HISTORY_DIR + '*.history'):
    with open(filename, 'rb') as f:
        history[filename] = pickle.load(f)
for key, val in history.items():
    print(key.replace(HISTORY_DIR, '').rstrip('.history'), val.keys())


def plot_training(history=None, metric='acc', title="Model Accuracy", loc='lower right'):
    model_list = []
    fig = plt.figure(figsize=(10, 8))
    for key, value in history.items():
        model_list.append(key.replace(HISTORY_DIR, '').rstrip('.history'))
        plt.plot(value[metric])
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel(metric)
    plt.legend(model_list, loc=loc)
    plt.show()


plot_training(history, metric='acc', title='Model Accuracy(train)', loc='lower right')
plot_training(history, metric='loss', title='Model Loss(train)', loc='upper right')
plot_training(history, metric='val_acc', title='Model Accuracy(val)', loc='lower right')
plot_training(history, metric='val_loss', title='Model Loss(val)', loc='upper right')
