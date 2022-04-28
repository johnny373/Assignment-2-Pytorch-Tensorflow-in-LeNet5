# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 22:10:02 2022

@author: Johnny
"""
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D

from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_pic_data_by_txt(pic_load_list): 
    tmp_pic_w, tmp_pic_h=  cv2.imread(pic_load_list[0], cv2.IMREAD_GRAYSCALE).shape
    tmp_size = 256
    batch_size = len(pic_load_list)
    tmp_x_array = np.zeros(batch_size * 32 * 32).reshape((batch_size, 32, 32))
        
    for index, tmp_pic_path in enumerate(pic_load_list):
        temp_pic = cv2.imread(pic_load_list[index], cv2.IMREAD_GRAYSCALE)
        temp_pic = crop_pic(temp_pic, tmp_size)
        temp_pic = cv2.resize(temp_pic, (32, 32), interpolation=cv2.INTER_CUBIC)
        tmp_x_array[index] = temp_pic
    
    tmp_x_array.astype('float32')
    tmp_x_array /= 255 
    return tmp_x_array

def crop_pic(image, size):
    h, w = image.shape
    x, y = w // 2, h // 2
    block_size = size // 2
    crop_image = image[y - block_size : y + block_size, x - block_size : x + block_size]
    
    return crop_image

def MakeOneHot(Y, D_out): 
    N = Y.shape[0]
    Z = np.zeros((N, D_out))
    Z[np.arange(N), Y] = 1
    return Z

def draw_losses(losses):
    t = np.arange(len(losses))
    plt.plot(t, losses)
    plt.show()

def get_batch_data(train_txt, batch_size, number_of_category):
    number_of_data = len(train_txt)
    batch_index = np.random.randint(number_of_data, size = batch_size)
    batch_x = load_pic_data_by_txt(list(train_txt.loc[batch_index, 'pic_path']))
    batch_y = np.array(train_txt.loc[batch_index, 'label'])
    return batch_x, batch_y

def draw_losses(losses):
    t = np.arange(len(losses))
    plt.plot(t, losses)
    plt.show()

class LeNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super(LeNet, self).__init__(name='LeNet', dynamic=False)
        self.conv_layer_1 = tf.keras.layers.Conv2D(
                filters=6,
                kernel_size=(5, 5),
                input_shape=(28, 28, 1),
                padding='valid',
                activation=tf.nn.relu
                )
        self.pool_layer_1 = tf.keras.layers.MaxPooling2D(padding='same')
        self.conv_layer_2 = tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=(5, 5),
                padding='valid',
                activation=tf.nn.relu
                )
        self.pool_layer_2 = tf.keras.layers.MaxPooling2D(padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.fc_layer_1 = tf.keras.layers.Dense(
                units=120,
                activation=tf.nn.relu
                )
        self.fc_layer_2 = tf.keras.layers.Dense(
                units=84,
                activation=tf.nn.relu
                )
        self.output_layer = tf.keras.layers.Dense(
                units=kwargs['num_classes'],
                activation=tf.nn.softmax
                )

    @tf.function
    def call(self, features):
        activation = self.conv_layer_1(features)
        activation = self.pool_layer_1(activation)
        activation = self.conv_layer_2(activation)
        activation = self.pool_layer_2(activation)
        activation = self.flatten(activation)
        activation = self.fc_layer_1(activation)
        activation = self.fc_layer_2(activation)
        output = self.output_layer(activation)
        return output

class LeNet_D(tf.keras.Model):
    def __init__(self, **kwargs):
        super(LeNet_D, self).__init__(name='LeNet_D', dynamic=True)
        self.conv_layer_1 = tf.keras.layers.Conv2D(
                filters=6,
                kernel_size=(5, 5),
                input_shape=(28, 28, 1),
                padding='valid',
                activation=tf.nn.relu
                )
        self.pool_layer_1 = tf.keras.layers.MaxPooling2D(padding='same')
        self.conv_layer_2 = tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=(5, 5),
                padding='valid',
                activation=tf.nn.relu
                )
        self.pool_layer_2 = tf.keras.layers.MaxPooling2D(padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.fc_layer_1 = tf.keras.layers.Dense(
                units=120,
                activation=tf.nn.relu
                )
        self.fc_layer_2 = tf.keras.layers.Dense(
                units=84,
                activation=tf.nn.relu
                )
        self.output_layer = tf.keras.layers.Dense(
                units=kwargs['num_classes'],
                activation=tf.nn.softmax
                )

    @tf.function
    def call(self, features):
        activation = self.conv_layer_1(features)
        activation = self.pool_layer_1(activation)
        activation = self.conv_layer_2(activation)
        activation = self.pool_layer_2(activation)
        activation = self.flatten(activation)
        activation = self.fc_layer_1(activation)
        activation = self.fc_layer_2(activation)
        output = self.output_layer(activation)
        return output
    
# Load Data
(train_features, train_labels), (test_features, test_labels) = tf.keras.datasets.mnist.load_data()
train_features = train_features.reshape(-1, 28, 28, 1)

batch_size = 32
number_of_category = 50

train_txt = pd.read_csv('train.txt', header = None, names = ['pic_path', 'label'], sep = ' ')
val_txt = pd.read_csv('val.txt', header = None, names = ['pic_path', 'label'], sep = ' ')
test_txt = pd.read_csv('test.txt', header = None, names = ['pic_path', 'label'], sep = ' ')

train_x, train_y = get_batch_data(train_txt, len(train_txt), number_of_category)
val_x, val_y = get_batch_data(val_txt, len(val_txt), number_of_category)
test_x, test_y = get_batch_data(test_txt, len(test_txt), number_of_category)

train_x = train_x[:, :, :, np.newaxis]
val_x = val_x[:, :, :, np.newaxis]
test_x = test_x[:, :, :, np.newaxis]

print('x_train shape:', train_x.shape)
print(train_x.shape[0], 'train samples')
print(test_x.shape[0], 'test samples')
print(train_x[0].shape, 'image shape')

train_y = tf.keras.utils.to_categorical(train_y)
val_y = tf.keras.utils.to_categorical(val_y)
test_y = tf.keras.utils.to_categorical(test_y)

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_dataset = train_dataset.prefetch(batch_size * 8)
train_dataset = train_dataset.shuffle(train_features.shape[0])
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

validation_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
validation_dataset = validation_dataset.batch((batch_size // 4))

test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_dataset = test_dataset.batch((batch_size // 4))


# Train static model
model = LeNet(num_classes = number_of_category)

model.compile(loss=tf.losses.categorical_crossentropy,
              optimizer=tf.optimizers.SGD(learning_rate=1e-4, decay=1e-6, momentum=9e-1),
              metrics=['accuracy'])

start = time.time()
history = model.fit(train_dataset,
                     epochs = 500,
                     validation_data=validation_dataset,
                     verbose = 1)

end = time.time()
print("執行時間：%f 秒" % (end - start))

pd.DataFrame(history.history['loss']).to_csv('Tensor_LeNet_loss.csv')
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

# plt.savefig('Tensor_LeNet_loss.png')
plt.close()

val_result = model.evaluate(validation_dataset)
test_result = model.evaluate(test_dataset)
result = [val_result[1], test_result[1]]

pd.DataFrame(result).to_csv('Tensor_LeNet_model_result.csv')

