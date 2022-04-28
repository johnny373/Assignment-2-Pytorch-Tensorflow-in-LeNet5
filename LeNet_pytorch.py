# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 19:23:05 2022

@author: Johnny
"""
import pandas as pd
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.utils import shuffle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def load_pic_data_by_txt(pic_load_list):
    tmp_pic_w, tmp_pic_h=  cv2.imread(pic_load_list[0], cv2.IMREAD_GRAYSCALE).shape
    tmp_size = 256
    batch_size = len(pic_load_list)
    tmp_x_array = np.zeros(batch_size * 1 * 32 * 32).reshape((batch_size, 1, 32, 32))
        
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

def pic_channel_reshape(image):
    h, w, d = image.shape
    rechannel_image = np.append(np.append(image[:,:,0], image[:,:,1]), image[:,:,2]).reshape(d, h, w)
    return rechannel_image

class LeNet(nn.Module):
	def __init__(self):
		super(LeNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, (5,5), padding=0)
		self.conv2 = nn.Conv2d(6, 16, (5,5))
		self.fc1   = nn.Linear(16*5*5, 120)
		self.fc2   = nn.Linear(120, 84)
		self.fc3   = nn.Linear(84, 50)
	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
		x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

# Load Data

batch_size = 32
number_of_category = 50

train_txt = pd.read_csv('train.txt', header = None, names = ['pic_path', 'label'], sep = ' ')
val_txt = pd.read_csv('val.txt', header = None, names = ['pic_path', 'label'], sep = ' ')
test_txt = pd.read_csv('test.txt', header = None, names = ['pic_path', 'label'], sep = ' ')

val_x, val_y = get_batch_data(val_txt, len(val_txt), number_of_category)
test_x, test_y = get_batch_data(test_txt, len(test_txt), number_of_category)

# LeNet5

net = LeNet()
print (net)

use_gpu = torch.cuda.is_available()

if use_gpu:
	net = net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

print ("Training phase")
nb_epoch = 989375
nb_index = 0
nb_batch = 32
losses = []

for epoch in range(nb_epoch):
        
    batch_x, batch_y = get_batch_data(train_txt, batch_size, number_of_category)
    batch_x = torch.from_numpy(batch_x)
    batch_y = torch.from_numpy(batch_y)
    
    mini_data  = Variable(batch_x.clone())
    mini_label = Variable(batch_y.clone(), requires_grad = False)
    mini_data  = mini_data.type(torch.FloatTensor)
    mini_label = mini_label.type(torch.LongTensor)

    if use_gpu:
        mini_data  = mini_data.cuda()
        mini_label = mini_label.cuda()
    
    optimizer.zero_grad()
    mini_out   = net(mini_data)
    mini_label = mini_label.view(batch_size)
    mini_loss  = criterion(mini_out, mini_label)
    mini_loss.backward()
    optimizer.step() 

    if (epoch + 1) % 1978 == 0:
        print("Epoch:%d/500, Loss = %f" %((epoch+1)//1978, mini_loss.item()))
        losses.append(mini_loss.item())

draw_losses(losses)
pd.DataFrame(losses).to_csv('Pytorch_LeNet5_loss.csv')
# LeNet5 val

val_x = torch.from_numpy(val_x)
nb_test = val_x.shape[0]

net.eval()

for each_sample in range(nb_test):
    sample_data = Variable(val_x.clone())
    sample_data = sample_data.type(torch.FloatTensor)
    if use_gpu:
        sample_data = sample_data.cuda()
    sample_out = net(sample_data)
    _, pred = torch.max(sample_out, 1)
    
    pred = pred.cpu().clone().numpy()

val_count = 0
for i in range(len(val_y)):
    if val_y[i] == pred[i]:
        val_count += 1

# LeNet5 test

test_x = torch.from_numpy(test_x)
nb_test = val_x.shape[0]

net.eval()

for each_sample in range(nb_test):
    sample_data = Variable(test_x.clone())
    sample_data = sample_data.type(torch.FloatTensor)
    if use_gpu:
        sample_data = sample_data.cuda()
    sample_out = net(sample_data)
    _, pred = torch.max(sample_out, 1)
    
    pred = pred.cpu().clone().numpy()

test_count = 0
for i in range(len(test_y)):
    if test_y[i] == pred[i]:
        test_count += 1

# LeNet5 val & test
result = [val_count, test_count]

pd.DataFrame(result).to_csv('Pytorch_LeNet5_model_result.csv')
