#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#In this first part, we just prepare our data (mnist) 
#for training and testing

#==============================================================================
import keras
from keras.datasets import mnist
#==============================================================================
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#==============================================================================
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)
# print(X_train.dtype)
# print(y_train.dtype)
# print(X_test.dtype)
# print(y_test.dtype)
#==============================================================================

#RESHAPING OF THE TRAIN AND TEST SETS
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).T
X_test = X_test.reshape(X_test.shape[0], num_pixels).T
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
X_train  = X_train / 255
X_test  = X_test / 255

#==============================================================================
# print(y_train.shape)
# print(y_train.dtype)
#==============================================================================
hidlayers_units = 64
#We want to have a binary classification: digit 0 is classified 1 and 
#all the other digits are classified 0

y_new = np.zeros(y_train.shape)
y_new[np.where(y_train==0.0)[0]] = 1
y_train = y_new

y_new = np.zeros(y_test.shape)
y_new[np.where(y_test==0.0)[0]] = 1
y_test = y_new


y_train = y_train.T
y_test = y_test.T


m = X_train.shape[1] #number of examples
m2 = X_test.shape[1]
#print(m)
#Now, we shuffle the training set
np.random.seed(138)
shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[:,shuffle_index], y_train[:,shuffle_index]
#print(np.sum(X_train))

#Now, we shuffle the test set
np.random.seed(138)
shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[:,shuffle_index], y_train[:,shuffle_index]
#print(X_train.shape)
#Display one image and corresponding label 
import matplotlib
import matplotlib.pyplot as plt
#i = 20
#print('y[{}]={}'.format(i, y_train[:,i]))
#plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
#plt.axis("off")
#plt.show()

#z = np.arange(-12.0, 12.0, 1.0, dtype = np.float32)


#####TO COMPLETE
def sigmoid(z):
    a= 1.0/(1+ np.exp(-z))
    return a

def loss(a,y):
    L = -(y*np.log(a+0.0001) + (1-y)*np.log(1-a+0.0001))
    return L

#Let start our work: creating a neural network
#First, we just use a single neuron. 

W1 = random_float_array =np.random.uniform(-0.05, 0.05, size=(hidlayers_units, num_pixels)) 
W2 = random_float_array =np.random.uniform(-0.05, 0.05, size=(1, hidlayers_units)) 
#print(X_train.shape)
#print(W.shape)
Avg_loss_train=0
Avg_loss_test=0
b1 = 0.0001
b2 = 0.0002
alpha = 0.8
maxepochs = 500
lossfctn_test = []
lossfctn_train = []
accuracy_test = []
accuracy_train = []

nepochs = []

def forward(X, y, W1, W2, b1, b2):
    Z1 = np.matmul(W1, X)
    Z1 = Z1 + np.ones(Z1.shape) * b1    
    A1 = sigmoid(Z1)
    Z2 = np.matmul(W2, A1)
    Z2 = Z2 + np.ones(Z2.shape) * b2 
    A2 = sigmoid(Z2)
    Loss = loss(A2, y)
    return A2, A1, Loss

def backward(X, A2, A1, y, W2, W1, b2, b1):
    dw2 = (1.0/m) * np.matmul((A2 - y) , A1.T)
    db2 = (1.0/m) * np.sum(A2 - y)
    dA1 = np.matmul(W2.T, (A2 - y))
    sigprime = np.multiply(A1, (1-A1))
    dZ1 = np.multiply(dA1, sigprime)
    dw1 = (1.0/m) * np.matmul(dZ1 , X.T)
    db1 = (1.0/m) * np.sum(dZ1)
    W2 = W2 - alpha * dw2
    #print(W2.shape)
    b2 = b2 - alpha * db2
    W1 = W1 - alpha * dw1
    b1 = b1 - alpha * db1
    return W1, b1, W2, b2
# =============================================================================
# #WORK 
for i in range(1,  maxepochs+1):
    A2_test, A1_test, Loss_test = forward(X_test, y_test, W1, W2, b1, b2)
    A2_train, A1_train, Loss_train = forward(X_train, y_train, W1, W2, b1, b2)
    
    W1, b1, W2, b2 = backward(X_train , A2_train, A1_train, y_train, W2, W1, b2, b1)
    #print(b)
    
    prediction_test = A2_test
    prediction_train = A2_train
    
    #print(prediction.shape,A2_test.shape,y_test.shape)
    #print(prediction.shape, y_test.shape)
    p_in_test = prediction_test >= 0.5
    p_in_train = prediction_train >= 0.5
    y_in_test = y_test
    y_in_train = y_train
  
    total_correct_train = p_in_train == y_in_train
    total_correct_test = p_in_test == y_in_test
  
    #print(total_correct_test.shape,sum(total_correct_test[0]))

    accu_test = sum(total_correct_test[0])/len(y_test[0])
    accu_train = sum(total_correct_train[0])/len(y_train[0])
    Avg_loss_train = np.sum(Loss_train)/len(Loss_train[0,:])
    Avg_loss_test = np.sum(Loss_test)/len(Loss_test[0,:])
    if i % 50 == 0:
        #print(Avg_loss_train)
        print(i,Avg_loss_test)
        print(accu_test)
   
    lossfctn_train.append(Avg_loss_train)
    accuracy_train.append(accu_train)
    accuracy_test.append(accu_test)
    lossfctn_test.append(Avg_loss_test)
    nepochs.append(i)
# =============================================================================
# PAINT THE EVOLUTION OF THE LOSS FUNCTION FOR 10000 EPOCHS starting from 10th epoch
plt.figure()
plt.plot(nepochs[1:maxepochs], lossfctn_train[1:maxepochs], label='Train Loss density function')
plt.plot(nepochs[1:maxepochs], lossfctn_test[1:maxepochs], label='Test Loss density function')
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='-')
# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('epochs')
plt.ylabel('Loss values')
plt.legend()
plt.title('Evolution of the Loss density function')
plt.savefig("ldf_2_2.png")


# PAINT THE EVOLUTION OF THE ACCURACY FUNCTION FOR 1000 EPOCHS starting from 10th epoch
plt.figure()
plt.plot(nepochs[1:maxepochs], accuracy_train[1:maxepochs], label='Train accuracy function')
plt.plot(nepochs[1:maxepochs], accuracy_test[1:maxepochs], label='Test accuracy function')
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='-')
# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('epochs')
plt.ylabel('Accuarcy values')
plt.legend()
plt.title('Evolution of the accuracy density function')
plt.savefig("accuracydf_2_2.png")

