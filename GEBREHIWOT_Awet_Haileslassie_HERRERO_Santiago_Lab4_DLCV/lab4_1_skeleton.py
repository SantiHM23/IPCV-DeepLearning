from __future__ import print_function

import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop

import matplotlib.pyplot as plt
import numpy as np


print('tensorflow:', tf.__version__)
print('keras:', keras.__version__)


#load (first download if necessary) the MNIST dataset
# (the dataset is stored in your home direcoty in ~/.keras/datasets/mnist.npz
#  and will take  ~11MB)
# data is already split in train and test datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train : 60000 images of size 28x28, i.e., x_train.shape = (60000, 28, 28)
# y_train : 60000 labels (from 0 to 9)
# x_test  : 10000 images of size 28x28, i.e., x_test.shape = (10000, 28, 28)
# x_test  : 10000 labels
# all datasets are of type uint8
print('x_train.shape=', x_train.shape)
print('y_test.shape=', y_test.shape)

#To input our values in our network Conv2D layer, we need to reshape the datasets, i.e.,
# pass from (60000, 28, 28) to (60000, 28, 28, 1) where 1 is the number of channels of our images
img_rows, img_cols = x_train.shape[1], x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

#Convert to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalize inputs from [0; 255] to [0; 1]
x_train = x_train / 255
x_test = x_test / 255

print('x_train.shape=', x_train.shape)
print('x_test.shape=', x_test.shape)

num_classes = 10

#Convert class vectors to binary class matrices ("one hot encoding")
## Doc : https://keras.io/utils/#to_categorical
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# num_classes is computed automatically here
# but it is dangerous if y_test has not all the classes
# It would be better to pass num_classes=np.max(y_train)+1



#Let start our work: creating a convolutional neural network

#####TO COMPLETE

# model building
model = Sequential()
#convolutional layer with rectified linear activation unit
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(img_rows,img_cols,1)))
#32 convolution filters used each of size 3x3
#choose the best featurs used via pooling
model.add(MaxPooling2D(pool_size=(2,2)))

# randomly turn neurons on and off to improve convergence
model.add(Dropout(0.25))


model.add(Conv2D(64, (3,3), activation='relu'))
#64 convolution filters used each of size 3x3

#choose the best featurs used via pooling
model.add(MaxPooling2D(pool_size=(2,2)))

# randomly turn neurons on and off to improve convergence
model.add(Dropout(0.3))


#flatten since too many dimensions, but  we only want a classfication output
model.add(Flatten())

#fully connected to get all output data
model.add(Dense(128, activation ='relu'))

#softmax activation function gor the output probablities
model.add(Dense(num_classes, activation='softmax'))

#Adaptive learning rate "adaDelta" popular form of sgd, and categorical cross entropy
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adaDelta', metrics = ['accuracy'])

batch_size = 128
num_epochs = 50

history = model.fit(x_train, y_train, batch_size=batch_size, epochs= num_epochs, verbose=1, validation_split=0.2)

score = model.evaluate(x_test,y_test,verbose=0)
#Test 
print('Test loss:', score[0])
print('Test accuracy:',score[1])



''''
##################################################################

# Note that the in order to plot  accuracy, the index of  model_log.history[]) could be 'val_accuarcy' or val_acc' , 'accuracy' or 'acc' 
depending on the tTensor flow Vesion, Please make sure to check it? other wise comment it out.

##################################################################
'''

# summarize history for accuracy
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("acc4_1.png")

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("ldf4_1.png")
plt.show()
