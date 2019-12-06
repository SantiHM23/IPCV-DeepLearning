import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#In this first part, we just prepare our data (mnist) 
#for training and testing

import keras
from keras.datasets import mnist
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

W = random_float_array =np.random.uniform(-0.05, 0.05, size=(1, num_pixels)) 

#print(X_train.shape)
#print(W.shape)
Avg_loss_train=0
Avg_loss_test=0
b = 0.0001
alpha = 1
maxepochs = 5000
lossfctn_test = []
lossfctn_train = []
nepochs = []

def forward(X, y, W, b):
    z = np.matmul(W, X)
    z = z + np.ones(z.shape) * b    
    aa = sigmoid(z)
    Loss = loss(aa, y)
    return aa, Loss

def backward(X, a, y, W, b):
    dw = (1.0/m) * np.matmul(X , (a - y).T)
    db = (1.0/m) * np.sum(a - y)
    W = W - alpha * dw.T
    b = b - alpha * db
    return W, b
# =============================================================================
# #WORK 
for i in range(0,  maxepochs):
    a_test, Loss_test = forward(X_test, y_test, W, b)
    a_train, Loss_train = forward(X_train, y_train, W, b)
    
    W, b = backward(X_train , a_train, y_train, W, b)
    #print(b)
    Avg_loss_train = np.sum(Loss_train)/len(Loss_train[0,:])
    Avg_loss_test = np.sum(Loss_test)/len(Loss_test[0,:])
    if i % 50 == 0:
        #print(Avg_loss_train)
        print(i,Avg_loss_test)
   
    lossfctn_train.append(Avg_loss_train)
    nepochs.append(i)
    lossfctn_test.append(Avg_loss_test)
# =============================================================================
# PAINT THE EVOLUTION OF THE LOSS FUNCTION FOR 10000 EPOCHS starting from 10th epoch
plt.figure()
plt.plot(nepochs[10:maxepochs], lossfctn_train[10:maxepochs], label='Train Loss density function')
plt.plot(nepochs[10:maxepochs], lossfctn_test[10:maxepochs], label='Test Loss density function')
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='-')
# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('epochs')
plt.ylabel('Loss values')
plt.legend()
plt.title('Evolution of the Loss density function')
plt.savefig("ldf.png")

# PAINT THE EVOLUTION OF THE LOSS FUNCTION FOR 10000 EPOCHS starting from 1000th epoch for better view (ZOOM)
plt.figure()
plt.plot(nepochs[300:maxepochs], lossfctn_train[300:maxepochs], label='Train Loss density function')
plt.plot(nepochs[300:maxepochs], lossfctn_test[300:maxepochs], label='Test Loss density function')
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='-')
# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('epochs')
plt.ylabel('Loss values')
plt.legend()
plt.title('Zoomed View Evolution of the Loss density function(Zoomed )')
plt.savefig("ldf_Zoom_view.png")