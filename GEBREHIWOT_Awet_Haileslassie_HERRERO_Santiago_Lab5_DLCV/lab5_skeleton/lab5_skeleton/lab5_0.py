# -*- coding: utf-8 -*-
from __future__ import print_function
import time, random, datetime, gc
from src.functions import *
from src.model import *
from src.data import *
#import tensorflow as tf
#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
#from keras.optimizers import RMSprop
#
## deeper cnn model for mnist
#from numpy import mean
#from numpy import std
#from matplotlib import pyplot
#from keras.utils import to_categorical
#from keras.layers import Flatten
#from keras.optimizers import SGD
#
#import matplotlib.pyplot as plt
#import numpy as np
#
#print('tensorflow:', tf.__version__)
#print('keras:', keras.__version__)



if __name__ == "__main__":
    make_path('/espace/DLCV')
    extract_RGB(data_path='/net/ens/DeepLearning/lab5/Data_TP/Videos', output_path='/espace/DLCV2/Data')
    stat_dataset(path='/espace/DLCV2/Data')
    compute_flow(data_path='/espace/DLCV2/Data', flow_calculation=True)


    
