# -*- coding: utf-8 -*-
from __future__ import print_function
import time, random, datetime, gc
from src.functions import *
from src.model import *
from src.data import *




if __name__ == "__main__":

    data_type='Flow'
    #data_type='RGB'

    flow_normalization = get_flow_normalization(visualization = 1)
    class_index, train_list, test_list = define_list(train_file = '/net/ens/DeepLearning/lab5/Data_TP/list/trainlist.txt',
                                                     test_file = '/net/ens/DeepLearning/lab5/Data_TP/list/testlist.txt',
                                                     class_file ='/net/ens/DeepLearning/lab5/Data_TP/list/classInd.txt')

    width = 100; height = 100; temporal_dim = 100; nb_class = len(class_index); batch_size = 10; nb_epoch = 30
    random.seed(1)


    test_generator = My_Data_Sequence_one_branch(test_list, flow_normalization, class_index, batch_size, data_type=data_type, shuffle=False)
    channels = 3
    weights_file='/net/ens/DeepLearning/lab5/Data_TP/models/RGB_model.hdf5'
    if data_type == 'Flow':
        channels = 2
        weights_file='/net/ens/DeepLearning/lab5/Data_TP/models/Flow_model.hdf5'


    #TODO
    #1) create 'one branch' model
    The_Model = make_one_branch_model(temporal_dim, width, height, channels, nb_class)
    #2) load weights file
    The_Model.load_weights(weights_file)
    #3) evaluate model 
    score = The_Model.evaluate_generator(test_generator, verbose=0)
    print(score[0])
    print(score[1])
