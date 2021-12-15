# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 17:35:38 2021

@author: DELL
"""
import numpy as np

from random import random, seed

""" This file contains all functions used in the programm, 
    espacially activate function and their derivatives
"""

# function to calculate sum
def my_sum(W,X,bias):
    return np.dot(W,X) + bias

# sigmoid function 
def sigmoid(x):
    return 1/(1+np.exp(-x))

# sigmoid derivative function
def sigmoid_derivative(x):
    return x*(1-x)
# relu function
def relu(x):
    result = []
    for i in x:
        if i <= 0:
            result.append(0)
        else:
            result.append(i)
    return np.array(result)
    
# derivative of relu function
def relu_derivative(x):
    result = []
    for i in x:
        if i <= 0:
            result.append(0)
        else:
            result.append(0)
    return np.array(result)

"""This function is a switcher to get the function chosen by the user to activate the laysers"""

def switch(argument):
    switcher = {
        "sigmoid":{"function":sigmoid,"derivative":sigmoid_derivative},
        "relu":{"function":relu,"derivative":relu_derivative}
    }
    return switcher.get(argument ,None)


def initilize_weights_bias(input_dim:int=2, hidden_dim:int=2,output_dim:int=2, weights_initializer:int=1, bias_initializer:int=1):
    #print(input_dim)    
    if weights_initializer==1 or weights_initializer==0 :
            weights_1 = []
            weights_2 = []
            for _ in range(hidden_dim):
                row = []
                for _ in range(input_dim):
                    row.append(weights_initializer)
                weights_1.append(row)
                
            for _ in range(output_dim):
                row = []
                for _ in range(hidden_dim):
                    row.append(weights_initializer)
                weights_2.append(row)
                    
        # if the user want to generate weights randomly
    elif weights_initializer==-1:
        seed(1)
        weights_1 = []
        weights_2 = []
        for _ in range(hidden_dim):
            row = []
            for _ in range(input_dim):
                row.append(random())
            weights_1.append(row)
                
        for _ in range(output_dim):
            row = []
            for _ in range(hidden_dim):
                row.append(random())
            weights_2.append(row)
    else:
        raise Exception("weights_initialize must be 0 or 1 or -1")
    # test bias initiaize
        
        
    # generate an array of input and output bias
    if bias_initializer==1 or bias_initializer==0 :
        # input bias
        bias1 = []
        for _ in range(hidden_dim):
            bias1.append(bias_initializer)
                
        # output bias
        bias2 = []
        for _ in range(output_dim):
            bias2.append(bias_initializer)
            
    # if the user want generate bias randomly 
    elif bias_initializer==-1:
        # input bias
        bias1 = []
        for _ in range(input_dim):
            bias1.append(random())
                
        # output bias
        bias2 = []
        for _ in range(hidden_dim):
            bias2.append(random())
    else:
        raise Exception("bias_initialize must be 0 or 1 or -1")
    
    #print(np.array(weights_1), np.array(weights_2), np.array(bias1), np.array(bias2))        
    return np.array(weights_1), np.array(weights_2), np.array(bias1), np.array(bias2)