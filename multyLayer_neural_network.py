# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 17:14:06 2021

@author: KOUHOU
"""
from layer import *
import numpy as np
from random import random, seed
from functions import *

class MultyLayerNN:
    """
        weights_initialize : value to initialize weights : 0 or 1, if -1 they will be random 
        bias_initialize : value to initialize bias : 0 or 1, if -1 they will be random 
        dim_hidden : number of neurons in the hidden layer
        dim_output : number of neurons in the output layer
        activate_fct hidden : function to use t activate neurons in hidden layer
        activate_fct output : function to use t activate neurons in output layer
        
    """
    
    # Constructor 
    def __init__(self,weights_initialize:int=0, bias_initialize:int=1, dim_hidden:int=2,activate_fct_hidden:str="relu", dim_output:int=1, activate_fct_output:str="sigmoid"):
        # Class attributes 
        self.weights_initializer = weights_initialize
        self.bias_initializer = bias_initialize
        self.hidden_dim = dim_hidden
        self.output_dim = dim_output
        self.weights_1 = np.array([], dtype=float)
        self.weights_2 = np.array([], dtype=float)
        self.bias1 = np.array([], dtype=float)
        self.bias2 = np.array([], dtype=float)
        
        # get the activate functions for the layers 
        # switch is a function that return a dic with to keys :  function and its derivative or None
        """ activate function for the hidden layer """
        activate_hidden = switch(activate_fct_hidden)
        if not activate_hidden:
            raise Exception("hidden activate function not found")
        else:
            self.activate_hidden = activate_hidden["function"]
            self.activate_hidden_derive = activate_hidden["derivative"]
            
        """ activate function for the output layer """
        activate_output = switch(activate_fct_output)
        if not activate_output:
            raise Exception("output activate function not found")
        else:
            self.activate_output = activate_output["function"]
            self.activate_output_derive = activate_output["derivative"]
        
        
    # function to fit the model 
    """ the function accept as input the dataset(x_train and y_train), number of epochs and learning rate"""
    
    def fit(self, X_train:np.ndarray, y_train:np.ndarray, epochs:int=1000, learning_rate:float=0.01):
        self.weights_1, self.weights_2, self.bias1, self.bias2 = initilize_weights_bias(X_train.shape[0], self.hidden_dim, self.output_dim, self.weights_initializer, self.bias_initializer)
        for i in range(epochs):
            # calculate the sum to generate hidden(H) layer using the choosen activate function
            
            In = my_sum(self.weights_1, X_train, self.bias1) 
            
            H = np.array(self.activate_hidden(In))
            
            # calculate the second sum based on hidden layer's neurons to generate output layer (predicted)
            In3 = my_sum(self.weights_2, H, self.bias2)            
            predicted = np.array(self.activate_output(In3))
            
            # train the model
            # update weights and bias in the layers 
            self.weights_2 = self.weights_2 - learning_rate * (predicted - y_train) * self.activate_output_derive(predicted) * H
            self.bias2 = self.bias2 - learning_rate * (predicted - y_train) * self.activate_output_derive(predicted)
            
            self.weights_1 = self.weights_1 - learning_rate * (predicted - y_train) * self.activate_hidden_derive(predicted) * self.weights_2 * self.activate_hidden_derive(H) * X_train
            self.bias1 = self.bias1 - learning_rate * (predicted - y_train) * self.activate_hidden_derive(predicted) * self.weights_2 * self.activate_hidden_derive(H)
            self.bias1 = self.bias1[0] 
        
        print(predicted)
        print(self.weights_1)
        print(self.weights_2)
        
        
    # function used to predict results 
    # the input is an array of 2 value and it return the predicted value
    def predict(self,X:np.ndarray):
        # generate the hidden layer based on the result of fit function
        predicted = []
        for x in X:
            In = my_sum(np.array(self.weights_1), x, self.bias1)
            H = np.array(self.activate_hidden(In))
            
            # the output layer that present the predicted value
            In3 = my_sum(np.array([self.weights_2]), H, self.bias2)
            predicted.append(np.array(self.activate_output(In3)))
        
        return np.array(predicted)
    
    