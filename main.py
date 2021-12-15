# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 17:30:02 2021

@author: KOUHOU
"""

from multyLayer_neural_network import MultyLayerNN
from functions import *
from layer import *
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


model = MultyLayerNN(1,0,2,"relu",1,"sigmoid")

X = np.array([0.1,0.2])
y = np.array([0.03])
y=0.03
model.fit(X,y,10000,0.1)


rng = np.random.RandomState(0)
xor_input = rng.randn(300, 2)
xor_output = np.array(np.logical_xor(xor_input[:, 0] > 0, xor_input[:, 1] > 0), dtype=int)
plt.figure(figsize=(10,8))
plot_decision_regions(X=xor_input, y=xor_output, clf=model, legend=2)
plt.title("XOR MLNN from scratch")
plt.show()


