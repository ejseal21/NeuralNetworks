'''adaline.py
Ethan Seal and Cole Turner
CS343: Neural Networks
Project 1: Single Layer Networks
ADALINE (ADaptive LInear NEuron) neural network for classification and regression
'''
import numpy as np
from adaline import Adaline


class AdalineLogistic(Adaline):
    def __init__(self, n_epochs=2000, learning_rate=0.001):
        Adaline.__init__(self, n_epochs, learning_rate)

    def activation(self, net_in):

        #use the new activation function, sigmoid
        output =  1 / (1 + np.exp(-net_in))

        return output

    def predict(self, features):


        net_in = self.net_input(features)
        activations = self.activation(net_in)
        activations[activations < 0.5] = 0
        activations[activations >= 0.5] = 1
        return activations.astype(int)

    def compute_loss(self, y, net_act):
        #expecting loss to have shape [num samples, ] and be the loss for each input
        loss= np.sum(-y * np.log(net_act) - (1-y) * np.log(1-net_act))
        return loss
    
