'''optimizer.py
Algorithms to optimize the weights during gradient descent / backprop
YOUR NAMES HERE
CS343: Neural Networks
Project 3: Convolutional Neural Networks
'''
import numpy as np
np.random.seed(0)


class Optimizer():
    def __init__(self):
        self.wts = None
        self.d_wts = None

    def prepare(self, wts, d_wts):
        '''Stores weights and their gradient before an update step is performed.
        '''
        
        self.wts = wts
        self.d_wts = d_wts

    def update_weights(self):
        pass

    @staticmethod
    def create_optimizer(name, **kwargs):
        '''
        Factory method that takes in a string, and returns a new object of the
        desired type. Called via Optimizer.create_optimizer().
        '''
        if name.lower() == 'sgd':
            return SGD(**kwargs)
        elif name.lower() == 'sgd_momentum':
            return SGD_Momentum(**kwargs)
        elif name.lower() == 'adam':
            return Adam(**kwargs)
        elif name.lower() == 'rmsprop':
            return RMSProp(**kwargs)
        else:
            raise ValueError('Unknown optimizer name!')


class RMSProp(Optimizer):
    '''Update weights using RMSProp update rule
        '''
    def __init__(self, lr, beta):
        '''
        Paramters:
        ----------
        lr: float > 0. Learning rate.
        beta: float 0 <= beta <= 1
        '''
        
        self.lr = lr
        self.beta = beta
        self.moving_avg = 0.0

    def update_weights(self):
        '''
        Updates weights according to RMSProp and returns a deep copy of the 
        updated weights for this time step.

        Returns:
        ----------
        A copy of the updated weights for this time step

        Citation: https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a
        '''
        self.moving_avg = self.beta * self.moving_avg + (1 - self.beta) * self.d_wts * self.d_wts
        self.wts = self.wts - (self.lr / sqrt(self.moving_avg)) * self.d_wts
        return self.wts.copy()

class SGD(Optimizer):
    '''Update weights using Stochastic Gradient Descent (SGD) update rule.
    '''
    def __init__(self, lr=0.1):
        '''
        Parameters:
        -----------
        lr: float > 0. Learning rate.
        '''
        self.lr = lr

    def update_weights(self):
        '''Updates the weights according to SGD and returns a deep COPY of the
        updated weights for this time step.

        Returns:
        -----------
        A COPY of the updated weights for this time step.
        '''

        self.wts = self.wts - self.lr*self.d_wts
        return self.wts.copy()
        


class SGD_Momentum(Optimizer):
    '''Update weights using Stochastic Gradient Descent (SGD) with momentum
    update rule.
    '''
    def __init__(self, lr=0.001, m=0.9):
        '''
        Parameters:
        -----------
        lr: float > 0. Learning rate.
        m: float 0 < m < 1. Amount of momentum from gradient on last time step.
        '''
        self.lr = lr
        self.m = m
        self.velocity = None

    def update_weights(self):
        '''Updates the weights according to SGD with momentum and returns a
        deep COPY of the updated weights for this time step.

        Returns:
        -----------
        A COPY of the updated weights for this time step.

        TODO: Write the SGD with momentum weight update rule.
        See notebook for review of equations.
        '''
        if self.velocity is None:
            self.velocity = 0

        self.velocity = (self.m * self.velocity) - (self.lr*self.d_wts)
        self.wts = self.wts + self.velocity
        
        return self.wts.copy()


class Adam(Optimizer):
    '''Update weights using the Adam update rule.
    '''
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, t=0):
        '''
        Parameters:
        -----------
        lr: float > 0. Learning rate.
        beta1: float 0 < m < 1. Amount of momentum from gradient on last time step.
        beta2: float 0 < m < 1. Amount of momentum from gradient on last time step.
        eps: float. Small number to prevent division by 0.
        t: int. Records the current time step: 0, 1, 2, ....
        '''
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = t

        self.m = None
        self.velocity = None

    def update_weights(self):
        '''Updates the weights according to Adam and returns a
        deep COPY of the updated weights for this time step.

        Returns:
        -----------
        A COPY of the updated weights for this time step.

        TODO: Write the Adam update rule
        See notebook for review of equations.

        Hints:
        -----------
        - Remember to initialize m and v.
        - Remember that t should = 1 on the 1st wt update.
        - Remember to update/save the new values of m, v between updates.
        '''
        
        try: #comparing an array to None throws an error, but if it throws an error, then it is not None
            if self.m == None:
                self.m = np.zeros(self.wts.shape)
            if self.velocity == None:
                self.velocity = np.zeros(self.wts.shape)
        except:
            pass       
        self.t += 1
        
        self.m = (self.beta1 * self.m) + (1- self.beta1) * self.d_wts
        self.velocity = (self.beta2 * self.velocity) + (1-self.beta2) * self.d_wts * self.d_wts
        n = self.m / (1-(self.beta1**self.t))
        u = self.velocity / (1-(self.beta2**self.t))
        self.wts = self.wts - (self.lr * n) / (np.sqrt(u) + self.eps)
        return self.wts