'''adaline.py
Ethan Seal and Cole Turner
CS343: Neural Networks
Project 1: Single Layer Networks
ADALINE (ADaptive LInear NEuron) neural network for classification and regression
'''
import numpy as np

class Adaline():
    ''' Single-layer neural network

    Network weights are organized [bias, wt1, wt2, wt3, ..., wtM] for a net with M input neurons.
    '''
    def __init__(self, n_epochs=1000, learning_rate=0.001):
        '''
        Parameters:
        ----------
        n_epochs: (int)
            Number of epochs to use for training the network
        learning_rate: (float)
            Learning rate used in weight updates during training
        '''
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate

        # Network weights: Bias is stored in self.wts[0], wt for neuron 1 is at self.wts[1],
        # wt for neuron 2 is at self.wts[2], ...
        self.wts = None
        # Record of training loss. Will be a list. Value at index i corresponds to loss on epoch i.
        self.loss_history = None
        # Record of training accuracy. Will be a list. Value at index i corresponds to acc. on epoch i.
        self.accuracy_history = None

    def get_wts(self):
        ''' Returns a copy of the network weight array'''
        return self.wts.copy()

    def get_num_epochs(self):
        ''' Returns the number of training epochs'''
        return self.n_epochs

    def get_learning_rate(self):
        ''' Returns the learning rate'''
        return self.learning_rate

    def set_learning_rate(self, learning_rate):
        '''Updates the value of self.learning_rate (learning rate)'''
        self.learning_rate = learning_rate

    def net_input(self, features):
        ''' Computes the net_input (weighted sum of input features,  wts, bias)

        NOTE: bias is the 1st element of self.wts. Wts for input neurons 1, 2, 3, ..., M occupy
        the remaining positions.

        Parameters:
        ----------
        features: ndarray. Shape = [Num samples, Num features]
            Collection of input vectors.

        Returns:
        ----------
        The net_input. Shape = [Num samples,]
        '''    


        net_input = np.dot(features, self.wts[1:]) 
        net_input += self.wts[0]


        return net_input

    def activation(self, net_in):
        '''
        Applies the activation function to the net input and returns the output neuron's activation.
        It is simply the identify function for vanilla ADALINE: f(x) = x

        Parameters:
        ----------
        net_in: ndarray. Shape = [Num samples,]

        Returns:
        ----------
        net_act. ndarray. Shape = [Num samples,]
        '''
        return net_in

    def compute_loss(self, y, net_act):
        ''' Computes the Sum of Squared Error (SSE) loss (over a single training epoch)

        Parameters:
        ----------
        y: ndarray. Shape = [Num samples,]
            True classes corresponding to each input sample in a training epoch (coded as -1 or +1).
        net_act: ndarray. Shape = [Num samples,]
            Output neuron's activation value (after activation function is applied)

        Returns:
        ----------
        float. The SSE loss (across a single training epoch).
        '''
        
        #expecting loss to have shape [num samples, ] and be the loss for each input
        loss = 0.5 * np.sum((y - net_act) ** 2)
        return loss
        

    def compute_accuracy(self, y, y_pred):
        ''' Computes accuracy (proportion correct) (across a single training epoch)

        Parameters:
        ----------
        y: ndarray. Shape = [Num samples,]
            True classes corresponding to each input sample in a training epoch  (coded as -1 or +1).
        y_pred: ndarray. Shape = [Num samples,]
            Predicted classes corresponding to each input sample (coded as -1 or +1).

        Returns:
        ----------
        The accuracy for each input sample in the epoch. ndarray. Shape = [Num samples,]
            Expressed as proportions in [0.0, 1.0]
        '''
        arr, counts = np.unique(np.equal(y, y_pred), return_counts=True) #calculate the unique values, and the number of times they occur

        if arr[0]: #if the first unique value is True, then compute proportion correct
            return counts[0]/y.size
        else: #otherwise, compute 1-prportion mismatched
            return 1-counts[0]/y.size 
        

    def gradient(self, errors, features):
        ''' Computes the error gradient of the loss function (for a single epoch).
        Used for backpropogation.

        Parameters:
        ----------
        errors: ndarray. Shape = [Num samples,]
            Difference between class and output neuron's activation value
        features: ndarray. Shape = [Num samples, Num features]
            Collection of input vectors.

        Returns:
        ----------
        grad_bias: float.
            Gradient with respect to the bias term
        grad_wts: ndarray. shape=(Num features,).
            Gradient with respect to the neuron weights in the input feature layer
        '''


        grad_bias = np.sum(errors)

        grad_wts = np.sum(np.multiply(np.expand_dims(errors, 1), features), axis = 0) 



        return grad_bias, grad_wts

    def predict(self, features):
        '''Predicts the class of each test input sample

        Parameters:
        ----------
        features: ndarray. Shape = [Num samples, Num features]
            Collection of input vectors.

        Returns:
        ----------
        The predicted classes (-1 or +1) for each input feature vector. Shape = [Num samples,]

        NOTE: Remember to apply the activation function!
        '''
        
      
        activations = self.activation(self.net_input(features))
        activations[activations < 0] = -1
        activations[activations >= 0] = 1
        return activations.astype(int)

    def fit(self, features, y, early_stopping=False, loss_tol=0.1):
        ''' Trains the network on the input features for self.n_epochs number of epochs

        Parameters:
        ----------
        features: ndarray. Shape = [Num samples, Num features]
            Collection of input vectors.
        y: ndarray. Shape = [Num samples,]
            Classees corresponding to each input sample (coded -1 or +1).

        Returns:
        ----------
        self.loss_history: Python list of network loss values for each epoch of training.
            Each loss value is the loss over a training epoch.
        self.acc_history: Python list of network accuracy values for each epoch of training
            Each accuracy value is the accuracy over a training epoch.

        TODO:
        1. Initialize the weights according to a Gaussian distribution centered
            at 0 with standard deviation of 0.01. Remember to initialize the bias in the same way.
        2. Write the main training loop where you:
            - Pass the inputs in each training epoch through the net.
            - Compute the error, loss, and accuracy (across the entire epoch).
            - Do backprop to update the weights and bias.
        '''
        self.wts = np.random.normal(0, 0.01, features.shape[1]+1)

        loss_history = []
        accuracy_history = []

        accuracy = 0
        loss = 0
        for epoch in range(self.n_epochs):
            
            
            #pass the inputs through
            net_in = self.net_input(features) # compute netact
            activation = self.activation(net_in)
            predictions = self.predict(features)

            #compute error, loss, and accuracy
            accuracy = self.compute_accuracy(y,predictions)
            loss = self.compute_loss(y, activation)

            error = y - activation
            if early_stopping and (epoch > 1):
                if abs(loss-loss_history[-1]) < loss_tol:
                    print("epoch:", epoch, "\nloss difference:", loss-loss_history[-1])
                    break
 
            #store the loss and accuracy values
            loss_history.append(loss)
            accuracy_history.append(accuracy)

            grad_bias, grad_wts = self.gradient(error, features)
            
            #backprop
            self.wts[1:] = self.wts[1:] + self.learning_rate * grad_wts
            self.wts[0] = self.wts[0] + self.learning_rate * grad_bias
            
        self.loss_history = loss_history
        self.accuracy_history = accuracy_history
        return loss_history, accuracy_history
