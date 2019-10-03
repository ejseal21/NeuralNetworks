'''single_layer_net.py
Constructs, trains, tests single layer neural networks
YOUR NAMES HERE
CS343: Neural Networks
Project 2: Multilayer Perceptrons
'''
import numpy as np


class SingleLayerNet():
    '''
    SingleLayerNet is a parent class for single layer networks with potentially different loss
    functions
    '''
    def __init__(self):
        # Network weights
        self.wts = None
        # Bias
        self.b = None

    def activation(self, net_in):
        '''Override. Don't fill this in'''
        pass

    def loss(self, net_in, y, reg=0):
        '''Override. Don't fill this in'''
        pass

    def gradient(self, features, net_act, y, reg=0):
        '''Override. Don't fill this in'''
        pass

    def accuracy(self, y, y_pred):
        ''' Computes the accuracy of classified samples. Proportion correct

        Parameters:
        -----------
        y: ndarray. int-coded true classes. shape=(Num samps,)
        y_pred: ndarray. int-coded predicted classes by the network. shape=(Num samps,)

        Returns:
        -----------
        float. accuracy in range [0, 1]
        '''
        pass

    def net_in(self, features):
        ''' Computes the net input (net weighted sum)
        Parameters:
        -----------
        features: ndarray. input data. shape=(num images (in mini-batch), num features)
        i.e. shape=(N, M)

        Note: shape of self.wts = (M, C), for C output neurons

        Returns:
        -----------
        net_input: ndarray. shape=(N, C)
        '''
        # features = np.reshape(features, (15, 3072))
        net_input = np.dot(features, self.wts)
        net_input += self.b
        return net_input

    def one_hot(self, y, num_classes):
        '''One-hot codes the output classes for a mini-batch

        Parameters:
        -----------
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,C-1
        num_classes: int. Number of unique output classes total

        Returns:
        -----------
        y_one_hot: One-hot coded class assignments.
            e.g. if y = [0, 2, 1] and num_classes (C) = 4 we have:
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]
        '''
        # y = y - 1 #test_gradient was  giving an error because y was not preprocessed
        one_hot_out = np.zeros((len(y), num_classes))
        one_hot_out[np.arange(len(y)), y] = 1

        return one_hot_out.astype(int)

    def fit(self, features, y, n_epochs=10000, lr=0.0001, mini_batch_sz=256, reg=0, verbose=2):
        ''' Trains the network to data in `features` belonging to the int-coded classes `y`.
        Implements stochastic mini-batch gradient descent

        Parameters:
        -----------
        features: ndarray. shape=(Num samples, num features)
        y: ndarray. int-coded class assignments of training samples. 0,...,numClasses-1
        n_epochs: int. Number of training epochs
        lr: float. Learning rate
        mini_batch_sz: int. Batch size per training iteration. 
            i.e. Chunk this many data samples together to process with the model per training
            iteration. Then we do gradient descent and update the wts. NOT the same thing as an epoch.
        reg: float. Regularization strength used when computing the loss and gradient.
        verbose: int. 0 means no print outs. Any value > 0 prints Current iteration number and
            training loss every 100 iterations.

        Returns:
        -----------
        loss_history: Python list of floats. Recorded training loss on every mini-batch / training
            iteration.

        NOTE:
        - Recall: training epoch is not the same thing as training iteration with mini-batch.
        If we have mini_batch_sz = 100 and N = 1000, then we have 10 iterations per epoch. Epoch
        still means entire pass through the training data "on average".

        TODO:
        -----------
        1) Initialize the wts/bias to small Gaussian numbers:
            mean 0, std 0.01, Wts shape=(num_feat, num_classes), b shape=(num_classes,)


        2) Implement mini-batch support: On every iter draw from our input samples (with replacement)
        a batch of samples equal in size to `mini_batch_sz`. Also keep track of the associated labels.
        THEY MUST MATCH UP!!
            - Keep in mind that mini-batch wt updates are different than epochs. There is a parameter
              for E (num epochs), not number of iterations.
            - Handle this edge case: we do SGD and mini_batch_sz = 1. Add a singleton dimension
              so that the "N"/sample_size dimension is still defined.
        4) Our labels are int coded (0,1,2,3...) but this representation doesnt work well for piping
        signals to the C output neurons (C = num classes). Transform the mini-batch labels to one-hot
        coding from int coding (see function above to write this code).
        5) Compute the "net in".
        6) Compute the activation values for the output neurons (you can defer the actual function
        implementation of this for later).
        7) Compute the cross-entropy loss (again, you can defer the details for now)
        8) Do backprop:
            a) Compute the error gradient for the mini-batch sample,
            b) update weights using gradient descent.
                - You may need a sign flip in the update rule compared to Project 1
                  (use the "vanilla" gradient descent equation). Reason for this is that we are
                  supporting multiple net activation functions and don't want to make assumptions
                  about the gradient form.

        HINTS:
        -----------
        2) Work in indices, not data elements.
        '''

        # initialize weights.
        # self.wts = np.random.normal(0, 0.01, features.shape[1]+1)
        num_samps, num_features = features.shape
        num_classes = len(np.unique(y))

        iter_per_epoch = max(int(num_samps / mini_batch_sz), 1)
        n_iter = n_epochs * iter_per_epoch

        # TODO: Initialize wts, bias here

        loss_history = []

        if verbose > 0:
            print(f'Starting to train network...There will be {n_epochs} epochs', end='')
            print(f' and {n_iter} iterations total, {iter_per_epoch} iter/epoch.')

            # TODO: Put this inside training loop
            if i % 100 == 0 and verbose > 0:
                print(f'  Completed iter {i}/{n_iter}. Training loss: {loss:.2f}.')

        if verbose > 0:
            print('Finished training!')

        return loss_history

    def predict(self, features):
        ''' Predicts the int-coded class value for network inputs ('features').

        Parameters:
        -----------
        features: ndarray. shape=(mini-batch size, num features)

        Returns:
        -----------
        y_pred: ndarray. shape=(mini-batch size,).
            This is the int-coded predicted class values for the inputs passed in.
            Note: You can figure out the predicted class assignments from net_in (i.e. you dont
            need to apply the net activation function — it will not affect the most active neuron).
        '''
        pass

    def test_loss(self):
        '''Override. Don't fill this in'''
        pass

    def test_gradient(self):
        '''Override. Don't fill this in'''
        pass


class SingleLayerNetSoftmax(SingleLayerNet):
    '''
    Single layer network that uses the softmax activation function and cross-entropy loss.
    '''

    def activation(self, net_in):
        '''Applies the softmax activation function on the net_in.

        Parameters:
        -----------
        net_in: ndarray. net in. shape=(mini-batch size, num output neurons)
        i.e. shape=(N, C)

        Returns:
        -----------
        f_z: ndarray. net_act transformed by softmax function. shape=(N, C)

        Tips:
        -----------
        - Remember the adjust-by-the-max trick (for each input samp) to prevent numeric overflow!
        This will make the max net_in value for a given input 0.
        - np.sum and np.max have a keepdims optional parameter that might be useful for avoiding
        going from shape=(X, Y) -> (X,). keepdims ensures the result has shape (X, 1).
        '''
        net_in_max = np.max(net_in, keepdims=True)
        net_act = np.exp(net_in - net_in_max) / np.sum(np.exp(net_in - net_in_max), axis=1, keepdims=True)

        return net_act

    def loss(self, net_in, y, reg=0):
        '''Computes the cross-entropy loss

        Parameters:
        -----------
        net_in: ndarray. input to output layer. shape=(mini-batch size, num output neurons)
        i.e. shape=(N, C)
        y: ndarray. correct class values, int-coded. shape=(mini-batch size,)
        reg: float. Regularization strength

        Returns:
        -----------
        loss: float. Regularized (!!!!) average loss over the mini batch 1

        Tips:
        -----------
        - Remember that the loss is the negative of the average softmax activation values of neurons
        coding the correct classes only.
        - It is handy to use arange indexing to select only the net_act values coded by the correct
          output neurons.
        - NO FOR LOOPS!
        - Remember to add on the regularization term, which has a 1/2 in front of it.
        ''' 

        net_act = self.activation(net_in)
       
        correctActs = net_act[np.arange(net_act.shape[0]), y]

        loss = -np.mean(np.log(correctActs), axis=0) + reg*(0.5 * np.sum(np.square(self.wts)))
        return loss

    def gradient(self, features, net_act, y, reg=0):
        '''Computes the gradient of the softmax version of the net

        Parameters:
        -----------
        features: ndarray. net inputs. shape=(mini-batch-size, Num features)
        net_act: ndarray. net outputs. shape=(mini-batch-size, C)
            In the softmax network, net_act for each input has the interpretation that
            it is a probability that the input belongs to each of the C output classes.
        y: ndarray. one-hot coded class labels. shape=(mini-batch-size, Num output neurons)
        reg: float. regularization strength.

        Returns:
        -----------   
        grad_wts: ndarray. Weight gradient. shape=(Num features, C)
        grad_b: ndarray. Bias gradient. shape=(C,)

        NOTE:
        - Gradient is the same as ADALINE, except we average over mini-batch in both wts and bias.
        - NO FOR LOOPS!
        - Don't forget regularization!!!! (Weights only, not for bias)
        '''

        # print(net_act)
        batch_size = net_act.shape[0]
        errors = net_act - y #errors should be (mini-batch size, C) <--this seems right now
        grad_bias =  np.sum(errors, axis=0)/batch_size 
        grad_wts = ((1/batch_size) * (features.T @ errors)) + (reg * self.wts)

        return grad_wts, grad_bias

    def test_loss(self, wts, b, features, labels):
        ''' Tester method for net_in and loss
        '''
        self.wts = wts
        self.b = b

        net_in = self.net_in(features)
        print(f'net in shape={net_in.shape}, min={net_in.min()}, max={net_in.max()}')
        print('Should be\nnet in shape=(15, 10), min=0.6586955718394152, max=1.4084436090585783\n')

        net_act = self.activation(net_in)
        print(f'net act shape={net_act.shape}, min={net_act.min()}, max={net_act.max()}')
        print('Should be\nnet act shape=(15, 10), min=0.07413774498031761, max=0.14167423767472354\n')
        return self.loss(net_in, labels, 0), self.loss(net_in, labels, 0.5)

    def test_gradient(self, wts, b, features, labels, num_unique_classes, reg=0):
        ''' Tester method for gradient
        '''
        self.wts = wts
        self.b = b

        net_in = self.net_in(features)
        print(f'net in: {net_in.shape}, {net_in.min()}, {net_in.max()}')

        net_act = self.activation(net_in)
        print("len(labels)",len(labels))
        labels_one_hot = self.one_hot(labels, num_unique_classes)
        print(f'y one hot: {labels_one_hot.shape}, sum is {np.sum(labels_one_hot)}')

        return self.gradient(features, net_act, labels_one_hot, reg=reg)
