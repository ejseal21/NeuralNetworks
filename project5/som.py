'''som.py
2D self-organizing map
CS343: Neural Networks
YOUR NAMES HERE
Project 5: Word Embeddings and SOMs
'''
import numpy as np


def lin2sub(ind, the_shape):
    '''Utility function that takes a linear index and converts it to subscripts.
    No changes necessary here.

    Parameters:
    ----------
    ind: int. Linear index to convert to subscripts
    the_shape: tuple. Shape of the ndarray from which `ind` was taken.

    Returns:
    ----------
    tuple of subscripts

    Example: ind=2, the_shape=(2,2) -> return (1, 0).
        i.e. [[_, _], [->SUBSCRIPT OF THIS ELEMENT<-, _]]
    '''
    return np.unravel_index(ind, the_shape)


class SOM:
    '''A 2D self-organzing map (SOM) neural network.
    '''
    def __init__(self, map_sz, n_features, max_iter, init_lr=0.2, init_sigma=10.0, verbose=False):
        '''Creates a new SOM with random weights in range [-1, 1]

        Parameters:
        ----------
        map_sz: int. Number of units in each dimension of the SOM.
            e.g. map_sz=9 -> the SOM will have 9x9=81 units arranged in a 9x9 grid
            n_features: int. Number of features in a SINGLE data sample feature vector.
        max_iter: int. Number of training iterations to do
        init_lr: float. INITIAL learning rate during learning. This will decay with time
            (iteration number). The effective learning rate will only equal this if t=0.
        init_sigma: float. INITIAL standard deviation of Gaussian neighborhood in which units
            cooperate during learning. This will decay with time (iteration number).
            The effective learning rate will only equal this if t=0.
        verbose: boolean. Whether to print out debug information at various stages of the algorithm.
            NOTE: if verbose=False, nothing should print out when running methods.

        TODO:
        - Initialize weights (self.wts) to standard normal random values (mu=0, sigma=1)
            Shape=(map_sz, map_sz, n_features).
            Weights should be normalized so that the L^2 norm (Euclidean distance) of EACH som
            unit's weight vector is 1.0
        - Initialize self.bmu_neighborhood_x and self.bmu_neighborhood_y to EACH be a 2D grid of
        (x,y) index values (i.e. x,y positions in the 2D grid), respectively, in the range 0,...,map_sz-1.
        shape of self.bmu_neighborhood_x: (map_sz, map_sz)
        shape of self.bmu_neighborhood_y: (map_sz, map_sz)
        Together, cooresponding values at each position in each array is an ordered pair of SOM unit
        (x,y) positions.
        '''
        self.n_features = n_features
        self.max_iter = max_iter
        self.init_lr = init_lr
        self.init_sigma = init_sigma
        self.verbose = verbose

        pass

    def get_wts(self):
        '''Returns a COPY of the weight vector.

        No changes necessary here.
        '''
        return self.wts.copy()

    def compute_decayed_param(self, t, param):
        '''Takes a hyperparameter (e.g. lr, sigma) and applies a time-dependent decay function.

        Parameters:
        ----------
        t: int. Current (training) time step.
        param: float. Parameter (e.g. lr, sigma) whose value we will decay.

        Returns:
        ----------
        float. The decayed parameter at time t

        TODO:
        - See notebook for decay equation to implement
        '''
        pass

    def gaussian(self, bmu_rc, sigma, lr):
        '''Generates a normalized 2D Gaussian, weighted by the the current learning rate, centered
        on `bmu_rc`.

        Parameters:
        ----------
        bmu_rc: tuple. x,y coordinates in the SOM grid of current best-matching unit (BMU).
            NOTE: bmu_rc is arranged row,col, which is y,x.
        sigma: float. Standard deviation of the Gaussian at the current training iteration.
            The parameter passed in is already decayed.
        lr: float. Learning rate at the current training iteration.
            The parameter passed in is already decayed.

        Returns:
        ----------
        ndarray. shape=(map_sz, map_sz). 2D Gaussian, weighted by the the current learning rate.

        TODO:
        - Evaluate a Gaussian on a 2D grid with shape=(map_sz, map_sz) centered on `bmu_rc`.
        - Normalize so that the maximum value in the kernel is `lr`
        '''
        pass

    def fit(self, train_data):
        '''Main training method

        Parameters:
        ----------
        train_data: ndarray. shape=(N, n_features) for N data samples.

        TODO:
        - Shuffle a COPY of the data samples (don't modify the original data passed in).
        - On each training iteration, select a data vector.
            - Compute the BMU, then update the weights of the BMU and its neighbors.

        NOTE: If self.max_iter > N, and the current iter > N, cycle back around and do another
        pass thru each training sample.
        '''

        if self.verbose:
            print(f'Starting training...')

        # TRAINING CODE HERE

        if self.verbose:
            print(f'Finished training.')

    def get_bmu(self, input_vector):
        '''Compute the best matching unit (BMU) given an input data vector.
        Uses Euclidean distance (L2 norm) as the distance metric.

        Parameters:
        ----------
        input_vector: ndarray. shape=(n_features,). One data sample vector.

        Returns:
        ----------
        tuple of (x,y) position of the BMU in the SOM grid.

        TODO:
        - Find the unit with the closest weights to the data vector. Return its subscripted position.
        '''
        pass

    def update_wts(self, t, input_vector, bmu_rc):
        '''Applies the SOM update rule to change the BMU (and neighboring units') weights,
        bringing them all closer to the data vector (cooperative learning).


        Parameters:
        ----------
        t: int. Current training iteration.
        input_vector: ndarray. shape=(n_features,). One data sample.
        bmu_rc: tuple. BMU (x,y) position in the SOM grid.

        Returns:
        ----------
        None

        TODO:
        - Decay the learning rate and Gaussian neighborhood standard deviation parameters.
        - Apply the SOM weight update rule. See notebook for equation.
        '''
        pass

    def error(self, data):
        '''Computes the quantization error: total error incurred by approximating all data vectors
        with the weight vector of the BMU.

        Parameters:
        ----------
        data: ndarray. shape=(N, n_features) for N data samples.

        Returns:
        ----------
        float. Average error over N data vectors

        TODO:
        - Progressively average the Euclidean distance between each data vector
        and the BMU weight vector.
        '''
        pass

    def u_matrix(self):
        '''Compute U-matrix, the distance each SOM unit wt and that of its 8 local neighbors.

        Returns:
        ----------
        ndarray. shape=(map_sz, map_sz). Total Euclidan distance between each SOM unit
            and its 8 neighbors.

        TODO:
        - Compute the U-matrix
        - Normalize it so that the dynamic range of values span [0, 1]

        '''
        pass

    def get_nearest_wts(self, data):
        '''Find the nearest SOM wt vector to each of data sample vectors.

        Parameters:
        ----------
        data: ndarray. shape=(N, n_features) for N data samples.

        Returns:
        ----------
        ndarray. shape=(N, n_features). The most similar weight vector for each data sample vector.

        TODO:
        - Compute and return the array of closest wts vectors to each of the input vectors.
        '''
        pass
