'''som.py
2D self-organizing map
CS343: Neural Networks
Trisha and Alice
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
        self.map_sz = map_sz

        self.wts = np.random.normal(loc=0, scale=1, size=(map_sz, map_sz, n_features))
        self.wts = self.wts
        for i in range(self.wts.shape[0]):
            for j in range(self.wts.shape[1]):
                self.wts[i, j] = self.wts[i, j] / np.linalg.norm(self.wts[i, j])
        
        self.bmu_neighborhood_x, self.bmu_neighborhood_y = np.meshgrid(np.arange(map_sz), np.arange(map_sz))

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
        return param * np.exp(-t/(self.max_iter/2))

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
        gaussian = np.zeros((self.map_sz, self.map_sz))
        for i in range(self.map_sz):
            for j in range(self.map_sz):
                gaussian[i, j] = lr * np.exp(-((i - bmu_rc[0])**2 + (j - bmu_rc[1])**2)/(2 * (sigma**2)))
        return gaussian

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
        copy = train_data.copy()
        # np.random.shuffle(copy)        

        if self.verbose:
            print(f'Starting training...')
        j = 0
        for i in range(self.max_iter):
            vec = copy[i + j]
            bmu = self.get_bmu(vec)
            self.update_wts(i+j, vec, bmu)
            if self.max_iter > train_data.shape[0] and i + j >= train_data.shape[0] - 1:
                #sends indexing back to the start, but lets i stay at its current value
                j -= train_data.shape[0]

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
        min_dist = np.Inf
        cur_ind = (-1, -1)
        for i in range(self.map_sz):
            for j in range(self.map_sz):
                cur_dist = np.linalg.norm(input_vector - self.wts[i, j]) 
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    cur_ind = (i, j)
        if cur_ind == (-1, -1):
            print("your indices in get_bmu are (-1, -1), so you probably have something messed up.")
        return cur_ind
        

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
        lr = self.compute_decayed_param(t, self.init_lr)
        sigma = self.compute_decayed_param(t, self.init_sigma)
        gauss = self.gaussian(bmu_rc, sigma, lr)

        for i in range(self.wts.shape[0]):
            for j in range(self.wts.shape[1]):
                self.wts[i, j] = self.wts[i, j] +  gauss[i, j] * (input_vector - self.wts[i, j])


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
        nearest_wts = self.get_nearest_wts(data)
        error = 0
        for i in range(data.shape[0]):
            distance = 0
            for j in range(len(nearest_wts[i])):
                distance += np.square(nearest_wts[i][j] - data[i][j])
            error += np.sqrt(distance)
        return error/data.shape[0]


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
        u = np.zeros((self.map_sz, self.map_sz))  # initialize u-matrix
        #looping over both dimensions of map
        for i in range(self.map_sz):
            for j in range(self.map_sz):
                local_sum = 0
                #looping over neighbors
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        #dont worry about middle one
                        if k != 0 or l != 0:
                            if j + l >= 0 and j + l < self.map_sz and i + k >= 0 and i + k < self.map_sz:
                                #norm function calculates l2 distance between two vectors
                                #compare current weight with all 8 surrounding weights, add the distances
                                local_sum += np.linalg.norm(
                                    self.wts[j, i] - self.wts[j + l, i + k])
                #set the correct index of the u-matrix to be the local sum
                u[j, i] = local_sum

        #return normalized u-matrix
        return u / np.max(u)


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
        nearest_wts = np.zeros(data.shape)
        for i in range(nearest_wts.shape[0]):
            bmu = self.get_bmu(data[i])
            nearest_wts[i, :] = self.wts[bmu]

        return nearest_wts
