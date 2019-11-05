def backward_netIn_to_prevLayer_netAct(self, d_upstream):
		'''Computes the dprev_net_act, d_wts, d_b gradients for a Dense layer.
		dprev_net_act is the gradient that gets us thru the current layer and TO the layer below.
		i.e. it will be the upstream gradient for the layer below the current one.

		Parameters:
		-----------
		d_upstream: Same shape as self.net_in (output of Dense backward_netAct_to_netIn()).
			shape=(mini_batch_sz, n_units)

		Returns:
		-----------
		dprev_net_act: gradient that gets us thru the current layer and TO the layer below.
			shape = (shape of self.input)
		d_wts: gradient of current layer's wts. shape=shape of self.wts = (n_prev_layer_units, units)
		d_b: gradient of current layer's bias. shape=(units,)

		NOTE:
		-----------
		Look back at your MLP project for inspiration.
			The rules/forms of equations are the same.
		Pay attention to shapes:
			You will need to do reshaping -before- computing one of the "d" variables,
			and -after- computing another of the "d" variables.
			Printing out shapes here when running the test code is super helpful.
			Shape errors will frequently show up at this backprop stage, one layer down.
		Regularize your wts
		'''

		dprev_net_act = d_upstream @ self.wts.T # shape = (self.input)

		# reshape input before computing d_wts
		x = np.reshape(self.input, dprev_net_act.shape)

		d_wts =  (d_upstream.T @ x).T
 
		d_b = np.sum(d_upstream, axis = 0)

		# reshaping to match self.input
		dprev_net_act = np.reshape(dprev_net_act, self.input.shape)

		return dprev_net_act, d_wts, d_b