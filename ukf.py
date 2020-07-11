import numpy as np
import scipy.linalg
from copy import deepcopy
from threading import Lock


class UKF:
	def __init__(self, num_states, process_noise, initial_state, initial_covar, alpha, k, beta, iterate_function, states_arr, sigmas_arr, win):

		self.n_dim = int(num_states)
		self.n_sig = 1 + num_states * 2
		self.Q = process_noise
		self.x = initial_state
		self.p = initial_covar
		self.beta = beta
		self.alpha = alpha
		self.k = k
		self.iterate = iterate_function
		self.states_store = states_arr
		self.sigmas_store = sigmas_arr
		self.lambd = pow(self.alpha, 2) * (self.n_dim + self.k) - self.n_dim

		self.covar_weights = np.full(self.n_sig, 1 / (2 * (self.n_dim + self.lambd)))
		self.mean_weights = np.full(self.n_sig, 1 / (2 * (self.n_dim + self.lambd)))
		self.W = win
		self.covar_weights[0] = (self.lambd / (self.n_dim + self.lambd)) + (1 - pow(self.alpha, 2) + self.beta)
		self.mean_weights[0] = (self.lambd / (self.n_dim + self.lambd))
		self.q = np.empty((1, 15))
		self.sigmas = self.__get_sigmas()

		self.lock = Lock()

	#function to calculate sigma points
	def __get_sigmas(self):

		ret = np.zeros((self.n_sig, self.n_dim))
		U = scipy.linalg.cholesky((self.n_dim + self.lambd) * self.p)

		ret[0] = self.x

		for k in range(self.n_dim):
			ret[k + 1] = self.x + U[k]
			ret[self.n_dim + k + 1] = self.x - U[k]

		return ret.T


	def update(self, states, data, r_matrix):

		diff_sum = 0
		self.lock.acquire()

		num_states = len(states)

		# create y, sigmas of just the states that are being updated
		sigmas_split = np.split(self.sigmas, self.n_dim)
		y = np.concatenate([sigmas_split[i] for i in states])

		# create y_mean, the mean of just the states that are being updated
		x_split = np.split(self.x, self.n_dim)
		y_mean = np.concatenate([x_split[i] for i in states])

		# differences in y from y mean
		y_diff = deepcopy(y)
		x_diff = deepcopy(self.sigmas)
		for i in range(self.n_sig):
			for j in range(num_states):
				y_diff[j][i] -= y_mean[j]
			for j in range(self.n_dim):
				x_diff[j][i] -= self.x[j]

		# covariance of measurement
		p_yy = np.zeros((num_states, num_states))
		for i, val in enumerate(np.array_split(y_diff, self.n_sig, 1)):
			p_yy += self.covar_weights[i] * val.dot(val.T)

		# add measurement noise
		p_yy += r_matrix

		# covariance of measurement with states
		p_xy = np.zeros((self.n_dim, num_states))
		for i, val in enumerate(zip(np.array_split(y_diff, self.n_sig, 1), np.array_split(x_diff, self.n_sig, 1))):
			p_xy += self.covar_weights[i] * val[1].dot(val[0].T)

		k = np.dot(p_xy, np.linalg.inv(p_yy))

		y_actual = data

		self.x += np.dot(k, (y_actual - y_mean))
		self.p -= np.dot(k, np.dot(p_yy, k.T))
		self.sigmas = self.__get_sigmas()

		self.states_store = np.append(self.states_store, np.atleast_3d(self.x), axis = 2)
		self.sigmas_store = np.append(self.sigmas_store, np.atleast_3d(self.sigmas), axis = 2)

		if self.states_store.shape[2] >= 11:

			for z in range(self.states_store.shape[2] - 1, self.states_store.shape[2] - 11, -1):

				sigmas_out = np.array([self.iterate(x, 0.01) for x in self.sigmas_store[:, :, (z - 1)].T]).T

				x_out = np.zeros(self.n_dim)

				for i in range(self.n_dim):
					x_out[i] = sum((self.mean_weights[j] * sigmas_out[i][j] for j in range(self.n_sig)))

				diff_sum += self.states_store[:, :, z] - np.atleast_2d(x_out)

			self.q = diff_sum / self.W








		self.lock.release()

		return self.x

	
	def predict(self, timestep):

		self.lock.acquire()

		#send each sigma points through f()
		sigmas_out = np.array([self.iterate(x, timestep) for x in self.sigmas.T]).T 

		x_out = np.zeros(self.n_dim)

		for i in range(self.n_dim):
			x_out[i] = sum((self.mean_weights[j] * sigmas_out[i][j] for j in range(self.n_sig)))

		p_out = np.zeros((self.n_dim, self.n_dim))

		for i in range(self.n_sig):
			diff = sigmas_out.T[i] - x_out
			diff = np.atleast_2d(diff)
			p_out += self.covar_weights[i] * np.dot(diff.T, diff)

		p_out += timestep * self.Q   #adding the process covariance 

		self.sigmas = sigmas_out
		self.x = x_out
		self.p = p_out

		self.lock.release()



	def get_state(self, index = -1):

		if index >= 0:
			return self.x[index]
		else:
			return self.x

	def get_covar(self):
		return self.p

	def set_state(self, value, index = -1):

		with self.lock:
			if index != -1:
				self.x[index] = value

			else:
				self.x = value

	def reset(self, state, covar):

		with self.lock:
			self.x = state
			self.p =covar





