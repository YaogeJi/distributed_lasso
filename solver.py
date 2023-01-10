import numpy as np
import time
from utils import proj_l1ball as proj
from sklearn import linear_model
import scipy
import copy


class OnlineSolver(object):
    def __init__(self, generator, network, gamma, computation=1, communication=1, cta=True) -> None:
        assert generator.m == network.shape[0]                # specifically for online version
        self.generator = generator
        self.network = network
        self.gamma = gamma
        self.computation = computation
        self.communication = communication
        self.cta = cta    #this is a switch to decide whether communicate first or adapt first 
        self.theta = np.ones([self.generator.m] + list(self.generator.theta.shape))  # + means extending the list to a np tenser: m*d*1
        

    def fit(self):
        loss_matrix = []
        theta_matrix = []
        for i in range(int(len(self.generator)/self.computation)): # if self.computation=1, this ratio means the total batches being used
            if self.cta:
                ref_theta = self.communicate()
            else:
                ref_theta = self.theta
            for local_rounds in range(self.computation):
                batch = self.generator.sample()
                loss = self.step(batch, ref_theta)
                if i % 100 == 0:
                    print(i, loss)
            if not self.cta:
                self.theta = self.communicate()
            loss_matrix.append(loss)
            theta_matrix.append(copy.deepcopy(self.theta))
        return theta_matrix, loss_matrix
    def step(self, batch):
        raise NotImplementedError
    def communicate(self):
        theta = np.expand_dims(np.linalg.matrix_power(self.network, self.communication) @ self.theta.squeeze(axis=2), axis=2)
        return theta



class CTA(OnlineSolver):
    def __init__(self, generator, network, gamma, computation=1, communication=1) -> None:
        super().__init__(generator, network, gamma, computation, communication, cta=True)

    def step(self, batch, ref_theta):
        x, y = batch
        m, n, d = x.shape
        gamma = self.gamma()
        self.theta = ref_theta - gamma / n * x.transpose(0,2,1) @ (x @ self.theta - y)
        assert self.theta.shape == (m, d, 1)
        assert y.shape == (m, n, 1)
        assert x.shape == (m, n, d)
        return np.linalg.norm((x @ self.theta - y).reshape(-1,1), ord=2) ** 2 / (m * n)

class ATC(CTA):
    def __init__(self, generator, network, gamma, computation=1, communication=1) -> None:
        super().__init__(generator, network, gamma, computation, communication)
        self.cta = False


class DualAveraging(object):
    def __init__(self, generator, network, gamma, computation=1, communication=1, cta=True, prox=0) -> None:
        assert generator.m == network.shape[0]
        self.generator = generator
        self.network = network
        self.gamma = gamma
        self.computation = computation
        self.communication = communication
        self.cta = cta
        self.prox = prox
        self.theta = np.ones([self.generator.m] + list(self.generator.theta.shape))
        self.mu = np.ones([self.generator.m] + list(self.generator.theta.shape))
        self.prox_center = self.prox * np.ones_like(self.theta)
        self.radius = np.linalg.norm(self.generator.theta, ord=1) * 1.1
        self.iter = 0
        self.sample_count = 0

    def fit(self):
        loss_matrix = []
        theta_matrix = []
        mu_matrix = []
        for i in range(int(len(self.generator)/self.computation)):
            if self.cta:
                ref_mu = self.communicate()
            else:
               ref_mu = self.mu
            for local_rounds in range(self.computation):
                batch = self.generator.sample()
                loss = self.step(batch, ref_mu)
                if i % 100 == 0:
                    print(i, loss)
            if not self.cta:
                self.mu = self.communicate()
            loss_matrix.append(loss)
            theta_matrix.append(copy.deepcopy(self.theta))
            mu_matrix.append(copy.deepcopy(self.mu))  
        return theta_matrix, loss_matrix
    
    def communicate(self):
       # theta = np.expand_dims(np.linalg.matrix_power(self.network, self.communication) @ self.theta.squeeze(axis=2), axis=2)
        mu = np.expand_dims(np.linalg.matrix_power(self.network, self.communication) @ self.mu.squeeze(axis=2), axis=2)
        return mu

    def step(self, batch, ref_mu):
        x, y = batch
        m, n, d = x.shape
        self.sample_count += n
        self.iter += 1
        gamma = self.gamma() * self.radius / np.sqrt(self.iter)
        p = 2*np.log(d)/(2*np.log(d)-1)
        q = p / (p-1)
        lmda = 4*self.generator.noise_dev * np.sqrt(np.log(d)/self.iter)
        gradient = 1 / n * x.transpose(0,2,1) @ (x @ self.theta - y)  #if either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
        self.mu = ref_mu + gradient + lmda * np.sign(self.theta)
        xi = np.clip((p - 1) * gamma * np.linalg.norm(self.mu.squeeze(axis=2), ord=q, axis=1, keepdims=True) * self.radius - 1, 0, None)
        self.theta = np.expand_dims(self.prox_center.squeeze(axis=2) + (self.radius * (p-1) * gamma)/(1 + xi) * np.abs(self.mu.squeeze(axis=2)) ** (q-1) * np.sign(self.mu.squeeze(axis=2)) * np.linalg.norm(self.mu.squeeze(axis=2), ord=q, axis=1, keepdims=True) ** (2 - q), axis=2)
        return np.linalg.norm((x @ self.theta - y).reshape(-1,1), ord=2) ** 2 / (m * n)
